"""Fixed held-out windows and byte-accurate per-domain validation."""

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import batched

import torch
from torch import Tensor

from scratchcoder.data.shards import TokenShard
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import cross_entropy
from scratchcoder.tokenizer.bpe import ByteBPE


@dataclass(frozen=True)
class ValidationWindow:
    tokens: tuple[int, ...]
    target_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokens, tuple)
            or len(self.tokens) < 2
            or any(type(token) is not int or token < 0 for token in self.tokens)
        ):
            raise ValueError("validation needs at least two nonnegative token IDs")
        if type(self.target_bytes) is not int or self.target_bytes <= 0:
            raise ValueError("validation needs a positive raw target byte count")


def shard_windows(
    shard: TokenShard, tokenizer: ByteBPE, *, documents: int, sequence_length: int
) -> tuple[ValidationWindow, ...]:
    if documents < 1 or sequence_length < 1:
        raise ValueError("positive validation window limits required")
    byte_tokens = tokenizer.byte_tokens()
    windows = []
    for index in range(min(documents, len(shard))):
        tokens = tuple(shard[index][: sequence_length + 1])
        count = sum(len(byte_tokens.get(token, b"")) for token in tokens[1:])
        windows.append(ValidationWindow(tokens, count))
    if not windows:
        raise ValueError("validation shard is empty")
    return tuple(windows)


def iter_document_windows(
    shard: TokenShard, tokenizer: ByteBPE, *, sequence_length: int, target_tokens: int | None
) -> Iterator[ValidationWindow]:
    """Every document in shard order, cut into windows of at most ``L + 1`` tokens.

    Windows start at positions ``0, L, 2L, ...`` of each document, so each token after
    the document's BOS is a target exactly once and no window crosses a document
    boundary. With ``target_tokens`` the stream stops once exactly that many targets
    have been yielded, truncating the last window. A window whose targets are all
    special tokens has no raw bytes, so it is skipped and its targets are not counted.
    Documents are read one at a time and nothing is retained between windows. Limits
    are checked on the call; a shard too short for ``target_tokens`` is only detected
    when the stream reaches its end.
    """
    if (
        type(sequence_length) is not int
        or sequence_length < 1
        or target_tokens is not None
        and (type(target_tokens) is not int or target_tokens < 1)
    ):
        raise ValueError("positive validation window limits required")
    if shard.vocab_size != tokenizer.vocab_size:
        raise ValueError("validation shard vocabulary differs from tokenizer")
    byte_tokens = tokenizer.byte_tokens()
    sizes = [len(byte_tokens.get(token, b"")) for token in range(shard.vocab_size)]
    return _stream_windows(shard, sizes, sequence_length, target_tokens)


def _stream_windows(
    shard: TokenShard, sizes: list[int], sequence_length: int, target_tokens: int | None
) -> Iterator[ValidationWindow]:
    remaining = target_tokens
    for index in range(len(shard)):
        document = shard[index]
        for start in range(0, len(document) - 1, sequence_length):
            stop = start + sequence_length + 1
            if remaining is not None:
                stop = min(stop, start + remaining + 1)
            tokens = tuple(document[start:stop])
            count = sum(sizes[token] for token in tokens[1:])
            if count == 0:
                continue
            yield ValidationWindow(tokens, count)
            if remaining is not None:
                remaining -= len(tokens) - 1
                if remaining == 0:
                    return
    if remaining is not None:
        raise ValueError("validation shard has fewer targets than target_tokens")


def document_windows(
    shard: TokenShard, tokenizer: ByteBPE, *, sequence_length: int, target_tokens: int | None
) -> tuple[ValidationWindow, ...]:
    """``iter_document_windows`` held in memory, about 40 bytes per token in CPython.

    For a full validation set of millions of tokens, stream ``iter_document_windows``
    into ``validate_batched`` instead.
    """
    windows = tuple(
        iter_document_windows(
            shard, tokenizer, sequence_length=sequence_length, target_tokens=target_tokens
        )
    )
    if not windows:
        raise ValueError("validation shard is empty")
    return windows


def selection_sha256(windows: Iterable[ValidationWindow]) -> str:
    """SHA-256 of the windows' tokens and byte counts in order, computed incrementally.

    The digest equals ``sha256(json.dumps([asdict(w) for w in windows], sort_keys=True))``,
    the held-out identity the pilot records, without building the whole JSON string.
    """
    digest = hashlib.sha256(b"[")
    for position, window in enumerate(windows):
        if position:
            digest.update(b", ")
        record = {"target_bytes": window.target_bytes, "tokens": window.tokens}
        digest.update(json.dumps(record, sort_keys=True).encode())
    digest.update(b"]")
    return digest.hexdigest()


@torch.no_grad()
def validate(model: GPT, windows: tuple[ValidationWindow, ...]) -> dict[str, float]:
    if not windows:
        raise ValueError("validation windows are empty")
    device = next(model.parameters()).device
    was_training = model.training
    total_nll = 0.0
    tokens = raw_bytes = 0
    model.eval()
    try:
        for window in windows:
            if (
                len(window.tokens) - 1 > model.config.context
                or max(window.tokens) >= model.config.vocab_size
            ):
                raise ValueError("validation window exceeds model context/vocabulary")
            ids = torch.tensor([window.tokens], dtype=torch.long, device=device)
            count = len(window.tokens) - 1
            nll = float(cross_entropy(model(ids[:, :-1]), ids[:, 1:])) * count
            if not math.isfinite(nll):
                raise FloatingPointError("nonfinite validation loss")
            total_nll += nll
            tokens += count
            raw_bytes += window.target_bytes
    finally:
        model.train(was_training)
    return {
        "loss": total_nll / tokens,
        "bits_per_byte": total_nll / math.log(2) / raw_bytes,
        "tokens": float(tokens),
        "bytes": float(raw_bytes),
    }


def _masked_nll_sum(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Summed target negative log likelihood over unmasked positions, in float64.

    Per-token values use the same float32 log-softmax gather as ``cross_entropy``.
    Padded positions are replaced by zero before the sum rather than multiplied by it,
    so a nonfinite value there cannot leak into the total.
    """
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("logits/target/mask shape mismatch")
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    chosen = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -chosen.masked_fill(~mask, 0.0).double().sum()


@torch.no_grad()
def validate_batched(
    model: GPT,
    windows: Iterable[ValidationWindow],
    *,
    batch_size: int = 16,
    autocast_bf16: bool = False,
    on_batch: Callable[[], None] | None = None,
) -> dict[str, float]:
    """``validate`` over right-padded batches; ``windows`` may be a lazy iterable.

    Rows are padded to the longest window in their batch and padded targets are masked
    out. Padding follows every real position, so causal attention keeps it from changing
    any scored logit. ``on_batch`` runs after each batch, for example to refresh a
    heartbeat during a long full-validation pass.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("validation batch size must be a positive integer")
    if type(autocast_bf16) is not bool:
        raise ValueError("autocast_bf16 must be a boolean")
    device = next(model.parameters()).device
    was_training = model.training
    total_nll = 0.0
    tokens = raw_bytes = 0
    model.eval()
    try:
        for batch in batched(windows, batch_size):
            if any(
                len(window.tokens) - 1 > model.config.context
                or max(window.tokens) >= model.config.vocab_size
                for window in batch
            ):
                raise ValueError("validation window exceeds model context/vocabulary")
            width = max(len(window.tokens) for window in batch)
            ids = torch.tensor(
                [list(window.tokens) + [0] * (width - len(window.tokens)) for window in batch],
                dtype=torch.long,
                device=device,
            )
            counts = [len(window.tokens) - 1 for window in batch]
            lengths = torch.tensor(counts, device=device)
            mask = torch.arange(width - 1, device=device)[None, :] < lengths[:, None]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_bf16
            ):
                logits = model(ids[:, :-1])
            nll = float(_masked_nll_sum(logits, ids[:, 1:], mask))
            if not math.isfinite(nll):
                raise FloatingPointError("nonfinite validation loss")
            total_nll += nll
            tokens += sum(counts)
            raw_bytes += sum(window.target_bytes for window in batch)
            if on_batch is not None:
                on_batch()
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("validation windows are empty")
    return {
        "loss": total_nll / tokens,
        "bits_per_byte": total_nll / math.log(2) / raw_bytes,
        "tokens": float(tokens),
        "bytes": float(raw_bytes),
    }
