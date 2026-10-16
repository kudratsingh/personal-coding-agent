"""Fixed held-out windows and byte-accurate per-domain validation."""

import math
from dataclasses import dataclass

import torch

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
