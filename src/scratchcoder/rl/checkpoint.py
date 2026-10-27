"""Weights-only checkpoints for RL and the more-SFT control.

RL and the control each start a fresh optimizer and never resume one, so a dev check
saves only the model weights with their identity. The file is a ``torch.save`` mapping::

    {"format": "weights-v1", "config": {"model": {...}}, "tokenizer_hash": ...,
     "run_id": ..., "step": ..., "metadata": {...}, "model": state_dict}

It loads with ``weights_only=True``, and it carries the keys the trainer reads from a
parent checkpoint (``config.model``, ``tokenizer_hash``, ``run_id``, ``model``) and the
keys ``scratchcoder.eval.checkpoint.load_checkpoint`` reads, so a selected RL or control
checkpoint can be evaluated or trained from like any other. The write is atomic: a
temporary file in the same directory, fsync, rename, then fsync of the directory.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import GPT

WEIGHTS_FORMAT = "weights-v1"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_weights(
    path: Path,
    model: GPT,
    *,
    tokenizer_hash: str,
    run_id: str,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write the model's weights atomically to ``path``; return the file's SHA-256.

    Tensors are copied to the CPU unchanged (same dtype, same values), so loading them back
    reproduces the weights bit for bit.
    """
    if not tokenizer_hash or not run_id:
        raise ValueError("a weights checkpoint needs tokenizer and run identities")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    state = {
        "format": WEIGHTS_FORMAT,
        "config": {"model": asdict(model.config)},
        "tokenizer_hash": tokenizer_hash,
        "run_id": run_id,
        "step": step,
        "metadata": dict(metadata or {}),
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return file_sha256(path)


def read_weights(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load a weights checkpoint (tensors only), rejecting altered bytes first."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("checkpoint bytes differ from the expected identity")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not {"config", "tokenizer_hash", "model"} <= set(state):
        raise ValueError("not a model checkpoint")
    return state


@dataclass(frozen=True)
class WeightsComparison:
    """Whether two checkpoints hold the same model weights, tensor by tensor."""

    same: bool
    tensors: int
    """How many tensors were compared (those of the first checkpoint)."""
    first_difference: str | None
    """The first tensor name, in sorted order, that is missing, shaped, typed or valued
    differently; ``None`` when the weights are the same."""


def _tensor_bytes(value: torch.Tensor) -> torch.Tensor:
    """A tensor's raw bytes as a flat ``uint8`` view: NaN, -0.0 and every bit pattern compare
    as stored, which ``torch.equal`` on the values would not."""
    return value.detach().reshape(-1).contiguous().view(torch.uint8)


def compare_weights(first: Path, second: Path) -> WeightsComparison:
    """Compare the model tensors of two checkpoints bit for bit, on the CPU.

    The same weights saved twice are two files with two SHA-256s (each file also records its
    run ID, step and metadata), so a file hash cannot say whether two endpoints are one model.
    This loads both (``read_weights``) and compares every tensor's name, dtype, shape and
    bytes. It reads both files whole: about 1.1 GB of memory for two 541 MB S checkpoints."""
    return _compare(read_weights(first)["model"], read_weights(second)["model"])


def compare_to_loaded(path: Path, reference: Path) -> WeightsComparison:
    """Compare the model tensors of ``path`` bit for bit with ``reference`` as ``load_weights``
    loads it, on the CPU: the weights a run that starts from ``reference`` holds.

    For a checkpoint with one floating dtype that is the file's own tensors. For one with
    several, the model keeps its own dtypes and ``load_state_dict`` casts into them, so a
    check saved from the loaded model differs from the file in dtype yet holds the weights
    the run started from; this compares against those, as the run 2 same-weights rule asks
    (the weights the run loaded, not the file)."""
    loaded = load_weights(reference).state_dict()
    return _compare(read_weights(path)["model"], loaded)


def _compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> WeightsComparison:
    for name in sorted(set(a) | set(b)):
        x, y = a.get(name), b.get(name)
        if (
            x is None
            or y is None
            or x.dtype != y.dtype
            or x.shape != y.shape
            or not torch.equal(_tensor_bytes(x), _tensor_bytes(y))
        ):
            return WeightsComparison(False, len(a), name)
    return WeightsComparison(True, len(a), None)


def load_weights(
    path: Path,
    *,
    tokenizer_hash: str | None = None,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> GPT:
    """A model built from a checkpoint's recorded shape with its weights loaded.

    The model takes the saved floating dtype first, so loading never rounds a weight.
    Works for weights-only checkpoints and the trainer's full checkpoints alike.
    """
    state = read_weights(path, expected_sha256=expected_sha256)
    if tokenizer_hash is not None and state["tokenizer_hash"] != tokenizer_hash:
        raise ValueError("checkpoint tokenizer differs")
    model = GPT(ModelConfig(**state["config"]["model"]))
    dtypes = {value.dtype for value in state["model"].values() if value.is_floating_point()}
    if len(dtypes) == 1:
        model.to(dtypes.pop())
    model.load_state_dict(state["model"])
    return model.to(device)
