"""Special-token rows of the tied embedding: a fresh start and their own learning rate.

Pretraining never shows the chat specials (``<|user_start|>`` through the reserved tokens,
IDs 257 to 271): as outputs they only ever receive the softmax's push-down, the same for
every unused token, so their tied embedding rows drift together toward one shared
direction (run B's parent: pairwise cosines of about 0.997). SFT then cannot tell
``<|assistant_end|>`` from ``<|python_start|>`` after ``<|assistant_start|>``, because with
a tied embedding the logit gap between two tokens is ``h . (e_i - e_j)`` and the difference
is nearly zero.

``reinitialize_special_rows`` gives each of those rows a fresh draw before SFT: the mean of
the ordinary rows (every row that is not a special) plus Gaussian noise whose per-dimension
standard deviation is the ordinary rows' own spread around that mean, from a seeded
generator on the CPU, so the result is the same on every device. A new row then looks
like a typical ordinary row in scale, and two new rows are no more alike than two
ordinary rows sharing that mean.

``row_scale`` is the per-element learning-rate multiplier for those rows (``[V, 1]``:
``SPECIAL_ROW_LR_SCALE`` on each special row, 1 elsewhere), which ``AdamW`` applies to the
embedding, a parameter without weight decay.

``<|bos|>`` (256) is left alone: pretraining shards begin every document with it, so its
row is trained as an input and as a target, and chat-v1 never makes it an SFT target (it
is the first token of every row and the padding). See
``docs/walkthroughs/special-token-rows.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from scratchcoder.tokenizer.bpe import FIRST_MERGE

BOS_ID = 256
SPECIAL_ROWS = tuple(range(BOS_ID + 1, FIRST_MERGE))
"""IDs 257 to 271: the chat-v1 specials (``<|user_start|>`` to ``<|python_end|>``), the
output markers and the seven reserved tokens. ``<|bos|>`` is not among them."""
SPECIAL_ROW_LR_SCALE = 10.0
"""The special rows train at ten times the base learning rate (run 1b's ruling)."""
INITIALIZATIONS = ("none", "ordinary-mean-noise")
METHOD = (
    "row = mean of the ordinary rows + noise, noise ~ N(0, s_d^2) per dimension d, s_d the "
    "ordinary rows' population standard deviation around that mean; drawn in float32 on the "
    "CPU from torch.Generator().manual_seed(seed), one row after another in ascending ID order"
)


def _rows(rows: Sequence[int], vocabulary: int) -> list[int]:
    chosen = list(rows)
    if (
        not chosen
        or any(type(row) is not int or not 0 <= row < vocabulary for row in chosen)
        or len(set(chosen)) != len(chosen)
        or chosen != sorted(chosen)
    ):
        raise ValueError("special rows must be distinct ascending IDs inside the vocabulary")
    if len(chosen) >= vocabulary:
        raise ValueError("the ordinary rows must not be empty")
    return chosen


def ordinary_mask(vocabulary: int, rows: Sequence[int] = SPECIAL_ROWS) -> Tensor:
    """True for every row that is not one of ``rows`` and not any special token."""
    mask = torch.ones(vocabulary, dtype=torch.bool)
    mask[BOS_ID : min(FIRST_MERGE, vocabulary)] = False
    mask[_rows(rows, vocabulary)] = False
    return mask


def _digest(values: Tensor) -> str:
    return hashlib.sha256(values.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def reinitialize_special_rows(
    weight: Tensor, *, seed: int, rows: Sequence[int] = SPECIAL_ROWS
) -> dict[str, Any]:
    """Overwrite ``rows`` of an embedding ``weight [V, D]`` in place; return the record.

    The statistics are taken over the ordinary rows (``ordinary_mask``) in float32; the
    noise comes from a CPU generator seeded with ``seed``, so the new rows are identical on
    every device and dtype round trip of the same weights.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("the embedding weight must be a [V, D] floating-point tensor")
    if type(seed) is not int or seed < 0:
        raise ValueError("the seed must be a nonnegative integer")
    vocabulary, width = weight.shape
    chosen = _rows(rows, vocabulary)
    source = weight.detach().float().cpu()
    ordinary = source[ordinary_mask(vocabulary, chosen)]
    mean = ordinary.mean(dim=0)
    spread = ordinary.std(dim=0, unbiased=False)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(chosen), width, generator=generator, dtype=torch.float32)
    fresh = mean + spread * noise
    before = source[chosen]
    weight[chosen] = fresh.to(device=weight.device, dtype=weight.dtype)
    unit = torch.nn.functional.normalize(fresh, dim=-1)
    cosines = unit @ unit.T
    off = cosines[~torch.eye(len(chosen), dtype=torch.bool)]
    return {
        "method": METHOD,
        "seed": seed,
        "rows": chosen,
        "ordinary_rows": int(ordinary.shape[0]),
        "mean_norm": float(mean.norm()),
        "mean_spread": float(spread.mean()),
        "before_sha256": _digest(before),
        "after_sha256": _digest(fresh),
        "max_cosine_after": float(off.max()) if off.numel() else None,
        "mean_cosine_after": float(off.mean()) if off.numel() else None,
    }


def row_scale(
    vocabulary: int, *, scale: float = SPECIAL_ROW_LR_SCALE, rows: Sequence[int] = SPECIAL_ROWS
) -> Tensor:
    """The ``[V, 1]`` learning-rate multiplier: ``scale`` on ``rows``, 1 elsewhere."""
    if type(vocabulary) is not int or vocabulary < 1:
        raise ValueError("the vocabulary must be a positive integer")
    if isinstance(scale, bool) or not isinstance(scale, int | float) or not 0 < scale < 1e6:
        raise ValueError("the learning-rate scale must be a positive finite number")
    multiplier = torch.ones(vocabulary, 1, dtype=torch.float32)
    multiplier[_rows(rows, vocabulary)] = float(scale)
    return multiplier
