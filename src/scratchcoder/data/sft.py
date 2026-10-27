"""SFT rows and their batch source: one chat-v1 conversation per padded row.

``SFTLoader`` reads each kept row once per pass, in an order anyone can recompute:
pass ``p`` sorts the rows by the first eight bytes of ``sha256(f"{seed}|{p}|{row id}")``,
read as a big-endian integer, with the row ID breaking ties. Every row becomes one
``[L]`` input row and one ``[L]`` target row: inputs are ``ids[:-1]`` and targets
``ids[1:]``, with targets outside the assistant mask and all padding set to
``IGNORE``. Padded input positions hold ``pad_id`` and follow every real position, so
causal attention keeps them from changing a scored logit. Rows after the last
conversation, up to the end of the final update, are entirely padding, so every
microbatch has the same ``[B, L]`` shape. A row's loss tokens (``loss_tokens``) are
the targets it contributes that are not ``IGNORE``: the count the trainer divides by
under ``update_targets``. See ``docs/walkthroughs/sft-batching.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.chat import BOS, Rendered
from scratchcoder.data.onepass import IGNORE


@dataclass(frozen=True)
class SFTRow:
    """One rendered conversation with its identity and source label."""

    id: str
    source: str
    ids: tuple[int, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or not isinstance(self.source, str):
            raise ValueError("an SFT row needs a nonempty ID and a source label")
        if not self.source:
            raise ValueError("an SFT row source label must be nonempty")
        Rendered(self.ids, self.mask)  # The same shape checks as a rendered conversation.
        if any(type(token) is not int or token < 0 for token in self.ids):
            raise ValueError("SFT token IDs must be nonnegative integers")

    @classmethod
    def of(cls, identifier: str, source: str, rendered: Rendered) -> SFTRow:
        return cls(identifier, source, rendered.ids, rendered.mask)

    @property
    def assistant_tokens(self) -> int:
        return sum(self.mask)

    def sha256(self) -> str:
        """SHA-256 of the row's canonical JSON (ID, source, IDs and loss spans)."""
        return hashlib.sha256(
            json.dumps(row_json(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def loss_tokens(row: SFTRow) -> int:
    """The row's loss tokens: its targets ``ids[1:]`` that fall under ``mask[1:]``.

    These are exactly the row's targets that the loader emits as something other than
    ``IGNORE`` (token IDs are nonnegative), which the trainer counts as real targets.
    The first token is never a target, so this also equals ``sum(mask)``. The SFT data
    build caps and records the language share with this count, and the loader's
    ``total_targets`` and tickets use it too.
    """
    return sum(row.mask[1:])


def loss_spans(mask: Sequence[bool]) -> list[list[int]]:
    """Half-open ``[start, end)`` index ranges where the mask is true."""
    spans: list[list[int]] = []
    for index, keep in enumerate(mask):
        if keep and (not spans or spans[-1][1] != index):
            spans.append([index, index + 1])
        elif keep:
            spans[-1][1] = index + 1
    return spans


def spans_mask(spans: Sequence[Sequence[int]], length: int) -> tuple[bool, ...]:
    mask = [False] * length
    previous = 0
    for span in spans:
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not previous <= span[0] < span[1] <= length
        ):
            raise ValueError("loss spans must be ordered, disjoint [start, end) ranges")
        mask[span[0] : span[1]] = [True] * (span[1] - span[0])
        previous = span[1]
    return tuple(mask)


def row_json(row: SFTRow) -> dict[str, Any]:
    return {"id": row.id, "source": row.source, "ids": list(row.ids), "spans": loss_spans(row.mask)}


def row_from_json(raw: Mapping[str, Any]) -> SFTRow:
    if not isinstance(raw, Mapping) or set(raw) != {"id", "source", "ids", "spans"}:
        raise ValueError("an SFT row has exactly the keys id, source, ids and spans")
    ids = raw["ids"]
    if not isinstance(ids, list):
        raise ValueError("SFT row ids must be a list")
    return SFTRow(raw["id"], raw["source"], tuple(ids), spans_mask(raw["spans"], len(ids)))


def write_rows(path: Path, rows: Iterable[SFTRow]) -> None:
    """One JSON row per line; refuses to overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row_json(row), sort_keys=True, separators=(",", ":")) + "\n")


def read_rows(path: Path) -> Iterator[SFTRow]:
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = row_from_json(json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid SFT row on line {number}") from error
            if row.id in seen:
                raise ValueError(f"duplicate SFT row id: {row.id}")
            seen.add(row.id)
            yield row


def pass_key(seed: int, epoch: int, identifier: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}|{epoch}|{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big"), identifier


def fits(row: SFTRow, sequence_length: int) -> bool:
    """A row of ``n`` tokens has ``n - 1`` input/target pairs; they must fit ``L``."""
    return len(row.ids) - 1 <= sequence_length


class SFTLoader:
    """Every kept row once per pass, one per ``[L]`` row, in a seeded order.

    A row is dropped and counted if its ``n - 1`` input/target pairs exceed ``L``
    (``over_length``) or none of its targets carries loss (``no_assistant_targets``).
    """

    def __init__(
        self,
        rows: Sequence[SFTRow],
        *,
        sequence_length: int,
        batch_size: int,
        accumulation_steps: int,
        seed: int = 0,
        passes: int = 1,
        pad_id: int = BOS,
    ) -> None:
        for label, value in (
            ("sequence_length", sequence_length),
            ("batch_size", batch_size),
            ("accumulation_steps", accumulation_steps),
            ("passes", passes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if type(seed) is not int or seed < 0 or type(pad_id) is not int or pad_id < 0:
            raise ValueError("the seed and pad ID must be nonnegative integers")
        if len({row.id for row in rows}) != len(rows):
            raise ValueError("SFT row IDs must be unique")
        dropped: Counter[str] = Counter()
        kept: list[SFTRow] = []
        for row in rows:
            if not fits(row, sequence_length):
                dropped["over_length"] += 1
            elif not loss_tokens(row):
                dropped["no_assistant_targets"] += 1
            else:
                kept.append(row)
        if not kept:
            raise ValueError("an SFT loader needs at least one row that fits")
        self.dropped = dict(sorted(dropped.items()))
        self.length = sequence_length
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.pad_id = pad_id
        self._rows = kept
        position = {row.id: index for index, row in enumerate(kept)}
        self._order = [
            position[row.id]
            for epoch in range(passes)
            for row in sorted(kept, key=lambda item: pass_key(seed, epoch, item.id))
        ]
        self.rows_per_update = batch_size * accumulation_steps
        self.planned_updates = -(-len(self._order) // self.rows_per_update)
        self.planned_rows = self.planned_updates * self.rows_per_update
        self.total_targets = passes * sum(loss_tokens(row) for row in kept)
        self.order_sha256 = hashlib.sha256(
            "\n".join(kept[index].id for index in self._order).encode()
        ).hexdigest()
        signature = {
            "format": "sft-1",
            "length": sequence_length,
            "batch": batch_size,
            "accumulation": accumulation_steps,
            "seed": seed,
            "passes": passes,
            "pad": pad_id,
            "rows": hashlib.sha256(
                "\n".join(sorted(f"{row.id}|{row.sha256()}" for row in rows)).encode()
            ).hexdigest(),
            "dropped": self.dropped,
            "order": self.order_sha256,
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self._emitted = 0

    @property
    def exhausted(self) -> bool:
        return self._emitted == self.planned_rows

    def reading_order(self) -> list[SFTRow]:
        """The kept rows in the order they are emitted, every pass included."""
        return [self._rows[index] for index in self._order]

    def _row(self, row: SFTRow) -> tuple[list[int], list[int]]:
        inputs, targets = Rendered(row.ids, row.mask).inputs_targets()
        padding = self.length - len(inputs)
        return inputs + [self.pad_id] * padding, targets + [IGNORE] * padding

    def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
        """One ``batch_size x L`` microbatch; padded rows have the source ``""``."""
        if self.exhausted:
            raise RuntimeError(
                f"SFT loader exhausted: all {self.planned_updates} planned updates "
                f"({self.planned_rows} rows) were already emitted"
            )
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        sources: list[str] = []
        for position in range(self._emitted, self._emitted + self.batch_size):
            if position < len(self._order):
                row = self._rows[self._order[position]]
                row_inputs, row_targets = self._row(row)
                sources.append(row.source)
            else:
                row_inputs, row_targets = [self.pad_id] * self.length, [IGNORE] * self.length
                sources.append("")
            inputs.append(row_inputs)
            targets.append(row_targets)
        self._emitted += self.batch_size
        return inputs, targets, sources

    def state_dict(self) -> dict[str, Any]:
        return {"signature": self.signature, "rows": self._emitted}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != {"signature", "rows"}:
            raise ValueError("invalid SFT loader state fields")
        if state["signature"] != self.signature:
            raise ValueError("loader configuration differs from saved state")
        rows = state["rows"]
        if type(rows) is not int or not 0 <= rows <= self.planned_rows or rows % self.batch_size:
            raise ValueError("SFT loader rows must be a multiple of the batch within the plan")
        self._emitted = rows

    def ticket(self) -> dict[str, Any]:
        """What has been read so far; ``complete`` is true only once exhausted."""
        read = self._order[: self._emitted]
        per_source: dict[str, dict[str, int]] = {}
        for index in read:
            row = self._rows[index]
            entry = per_source.setdefault(row.source, {"rows": 0, "targets": 0})
            entry["rows"] += 1
            entry["targets"] += loss_tokens(row)
        return {
            "complete": self.exhausted,
            "rows": len(read),
            "targets": sum(entry["targets"] for entry in per_source.values()),
            "updates": self._emitted // self.rows_per_update,
            "order_sha256": self.order_sha256,
            "dropped": dict(self.dropped),
            "per_source": dict(sorted(per_source.items())),
        }
