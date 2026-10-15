"""Deterministic source-mixed, rank-partitioned sequence batches."""

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from scratchcoder.data.shards import TokenShard


class Documents(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Sequence[int]: ...


class BatchSource(Protocol):
    def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def source_identity(source: Documents) -> str:
    if isinstance(source, TokenShard):
        return source.identity
    digest = hashlib.sha256()
    for index in range(len(source)):
        digest.update(json.dumps(list(source[index]), separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


class MixedLoader:
    def __init__(
        self,
        sources: Mapping[str, Documents],
        shares: dict[str, float],
        *,
        sequence_length: int,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if sequence_length < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid loader dimensions/rank")
        if (
            set(sources) != set(shares)
            or not shares
            or any(not 0 < x <= 1 for x in shares.values())
            or abs(sum(shares.values()) - 1) > 1e-8
        ):
            raise ValueError("source shares must be positive, sum to one, and match sources")
        self.sources = sources
        self.names = sorted(sources)
        self.shares = shares
        self.length = sequence_length
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices = {
            name: list(range(rank, len(sources[name]), world_size)) for name in self.names
        }
        if any(not indices for indices in self.indices.values()):
            raise ValueError("each rank needs at least one document per source")
        for indices in self.indices.values():
            self.rng.shuffle(indices)
        self.positions = dict.fromkeys(self.names, 0)
        self.buffers: dict[str, list[int]] = {name: [] for name in self.names}
        signature = {
            "length": sequence_length,
            "batch": batch_size,
            "seed": seed,
            "rank": rank,
            "world": world_size,
            "shares": shares,
            "sizes": {name: len(source) for name, source in sources.items()},
            "identities": {name: source_identity(source) for name, source in sources.items()},
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        used_sources: list[str] = []
        for _ in range(self.batch_size):
            name = self.rng.choices(self.names, [self.shares[item] for item in self.names])[0]
            buffer = self.buffers[name]
            while len(buffer) < self.length + 1:
                indices = self.indices[name]
                index = indices[self.positions[name] % len(indices)]
                document = list(self.sources[name][index])
                if not document:
                    raise ValueError("loader received an empty document")
                buffer.extend(document)
                self.positions[name] += 1
            inputs.append(buffer[: self.length])
            targets.append(buffer[1 : self.length + 1])
            del buffer[: self.length]
            used_sources.append(name)
        return inputs, targets, used_sources

    def state_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "rng": self.rng.getstate(),
            "positions": dict(self.positions),
            "buffers": {name: list(items) for name, items in self.buffers.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["signature"] != self.signature:
            raise ValueError("loader configuration differs from saved state")
        rng = state["rng"]
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
        self.positions = dict(state["positions"])
        self.buffers = {name: list(items) for name, items in state["buffers"].items()}
