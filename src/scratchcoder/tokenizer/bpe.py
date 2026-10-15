"""Deterministic byte-level BPE with incremental weighted pair counts."""

import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
    *(f"<|reserved_{i}|>" for i in range(7)),
)
FIRST_MERGE = 256 + len(SPECIAL_TOKENS)
CHUNKS = re.compile(r"[^\S\n]+|\n+|[^\s]+")


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    output: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            output.append(new_id)
            i += 2
        else:
            output.append(tokens[i])
            i += 1
    return output


@dataclass(frozen=True)
class ByteBPE:
    merges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        known = set(range(256))
        seen: set[tuple[int, int]] = set()
        for index, pair in enumerate(self.merges):
            if (
                len(pair) != 2
                or any(type(value) is not int for value in pair)
                or pair[0] not in known
                or pair[1] not in known
                or pair in seen
            ):
                raise ValueError("invalid merge table")
            seen.add(pair)
            known.add(FIRST_MERGE + index)

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE + len(self.merges)

    @property
    def bos_id(self) -> int:
        return 256

    def byte_tokens(self) -> dict[int, bytes]:
        tokens = {i: bytes([i]) for i in range(256)}
        for index, (left, right) in enumerate(self.merges):
            tokens[FIRST_MERGE + index] = tokens[left] + tokens[right]
        return tokens

    def encode(self, text: str, *, allow_special: bool = False) -> list[int]:
        specials = {token: 256 + index for index, token in enumerate(SPECIAL_TOKENS)}
        pieces = (
            re.split("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")", text)
            if allow_special
            else [text]
        )
        encoded: list[int] = []
        for piece in pieces:
            if allow_special and piece in specials:
                encoded.append(specials[piece])
                continue
            for chunk in CHUNKS.findall(piece):
                ids = list(chunk.encode("utf-8"))
                for index, pair in enumerate(self.merges):
                    ids = merge_pair(ids, pair, FIRST_MERGE + index)
                encoded.extend(ids)
        return encoded

    def decode(self, ids: Iterable[int], *, skip_special: bool = False) -> str:
        tokens = self.byte_tokens()
        output = bytearray()
        for token_id in ids:
            if type(token_id) is not int:
                raise ValueError("token IDs must be integers")
            if 256 <= token_id < FIRST_MERGE:
                if not skip_special:
                    output.extend(SPECIAL_TOKENS[token_id - 256].encode())
            elif token_id in tokens:
                output.extend(tokens[token_id])
            else:
                raise ValueError(f"unknown token id: {token_id}")
        return output.decode("utf-8", errors="replace")

    def serialized(self) -> str:
        return json.dumps(
            {"version": 1, "specials": SPECIAL_TOKENS, "merges": self.merges},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialized().encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.serialized() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteBPE":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "specials", "merges"}
            or type(raw["version"]) is not int
            or raw["version"] != 1
            or raw["specials"] != list(SPECIAL_TOKENS)
        ):
            raise ValueError("unsupported tokenizer format")
        if not isinstance(raw["merges"], list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            for pair in raw["merges"]
        ):
            raise ValueError("invalid merge table structure")
        return cls(tuple((pair[0], pair[1]) for pair in raw["merges"]))

    @classmethod
    def train(cls, corpus: Iterable[str], vocab_size: int) -> "ByteBPE":
        if type(vocab_size) is not int or not FIRST_MERGE <= vocab_size <= 65536:
            raise ValueError("vocabulary must include byte/special tokens and fit uint16")
        chunks: Counter[str] = Counter()
        for text in corpus:
            chunks.update(CHUNKS.findall(text))
        sequences = [list(chunk.encode("utf-8")) for chunk in chunks]
        weights = list(chunks.values())
        counts: Counter[tuple[int, int]] = Counter()
        locations: dict[tuple[int, int], set[int]] = defaultdict(set)
        for index, sequence in enumerate(sequences):
            for pair, frequency in Counter(zip(sequence, sequence[1:], strict=False)).items():
                counts[pair] += frequency * weights[index]
                locations[pair].add(index)
        heap = [(-count, pair) for pair, count in counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - FIRST_MERGE and heap:
            negative, pair = heapq.heappop(heap)
            if counts[pair] <= 0 or -negative != counts[pair]:
                continue
            new_id = FIRST_MERGE + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for index in sorted(locations[pair].copy()):
                old = sequences[index]
                old_pairs = Counter(zip(old, old[1:], strict=False))
                new = merge_pair(old, pair, new_id)
                new_pairs = Counter(zip(new, new[1:], strict=False))
                for other in old_pairs.keys() | new_pairs.keys():
                    counts[other] += (new_pairs[other] - old_pairs[other]) * weights[index]
                    changed.add(other)
                    if other in new_pairs:
                        locations[other].add(index)
                    else:
                        locations[other].discard(index)
                sequences[index] = new
            for other in changed:
                if counts[other] > 0:
                    heapq.heappush(heap, (-counts[other], other))
        return cls(tuple(merges))
