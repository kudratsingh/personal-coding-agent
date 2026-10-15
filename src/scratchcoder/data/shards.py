"""Little-endian uint16 shards with explicit document boundaries and integrity."""

import hashlib
import json
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path

from scratchcoder.common.records import atomic_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_shard(
    path: Path,
    documents: Iterable[Sequence[int]],
    *,
    vocab_size: int,
    tokenizer_hash: str,
    bos_id: int = 256,
) -> Path:
    if not 257 <= vocab_size <= 65536 or not 0 <= bos_id < vocab_size:
        raise ValueError("shards require a uint16 vocabulary and valid BOS")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently overwriting an existing shard/manifest pair.
    offsets: list[tuple[int, int]] = []
    offset = 0
    try:
        with path.open("xb") as stream:
            for document in documents:
                tokens = [bos_id, *document]
                if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
                    raise ValueError("token ID outside vocabulary")
                stream.write(struct.pack(f"<{len(tokens)}H", *tokens))
                offsets.append((offset, len(tokens)))
                offset += len(tokens)
    except FileExistsError:
        raise
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    manifest = path.with_suffix(path.suffix + ".json")
    atomic_json(
        manifest,
        {
            "version": 1,
            "dtype": "uint16-le",
            "vocab_size": vocab_size,
            "tokenizer_hash": tokenizer_hash,
            "sha256": file_sha256(path),
            "documents": offsets,
        },
    )
    return manifest


class TokenShard:
    """Read individual documents on demand rather than loading a corpus in RAM."""

    def __init__(self, path: Path, *, tokenizer_hash: str) -> None:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if metadata["version"] != 1 or metadata["dtype"] != "uint16-le":
            raise ValueError("unsupported shard format")
        if metadata["tokenizer_hash"] != tokenizer_hash or metadata["sha256"] != file_sha256(path):
            raise ValueError("shard or tokenizer identity mismatch")
        self.path = path
        self.identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        self.offsets: list[tuple[int, int]] = [tuple(item) for item in metadata["documents"]]
        size = path.stat().st_size // 2
        if path.stat().st_size % 2 or any(
            offset < 0 or length < 1 or offset + length > size for offset, length in self.offsets
        ):
            raise ValueError("invalid document offsets")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> list[int]:
        offset, length = self.offsets[index]
        with self.path.open("rb") as stream:
            stream.seek(offset * 2)
            return list(struct.unpack(f"<{length}H", stream.read(length * 2)))
