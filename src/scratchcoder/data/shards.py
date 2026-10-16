"""Little-endian uint16 shards with explicit document boundaries and integrity."""

import hashlib
import json
import re
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
    if (
        type(vocab_size) is not int
        or not 257 <= vocab_size <= 65536
        or type(bos_id) is not int
        or not 0 <= bos_id < vocab_size
    ):
        raise ValueError("shards require a uint16 vocabulary and valid BOS")
    if not isinstance(tokenizer_hash, str) or not tokenizer_hash.strip():
        raise ValueError("nonempty tokenizer identity required")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_suffix(path.suffix + ".json")
    if manifest.exists():
        raise FileExistsError(manifest)
    # Exclusive creation avoids silently overwriting an existing shard/manifest pair.
    offsets: list[tuple[int, int]] = []
    offset = 0
    stream = path.open("xb")
    try:
        with stream:
            for document in documents:
                tokens = [bos_id, *document]
                if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
                    raise ValueError("token ID outside vocabulary")
                stream.write(struct.pack(f"<{len(tokens)}H", *tokens))
                offsets.append((offset, len(tokens)))
                offset += len(tokens)
            stream.flush()
            import os

            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    atomic_json(
        manifest,
        {
            "version": 1,
            "dtype": "uint16-le",
            "vocab_size": vocab_size,
            "bos_id": bos_id,
            "tokenizer_hash": tokenizer_hash,
            "sha256": file_sha256(path),
            "documents": offsets,
        },
    )
    return manifest


class TokenShard:
    """Read individual documents on demand rather than loading a corpus in RAM."""

    def __init__(
        self,
        path: Path,
        *,
        tokenizer_hash: str,
        vocab_size: int | None = None,
        bos_id: int | None = None,
    ) -> None:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        required = {"version", "dtype", "vocab_size", "tokenizer_hash", "sha256", "documents"}
        if (
            not isinstance(metadata, dict)
            or not required <= set(metadata)
            or set(metadata) - required - {"bos_id"}
            or type(metadata["version"]) is not int
            or metadata["version"] != 1
            or metadata["dtype"] != "uint16-le"
        ):
            raise ValueError("unsupported shard format")
        declared_vocab = metadata["vocab_size"]
        if type(declared_vocab) is not int or not 257 <= declared_vocab <= 65536:
            raise ValueError("invalid shard vocabulary")
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size != declared_vocab):
            raise ValueError("shard vocabulary differs from tokenizer")
        declared_bos = metadata.get("bos_id")
        if "bos_id" in metadata and (
            type(declared_bos) is not int or not 0 <= declared_bos < declared_vocab
        ):
            raise ValueError("invalid shard BOS")
        if bos_id is not None and (
            type(bos_id) is not int
            or not 0 <= bos_id < declared_vocab
            or declared_bos is not None
            and bos_id != declared_bos
        ):
            raise ValueError("shard BOS differs from tokenizer")
        if (
            not isinstance(tokenizer_hash, str)
            or not tokenizer_hash.strip()
            or not isinstance(metadata["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])
        ):
            raise ValueError("invalid shard content/tokenizer identity")
        documents = metadata["documents"]
        if not isinstance(documents, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(type(value) is not int for value in item)
            for item in documents
        ):
            raise ValueError("invalid shard document offsets")
        if metadata["tokenizer_hash"] != tokenizer_hash or metadata["sha256"] != file_sha256(path):
            raise ValueError("shard or tokenizer identity mismatch")
        self.path = path
        self.vocab_size: int = declared_vocab
        # Legacy v1 metadata omitted BOS. Only an explicit caller can supply that information.
        self.bos_id: int | None = declared_bos if declared_bos is not None else bos_id
        self.identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        self.offsets: list[tuple[int, int]] = [tuple(item) for item in metadata["documents"]]
        size = path.stat().st_size // 2
        if path.stat().st_size % 2 or any(
            offset < 0 or length < 1 or offset + length > size for offset, length in self.offsets
        ):
            raise ValueError("invalid document offsets")
        expected = 0
        for offset, length in self.offsets:
            if type(offset) is not int or type(length) is not int or offset != expected:
                raise ValueError("shard documents must cover tokens contiguously without overlap")
            expected += length
        if expected != size:
            raise ValueError("shard metadata does not cover all token bytes")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> list[int]:
        offset, length = self.offsets[index]
        with self.path.open("rb") as stream:
            stream.seek(offset * 2)
            tokens = list(struct.unpack(f"<{length}H", stream.read(length * 2)))
        if any(token >= self.vocab_size for token in tokens):
            raise ValueError("shard token ID outside declared vocabulary")
        if self.bos_id is not None and tokens[0] != self.bos_id:
            raise ValueError("shard document does not start with declared BOS")
        return tokens
