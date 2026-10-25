"""Bind the admitted S rehearsal build to runs A and B; stage, fetch and verify its files.

``build`` (Mac)
    Require a completed, clean, admitted and verified build of this exact spec, with
    identities that recompute in spec stream order. Re-hash every shard and its
    documents, then write the binding: each shard, manifest and the tokenizer with size,
    SHA-256 and R2 key, and for each run stage its ordered shards, document-order
    digest, token, window and update counts, and the order-independent digest of its
    documents' token hashes. Run A's stage digests must combine to run B's.
``stage`` (Mac)
    Upload the pinned files and the build records under the storage cap, the binding last.
``fetch`` (Pod)
    Download one run's files by key, each pinned to its size and SHA-256. A file already
    present with the pinned bytes is kept; a present file with other bytes stops the fetch.
``verify`` (Pod or Mac)
    Check the binding's own SHA-256, re-hash every file a run needs under a root, open
    each shard and recompute each stage's order digest. The binding is only ever read.

R2 credentials come only from the environment and are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import numpy.typing as npt

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.onepass import MODULUS, combine_digests, order_digest, pass_order
from scratchcoder.data.shards import TokenShard, file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from tools.pilot_data import _load_tokenizer
from tools.r2_store import Credentials, Store, upload

SCHEMA_VERSION = 1
KIND = "rehearsal-data-binding"
VOCAB_SIZE = 65_536
"""The S model's vocabulary: the tokenizer and every shard must declare exactly this."""
BOS_ID = 256
SEQUENCE_LENGTH = 1024
MICROBATCH = 16
ACCUMULATION = 8
ORDER_SEED = 17
RUNS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "A": (("stage1", ("stage1-web",)), ("stage2", ("stage2-python", "stage2-web"))),
    "B": (("mixed", ("stage1-web", "stage2-web", "stage2-python")),),
}
"""Each run's stages, and the training streams each stage reads in that order."""
VALIDATION_STREAMS = ("validation-web", "validation-python")
SUBSET_RULE = {"target_tokens_per_domain": 1_000_000, "sequence_length": SEQUENCE_LENGTH}
"""Recorded only: the interval-validation subset is chosen on the Pod by this rule."""
PACKING = {
    "sequence_length": SEQUENCE_LENGTH,
    "microbatch": MICROBATCH,
    "accumulation": ACCUMULATION,
    "windows_per_update": MICROBATCH * ACCUMULATION,
    "tokens_per_update": MICROBATCH * ACCUMULATION * SEQUENCE_LENGTH,
}
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""One plain path component; it cannot be empty, '.', '..' or hidden."""
ENV_PREFIX = re.compile(r"[A-Z][A-Z0-9_]*_")

BINDING_KEYS = {
    "schema_version",
    "kind",
    "created_at",
    "written_to",
    "binder",
    "build",
    "r2_prefix",
    "packing",
    "tokenizer",
    "stream_order",
    "streams",
    "runs",
    "same_multiset",
    "validation",
}
BUILD_KEYS = {
    "name",
    "spec_sha256",
    "report_sha256",
    "verification_sha256",
    "tokens_identity",
    "build_identity",
    "git_commit",
    "admission",
    "holdout_sha256",
    "documents",
    "exclusions",
}
TOKENIZER_KEYS = {
    "path",
    "source",
    "key",
    "sha256",
    "size_bytes",
    "identity",
    "vocab_size",
    "bos_id",
}
STREAM_KEYS = {
    "domain",
    "split",
    "quota_tokens",
    "tokens",
    "documents",
    "multiset_sha256",
    "shards",
}
SHARD_KEYS = {
    "file",
    "sha256",
    "manifest_sha256",
    "size_bytes",
    "manifest_size_bytes",
    "documents",
    "tokens",
    "key",
    "manifest_key",
}
STAGE_KEYS = {
    "name",
    "streams",
    "shards",
    "order_seed",
    "order_sha256",
    "documents",
    "tokens",
    "targets",
    "windows",
    "planned_updates",
    "multiset_sha256",
}
RUN_KEYS = {"stages", "documents", "tokens", "multiset_sha256"}


def _hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    return value


def _relative(value: Any, name: str) -> str:
    """A relative POSIX path of plain components, so it cannot leave its root."""
    if not isinstance(value, str) or not all(PART.fullmatch(part) for part in value.split("/")):
        raise ValueError(f"{name} must be a relative path of plain components: {value!r}")
    return value


def _prefix(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("/"):
        raise ValueError("the R2 prefix must be plain path components ending in '/'")
    _relative(value[:-1], "the R2 prefix")
    return value


def _directory(root: Path) -> Path:
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{root} must be a real directory, not a symlink")
    return root


def _walk(root: Path, relative: str, *, create: bool = False) -> tuple[Path, os.stat_result | None]:
    """``root/relative`` reached without following any symlink, and the entry's lstat.

    Missing directories are created when ``create`` is set; a missing file gives None.
    """
    path = _directory(root)
    parts = _relative(relative, "path").split("/")
    for part in parts[:-1]:
        path = path / part
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            if not create:
                raise
            path.mkdir()
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{relative}: {part} is not a real directory")
    path = path / parts[-1]
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return path, None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{relative} must be a regular file, not a symlink")
    return path, info


def _regular(root: Path, relative: str) -> Path:
    path, info = _walk(root, relative)
    if info is None:
        raise FileNotFoundError(f"{relative} is missing under {root}")
    return path


def _pinned(root: Path, relative: str, size: int, sha256: str) -> Path:
    path = _regular(root, relative)
    if path.stat().st_size != size or file_sha256(path) != sha256:
        raise ValueError(f"{relative}: size or SHA-256 differs from the binding")
    return path


def _plain_file(path: Path) -> Path:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path} must be a regular file, not a symlink")
    return path


def _serialized(binding: Mapping[str, Any]) -> bytes:
    return (json.dumps(binding, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def binding_sha256(binding: Mapping[str, Any]) -> str:
    """The binding file's SHA-256; a binding is only ever written in this serialization."""
    return hashlib.sha256(_serialized(binding)).hexdigest()


def binding_key(binding: Mapping[str, Any]) -> str:
    """The binding's own R2 key, named by its SHA-256 so no upload can replace it."""
    return f"{binding['r2_prefix']}bindings/{binding_sha256(binding)}.json"


def _write_new(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` whole, and only if nothing is there yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        # A hard link fails if anything, even a dangling symlink, already holds the name.
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _packed(tokens: int) -> dict[str, int]:
    """The runs count every shard token as one target; the last window and update are partial."""
    windows = -(-tokens // SEQUENCE_LENGTH)
    return {
        "tokens": tokens,
        "targets": tokens,
        "windows": windows,
        "planned_updates": -(-windows // PACKING["windows_per_update"]),
    }


def _training_streams(runs: Sequence[str] = tuple(RUNS)) -> list[str]:
    return list(dict.fromkeys(name for run in runs for _, names in RUNS[run] for name in names))


def check_binding(binding: Any) -> dict[str, Any]:
    """Validate a binding's structure and arithmetic without touching any file."""
    top = _exact(binding, BINDING_KEYS, "binding")
    if top["schema_version"] != SCHEMA_VERSION or top["kind"] != KIND:
        raise ValueError("not a rehearsal data binding of this schema")
    if top["packing"] != PACKING:
        raise ValueError("the binding's packing differs from the S model's")
    if not isinstance(top["created_at"], str) or not isinstance(top["written_to"], str):
        raise ValueError("the binding's creation time and path must be strings")
    _exact(top["binder"], {"git_commit", "dirty"}, "binder")
    prefix = _prefix(top["r2_prefix"])
    build = _exact(top["build"], BUILD_KEYS, "build")
    for key in (
        "spec_sha256",
        "report_sha256",
        "verification_sha256",
        "tokens_identity",
        "build_identity",
        "holdout_sha256",
    ):
        _hex(build[key], f"build {key}")
    if not isinstance(build["admission"], dict) or not isinstance(build["git_commit"], str):
        raise ValueError("the binding must record the build's admission and commit")
    for key in ("documents", "exclusions"):
        record = _exact(build[key], {"file", "sha256"}, f"build {key}")
        _relative(record["file"], f"build {key} file")
        _hex(record["sha256"], f"build {key} sha256")
    tokenizer = _exact(top["tokenizer"], TOKENIZER_KEYS, "tokenizer")
    path = _relative(tokenizer["path"], "tokenizer path")
    if (
        path.count("/") != 1
        or not path.startswith("tokenizer/")
        or tokenizer["key"] != prefix + path
    ):
        raise ValueError("the tokenizer must sit at tokenizer/<name> under its prefixed key")
    _hex(tokenizer["sha256"], "tokenizer sha256")
    _hex(tokenizer["identity"], "tokenizer identity")
    _count(tokenizer["size_bytes"], "tokenizer size")
    if tokenizer["vocab_size"] != VOCAB_SIZE or tokenizer["bos_id"] != BOS_ID:
        raise ValueError(f"the S model needs vocabulary {VOCAB_SIZE} and BOS {BOS_ID}")
    order, streams = top["stream_order"], top["streams"]
    if (
        not isinstance(order, list)
        or not all(isinstance(name, str) for name in order)
        or not isinstance(streams, dict)
        or len(set(order)) != len(order)
        or set(order) != set(streams)
        or not {*_training_streams(), *VALIDATION_STREAMS} <= set(order)
    ):
        raise ValueError("stream_order must list every bound stream once, the runs' included")
    seen: set[str] = set()
    for name in order:
        stream = _exact(streams[name], STREAM_KEYS, f"stream {name}")
        if not PART.fullmatch(name) or not isinstance(stream["shards"], list):
            raise ValueError(f"stream {name!r} has an unsafe name or no shard list")
        if (name in VALIDATION_STREAMS) != (stream["split"] == "validation"):
            raise ValueError(f"{name}: validation streams, and only they, are the validation split")
        _hex(stream["multiset_sha256"], f"{name} multiset")
        for shard in stream["shards"]:
            entry = _exact(shard, SHARD_KEYS, f"{name} shard")
            file = _relative(entry["file"], "shard file")
            parent, _, base = file.rpartition("/")
            if parent != f"shards/{name}" or not base.endswith(".bin"):
                raise ValueError(f"{file}: shards must sit at shards/{name}/<name>.bin")
            if entry["key"] != prefix + file or entry["manifest_key"] != entry["key"] + ".json":
                raise ValueError(f"{file}: R2 keys must be the prefix plus the file path")
            for key in ("sha256", "manifest_sha256"):
                _hex(entry[key], f"{file} {key}")
            for key in ("documents", "tokens", "manifest_size_bytes"):
                _count(entry[key], f"{file} {key}")
            if entry["size_bytes"] != 2 * entry["tokens"]:
                raise ValueError(f"{file}: a uint16 shard holds two bytes per token")
            if entry["sha256"] in seen:
                raise ValueError(f"{file}: a shard is listed twice")
            seen.add(entry["sha256"])
        documents = sum(shard["documents"] for shard in stream["shards"])
        tokens = sum(shard["tokens"] for shard in stream["shards"])
        if (
            stream["documents"] != documents
            or stream["tokens"] != tokens
            or stream["quota_tokens"] not in (None, tokens)
        ):
            raise ValueError(f"{name}: totals differ from its shards or its quota")
    runs = _exact(top["runs"], set(RUNS), "runs")
    for run, stages in RUNS.items():
        entry = _exact(runs[run], RUN_KEYS, f"run {run}")
        recorded = entry["stages"]
        if not isinstance(recorded, list) or len(recorded) != len(stages):
            raise ValueError(f"run {run} must list exactly the stages {[s for s, _ in stages]}")
        for (stage_name, names), item in zip(stages, recorded, strict=True):
            stage = _exact(item, STAGE_KEYS, f"run {run} stage")
            label = f"run {run} {stage_name}"
            if stage["name"] != stage_name or stage["streams"] != list(names):
                raise ValueError(f"{label}: the stage or its streams differ from the run design")
            shards = [shard for n in names for shard in streams[n]["shards"]]
            if stage["shards"] != [shard["file"] for shard in shards]:
                raise ValueError(f"{label}: the shard list differs from its streams' shards")
            _count(stage["order_seed"], f"{label} order seed")
            _hex(stage["order_sha256"], f"{label} order")
            expected = {
                "documents": sum(shard["documents"] for shard in shards),
                **_packed(sum(shard["tokens"] for shard in shards)),
                "multiset_sha256": combine_digests(*(streams[n]["multiset_sha256"] for n in names)),
            }
            if {key: stage[key] for key in expected} != expected:
                raise ValueError(f"{label}: counts or multiset differ from its streams")
        totals = {
            "documents": sum(stage["documents"] for stage in recorded),
            "tokens": sum(stage["tokens"] for stage in recorded),
            "multiset_sha256": combine_digests(*(stage["multiset_sha256"] for stage in recorded)),
        }
        if {key: entry[key] for key in totals} != totals:
            raise ValueError(f"run {run}: totals differ from its stages")
    same = _exact(top["same_multiset"], set(RUNS), "same_multiset")
    if len({*same.values(), *(runs[run]["multiset_sha256"] for run in RUNS)}) != 1:
        raise ValueError("runs A and B must read the same token multiset")
    validation = _exact(top["validation"], {"streams", "subset_rule"}, "validation")
    if (
        validation["streams"] != list(VALIDATION_STREAMS)
        or validation["subset_rule"] != SUBSET_RULE
    ):
        raise ValueError("the validation streams or subset rule differ")
    return top


def load_binding(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Read and check a binding whose bytes hash to ``expected_sha256``."""
    _hex(expected_sha256, "the expected binding SHA-256")
    data = _plain_file(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("the binding's SHA-256 differs from the expected value")
    binding = check_binding(json.loads(data))
    if _serialized(binding) != data:
        raise ValueError("the binding is not in its canonical serialization")
    return binding


def _require_admitted(spec_sha256: str, report: Mapping[str, Any]) -> None:
    if report.get("status") != "completed":
        raise ValueError("the build report is not completed")
    if report.get("dirty") is not False or not isinstance(report.get("git_commit"), str):
        raise ValueError("the build did not run from a clean, known commit")
    if report.get("spec_sha256") != spec_sha256:
        raise ValueError("the build report was made from a different spec")
    admission, holdout = report.get("admission"), report.get("holdout")
    if report.get("final_training_admission") is not True or not isinstance(admission, dict):
        raise ValueError("the build is not admitted for training")
    if not isinstance(holdout, dict) or admission.get("holdout_sha256") != holdout.get("sha256"):
        raise ValueError("training admission needs the recorded task-pool holdout exclusion")
    _hex(holdout.get("sha256"), "the holdout SHA-256")


def _require_verified(
    report: Mapping[str, Any], verification: Mapping[str, Any], report_sha256: str
) -> None:
    if verification.get("status") != "passed":
        raise ValueError("the build's verification did not pass")
    if verification.get("report_sha256") != report_sha256:
        raise ValueError("verification.json was written for a different report.json")
    if (
        any(
            verification.get(key) != report.get(key)
            for key in ("tokens_identity", "build_identity")
        )
        or verification.get("final_training_admission") is not True
    ):
        raise ValueError("verification.json records other identities or no admission")


def _shard_multiset(path: Path, offsets: Sequence[tuple[int, int]]) -> int:
    """Sum of each document's SHA-256 over its uint16-LE bytes, BOS included, mod 2**256."""
    total = 0
    if not offsets:
        return total
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as view:
        for offset, length in offsets:
            digest = hashlib.sha256(view[2 * offset : 2 * (offset + length)]).digest()
            total += int.from_bytes(digest, "big")
    return total % MODULUS


def _listing_multisets(
    path: Path,
    sha256: str,
    names: Sequence[str],
    lengths: Mapping[tuple[str, str], npt.NDArray[np.int64]],
) -> dict[str, str]:
    """Per-stream multiset digests of documents.jsonl's token hashes, read as a stream.

    Each row must sit at its shard's next position and carry that document's length.
    """
    if file_sha256(path) != sha256:
        raise ValueError("documents.jsonl differs from the report")
    position = dict.fromkeys(lengths, 0)
    totals = dict.fromkeys(names, 0)
    with path.open("rb") as lines:
        for line in lines:
            row = json.loads(line)
            key = (row["stream"], row["shard"])
            index = position.get(key)
            if (
                index is None
                or row["index"] != index
                or index >= len(lengths[key])
                or row["tokens"] != int(lengths[key][index])
            ):
                raise ValueError(f"documents.jsonl disagrees with {key} at index {row['index']}")
            position[key] = index + 1
            value = int(_hex(row["tokens_sha256"], "tokens_sha256"), 16)
            totals[key[0]] = (totals[key[0]] + value) % MODULUS
    missing = sorted(key for key, count in position.items() if count != len(lengths[key]))
    if missing:
        raise ValueError(f"documents.jsonl lacks documents of {missing}")
    return {name: f"{total:064x}" for name, total in totals.items()}


def build_binding(
    build_dir: Path,
    spec_path: Path,
    output: Path,
    *,
    r2_prefix: str,
    order_seed: int = ORDER_SEED,
) -> dict[str, Any]:
    """Check the admitted build in ``build_dir`` (records in ``output/``); write its binding."""
    # Imported here so the Pod side (fetch, verify) never loads the build's dependencies.
    from tools import rehearsal_build

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists; a binding is never rewritten")
    prefix = _prefix(r2_prefix)
    if type(order_seed) is not int or order_seed < 0:
        raise ValueError("the order seed must be a nonnegative integer")
    spec = rehearsal_build.load_spec(spec_path)
    records = build_dir / "output"
    report_path = _regular(records, "report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require_admitted(spec.sha256, report)
    verification_path = _regular(records, "verification.json")
    report_sha256 = file_sha256(report_path)
    _require_verified(report, json.loads(verification_path.read_text()), report_sha256)
    # identities() walks spec.streams; report.json's keys are sorted, not in spec order.
    recomputed = rehearsal_build.identities(spec, report)
    if recomputed != {key: report[key] for key in recomputed}:
        raise ValueError("the report's identities differ from a recomputation in spec order")
    names = [stream.name for stream in spec.streams]
    required = {*_training_streams(), *VALIDATION_STREAMS}
    if not required <= set(names) or set(report["streams"]) != set(names):
        raise ValueError(f"the spec and report must hold the streams {sorted(required)}")
    tokenizer_spec = spec.tokenizer
    if report["tokenizer"] != tokenizer_spec:
        raise ValueError("the report's tokenizer differs from the spec's")
    if tokenizer_spec["vocab_size"] != VOCAB_SIZE or tokenizer_spec["bos_id"] != BOS_ID:
        raise ValueError(f"the S model needs vocabulary {VOCAB_SIZE} and BOS {BOS_ID}")
    source = Path(tokenizer_spec["path"])
    _load_tokenizer(
        source,
        expected_identity=tokenizer_spec["identity"],
        expected_file_sha256=tokenizer_spec["file_sha256"],
        expected_vocab_size=VOCAB_SIZE,
        expected_bos_id=BOS_ID,
    )
    tokenizer_path = _relative(f"tokenizer/{source.name}", "tokenizer path")
    identity = tokenizer_spec["identity"]
    streams: dict[str, dict[str, Any]] = {}
    lengths: dict[tuple[str, str], npt.NDArray[np.int64]] = {}
    for stream in spec.streams:
        recorded = report["streams"][stream.name]
        if (recorded["domain"], recorded["split"], recorded["quota_tokens"]) != (
            stream.domain,
            stream.split,
            stream.tokens,
        ):
            raise ValueError(f"{stream.name}: the report's stream differs from the spec's")
        shards = []
        total = 0
        for shard in recorded["shards"]:
            file = _relative(shard["file"], "shard file")
            path, manifest = _regular(records, file), _regular(records, file + ".json")
            if (
                file_sha256(path) != shard["sha256"]
                or file_sha256(manifest) != shard["manifest_sha256"]
            ):
                raise ValueError(f"{file}: shard or manifest SHA-256 differs from the report")
            opened = TokenShard(path, tokenizer_hash=identity, vocab_size=VOCAB_SIZE, bos_id=BOS_ID)
            sizes = np.fromiter((n for _, n in opened.offsets), dtype=np.int64, count=len(opened))
            if len(opened) != shard["documents"] or int(sizes.sum()) != shard["tokens"]:
                raise ValueError(f"{file}: documents or tokens differ from the report")
            lengths[(stream.name, Path(file).name)] = sizes
            total = (total + _shard_multiset(path, opened.offsets)) % MODULUS
            shards.append(
                {
                    "file": file,
                    "sha256": shard["sha256"],
                    "manifest_sha256": shard["manifest_sha256"],
                    "size_bytes": path.stat().st_size,
                    "manifest_size_bytes": manifest.stat().st_size,
                    "documents": shard["documents"],
                    "tokens": shard["tokens"],
                    "key": prefix + file,
                    "manifest_key": prefix + file + ".json",
                }
            )
        streams[stream.name] = {
            "domain": stream.domain,
            "split": stream.split,
            "quota_tokens": stream.tokens,
            "tokens": sum(shard["tokens"] for shard in shards),
            "documents": sum(shard["documents"] for shard in shards),
            "multiset_sha256": f"{total:064x}",
            "shards": shards,
        }
        if (streams[stream.name]["tokens"], streams[stream.name]["documents"]) != (
            recorded["tokens"],
            recorded["documents"],
        ):
            raise ValueError(f"{stream.name}: shard totals differ from the report")
    listing = report["documents"]
    listed = _listing_multisets(
        _regular(records, _relative(listing["file"], "documents file")),
        listing["sha256"],
        names,
        lengths,
    )
    for name in names:
        if listed[name] != streams[name]["multiset_sha256"]:
            raise ValueError(f"{name}: documents.jsonl token hashes differ from the shard bytes")
    runs: dict[str, dict[str, Any]] = {}
    for run, stages in RUNS.items():
        rows = []
        for stage_name, stage_streams in stages:
            stage_shards = [shard for n in stage_streams for shard in streams[n]["shards"]]
            order = pass_order([(s["sha256"], s["documents"]) for s in stage_shards], order_seed)
            rows.append(
                {
                    "name": stage_name,
                    "streams": list(stage_streams),
                    "shards": [shard["file"] for shard in stage_shards],
                    "order_seed": order_seed,
                    "order_sha256": order_digest(order),
                    "documents": sum(shard["documents"] for shard in stage_shards),
                    **_packed(sum(shard["tokens"] for shard in stage_shards)),
                    "multiset_sha256": combine_digests(*(listed[n] for n in stage_streams)),
                }
            )
        runs[run] = {
            "stages": rows,
            "documents": sum(row["documents"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "multiset_sha256": combine_digests(*(row["multiset_sha256"] for row in rows)),
        }
    if runs["A"]["multiset_sha256"] != runs["B"]["multiset_sha256"]:
        raise ValueError("run A's stage multisets do not combine to run B's")
    commit, dirty = git_state()
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_at": utc_now(),
        "written_to": str(output.absolute()),
        "binder": {"git_commit": commit, "dirty": dirty},
        "build": {
            "name": report["name"],
            "spec_sha256": spec.sha256,
            "report_sha256": report_sha256,
            "verification_sha256": file_sha256(verification_path),
            "tokens_identity": report["tokens_identity"],
            "build_identity": report["build_identity"],
            "git_commit": report["git_commit"],
            "admission": report["admission"],
            "holdout_sha256": report["holdout"]["sha256"],
            "documents": {"file": listing["file"], "sha256": listing["sha256"]},
            "exclusions": {
                "file": report["exclusions"]["file"],
                "sha256": report["exclusions"]["sha256"],
            },
        },
        "r2_prefix": prefix,
        "packing": PACKING,
        "tokenizer": {
            "path": tokenizer_path,
            "source": str(source.absolute()),
            "key": prefix + tokenizer_path,
            "sha256": tokenizer_spec["file_sha256"],
            "size_bytes": source.stat().st_size,
            "identity": identity,
            "vocab_size": VOCAB_SIZE,
            "bos_id": BOS_ID,
        },
        "stream_order": names,
        "streams": streams,
        "runs": runs,
        "same_multiset": {run: runs[run]["multiset_sha256"] for run in RUNS},
        "validation": {"streams": list(VALIDATION_STREAMS), "subset_rule": SUBSET_RULE},
    }
    # A reader applies the same rules, so a binding that fails them is never written.
    check_binding(binding)
    _write_new(output, _serialized(binding))
    return binding


def run_files(binding: Mapping[str, Any], run: str) -> list[dict[str, Any]]:
    """Every file one run reads: its stages' and both validation streams' shards and
    manifests, then the tokenizer, each with its relative path, R2 key, size and SHA-256."""
    if run not in RUNS:
        raise ValueError(f"unknown run {run!r}; the runs are {sorted(RUNS)}")
    needed = {*_training_streams([run]), *VALIDATION_STREAMS}
    files = []
    for name in binding["stream_order"]:
        if name not in needed:
            continue
        for shard in binding["streams"][name]["shards"]:
            files.append(
                {
                    "path": shard["file"],
                    "key": shard["key"],
                    "size_bytes": shard["size_bytes"],
                    "sha256": shard["sha256"],
                }
            )
            files.append(
                {
                    "path": shard["file"] + ".json",
                    "key": shard["manifest_key"],
                    "size_bytes": shard["manifest_size_bytes"],
                    "sha256": shard["manifest_sha256"],
                }
            )
    tokenizer = binding["tokenizer"]
    files.append({key: tokenizer[key] for key in ("path", "key", "size_bytes", "sha256")})
    return files


def verify_binding(
    binding_path: Path, root: Path, *, expected_sha256: str, run: str | None = None
) -> dict[str, Any]:
    """Check every file the run (or, without ``run``, both runs) reads under ``root``.

    Returns the open shards per stream, each stage with its shards and recomputed
    document order, and the loaded tokenizer. The binding file is only read.
    """
    binding = load_binding(binding_path, expected_sha256=expected_sha256)
    selected = list(RUNS) if run is None else [run]
    files = {item["path"]: item for name in selected for item in run_files(binding, name)}
    for item in files.values():
        _pinned(root, item["path"], item["size_bytes"], item["sha256"])
    entry = binding["tokenizer"]
    tokenizer = ByteBPE.load(_regular(root, entry["path"]))
    if (
        tokenizer.sha256 != entry["identity"]
        or tokenizer.vocab_size != VOCAB_SIZE
        or tokenizer.bos_id != BOS_ID
    ):
        raise ValueError("the tokenizer's identity, vocabulary or BOS differs from the binding")
    needed = {*_training_streams(selected), *VALIDATION_STREAMS}
    streams: dict[str, list[tuple[str, TokenShard]]] = {}
    for name in (name for name in binding["stream_order"] if name in needed):
        opened = []
        for shard in binding["streams"][name]["shards"]:
            token_shard = TokenShard(
                _regular(root, shard["file"]),
                tokenizer_hash=entry["identity"],
                vocab_size=VOCAB_SIZE,
                bos_id=BOS_ID,
            )
            tokens = sum(length for _, length in token_shard.offsets)
            if len(token_shard) != shard["documents"] or tokens != shard["tokens"]:
                raise ValueError(f"{shard['file']}: documents or tokens differ from the binding")
            opened.append((shard["sha256"], token_shard))
        streams[name] = opened
    stages: dict[str, list[dict[str, Any]]] = {}
    for name in selected:
        stages[name] = []
        for stage in binding["runs"][name]["stages"]:
            shards = [pair for stream in stage["streams"] for pair in streams[stream]]
            order = pass_order([(sha, len(shard)) for sha, shard in shards], stage["order_seed"])
            if order_digest(order) != stage["order_sha256"]:
                raise ValueError(f"run {name} {stage['name']}: document order differs")
            stages[name].append({**stage, "shards": shards, "order": order})
    return {
        "binding_sha256": expected_sha256,
        "build": binding["build"],
        "packing": binding["packing"],
        "validation": binding["validation"],
        "streams": streams,
        "stages": stages,
        "tokenizer": tokenizer,
    }


def fetch(
    binding: Mapping[str, Any],
    root: Path,
    store: Store,
    *,
    run: str,
    receipt: Path,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Download every file ``run`` reads into ``root`` and write a receipt.

    Each download is pinned to the binding's size and SHA-256 and published only when
    both match. A present file with the pinned bytes is kept; any other stops the fetch.
    """
    checked = check_binding(binding)
    files = run_files(checked, run)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for item in files:
        target, info = _walk(root, item["path"], create=True)
        if info is not None:
            if info.st_size != item["size_bytes"] or file_sha256(target) != item["sha256"]:
                raise ValueError(
                    f"{item['path']} is present with other bytes; remove it to refetch"
                )
            row = {**item, "method": "present", "bytes_transferred": 0}
        else:
            got = store.download(
                item["key"], target, sha256=item["sha256"], size=item["size_bytes"]
            )
            row = {
                **item,
                "method": "downloaded",
                "bytes_transferred": got["bytes"],
                "attempts": got["attempts"],
                "seconds": got["seconds"],
            }
        rows.append(row)
        progress(f"{row['method']} {item['path']} ({item['size_bytes']:,} bytes)")
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "run": run,
        "binding_sha256": binding_sha256(checked),
        "tokens_identity": checked["build"]["tokens_identity"],
        "build_identity": checked["build"]["build_identity"],
        "bucket": store.credentials.bucket,
        "root": str(root),
        "files": rows,
        "bytes_transferred": sum(row["bytes_transferred"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(receipt, record)
    return record


def staging_files(binding: Mapping[str, Any], build_dir: Path) -> list[tuple[str, Path]]:
    """(R2 key, local path) pairs for ``r2_store.upload``, each checked against its pin.

    The shards, manifests and tokenizer come first, then documents.jsonl,
    exclusions.jsonl, report.json and verification.json, and the binding last, so a
    reader never finds a binding before the files it names.
    """
    checked = check_binding(binding)
    records = build_dir / "output"
    prefix, build = checked["r2_prefix"], checked["build"]
    pinned: list[tuple[str, Path, int | None, str]] = []
    for name in checked["stream_order"]:
        for shard in checked["streams"][name]["shards"]:
            file = shard["file"]
            pinned.append(
                (shard["key"], _regular(records, file), shard["size_bytes"], shard["sha256"])
            )
            pinned.append(
                (
                    shard["manifest_key"],
                    _regular(records, file + ".json"),
                    shard["manifest_size_bytes"],
                    shard["manifest_sha256"],
                )
            )
    tokenizer = checked["tokenizer"]
    pinned.append(
        (
            tokenizer["key"],
            _plain_file(Path(tokenizer["source"])),
            tokenizer["size_bytes"],
            tokenizer["sha256"],
        )
    )
    for record in (build["documents"], build["exclusions"]):
        pinned.append(
            (prefix + record["file"], _regular(records, record["file"]), None, record["sha256"])
        )
    for name, digest in (
        ("report.json", build["report_sha256"]),
        ("verification.json", build["verification_sha256"]),
    ):
        pinned.append((prefix + name, _regular(records, name), None, digest))
    pairs = []
    for key, path, size, sha256 in pinned:
        if (size is not None and path.stat().st_size != size) or file_sha256(path) != sha256:
            raise ValueError(f"{path}: bytes differ from the binding")
        pairs.append((key, path))
    written = _plain_file(Path(checked["written_to"]))
    if written.read_bytes() != _serialized(checked):
        raise ValueError(f"{written} is not this binding")
    pairs.append((binding_key(checked), written))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("two staged files share an R2 key")
    return pairs


def r2_credentials(
    environ: Mapping[str, str] = os.environ, *, key_prefix: str = "R2_"
) -> Credentials:
    """R2 credentials from the environment, never printed.

    ``key_prefix`` names the key pair, e.g. ``R2_POD_`` for ``R2_POD_ACCESS_KEY_ID`` and
    ``R2_POD_SECRET_ACCESS_KEY``; ``R2_ENDPOINT`` and ``R2_BUCKET`` are shared.
    """
    if not ENV_PREFIX.fullmatch(key_prefix):
        raise ValueError("the key prefix must be uppercase letters, digits and '_', like R2_POD_")
    names = {
        "R2_ENDPOINT": "R2_ENDPOINT",
        "R2_BUCKET": "R2_BUCKET",
        "R2_ACCESS_KEY_ID": key_prefix + "ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY": key_prefix + "SECRET_ACCESS_KEY",
    }
    missing = [variable for variable in names.values() if not environ.get(variable)]
    if missing:
        raise ValueError("set the environment variables " + ", ".join(missing))
    return Credentials.from_environment(
        {name: environ[variable] for name, variable in names.items()}
    )


def summary(binding: Mapping[str, Any]) -> dict[str, Any]:
    """The identities and per-stage counts and digests, for printing."""
    fields = ("name", "documents", "tokens", "windows", "planned_updates", "order_sha256")
    return {
        "tokens_identity": binding["build"]["tokens_identity"],
        "build_identity": binding["build"]["build_identity"],
        "same_multiset": binding["same_multiset"]["A"],
        "runs": {
            run: [{key: stage[key] for key in fields} for stage in binding["runs"][run]["stages"]]
            for run in RUNS
        },
    }


def _verified_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("name", "documents", "tokens", "planned_updates", "order_sha256")
    return {
        "binding_sha256": result["binding_sha256"],
        "tokenizer": result["tokenizer"].sha256,
        "streams": {name: len(shards) for name, shards in result["streams"].items()},
        "stages": {
            run: [{key: stage[key] for key in fields} for stage in stages]
            for run, stages in result["stages"].items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="write the binding of an admitted build (Mac)")
    build.add_argument("--build", type=Path, required=True, help="the rehearsal build directory")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True, help="the binding file to create")
    build.add_argument("--r2-prefix", required=True, help="R2 key prefix ending in '/'")
    build.add_argument("--order-seed", type=int, default=ORDER_SEED)
    verify = commands.add_parser("verify", help="check a run's files under a root")
    fetch_command = commands.add_parser("fetch", help="download a run's files from R2 (Pod)")
    stage = commands.add_parser("stage", help="upload the bound files and the binding (Mac)")
    for command in (verify, fetch_command, stage):
        command.add_argument("--binding", type=Path, required=True)
        command.add_argument("--expected-sha256", required=True, help="the binding's SHA-256")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--run", choices=sorted(RUNS), help="only this run's files")
    fetch_command.add_argument("--root", type=Path, required=True)
    fetch_command.add_argument("--run", choices=sorted(RUNS), required=True)
    fetch_command.add_argument("--receipt", type=Path, required=True)
    fetch_command.add_argument(
        "--key-prefix", default="R2_", help="prefix of the key-pair variables, e.g. R2_POD_"
    )
    stage.add_argument("--build", type=Path, required=True, help="the rehearsal build directory")
    stage.add_argument("--receipt", type=Path, required=True)
    stage.add_argument("--cap-usd", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.command == "build":
        binding = build_binding(
            args.build, args.spec, args.output, r2_prefix=args.r2_prefix, order_seed=args.order_seed
        )
        details = {"sha256": binding_sha256(binding), "key": binding_key(binding)}
        print(json.dumps({"binding": str(args.output), **details, **summary(binding)}, indent=1))
    elif args.command == "verify":
        result = verify_binding(
            args.binding, args.root, expected_sha256=args.expected_sha256, run=args.run
        )
        print(json.dumps(_verified_summary(result), indent=1))
    elif args.command == "fetch":
        binding = load_binding(args.binding, expected_sha256=args.expected_sha256)
        store = Store(r2_credentials(key_prefix=args.key_prefix))
        record = fetch(binding, args.root, store, run=args.run, receipt=args.receipt)
        result = verify_binding(
            args.binding, args.root, expected_sha256=args.expected_sha256, run=args.run
        )
        print(
            json.dumps(
                {"bytes_transferred": record["bytes_transferred"], **_verified_summary(result)},
                indent=1,
            )
        )
    else:
        binding = load_binding(args.binding, expected_sha256=args.expected_sha256)
        files = staging_files(binding, args.build)
        record = upload(Store(r2_credentials()), files, cap_usd=args.cap_usd, receipt=args.receipt)
        print(
            json.dumps(
                {"binding_key": binding_key(binding)}
                | {key: record[key] for key in ("bucket_bytes_after", "monthly_storage_usd_after")},
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
