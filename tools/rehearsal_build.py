"""Build the S rehearsal pretraining data: pinned containers to exact-quota token shards.

Four stages share one build directory. Each refuses to overwrite a completed record.

``download``
    Resolve every source's pinned revision on the Hub and record each listed file's
    size and LFS SHA-256. Fetch each file once over HTTPS, hashing while it streams;
    a file is published only when its size and hash match. A file an earlier
    verified receipt already holds is hard-linked after the same check.
``licenses``
    Join every row of the Python containers to The Stack dedup v1.2
    ``max_stars_repo_licenses`` on the exact repository and path
    (``tools.stack_licenses``: key and license columns plus each footer only).
``build``
    Review the containers in row-group units across worker processes:
    - Python licenses: an explicit header first, otherwise verified repository
      metadata in which every listed ID is on the D-05 allowlist.
    - The existing per-document quality rules and the benchmark contamination index.
    - Frozen-tokenizer encoding of every admitted document.
    Then walk the units in spec order: drop exact duplicates across the whole build,
    allocate splits by content hash, fill each stream's token quota exactly, write
    uint16 shards, and run ``verify``.
``verify``
    Reopen every shard and check its hash, offsets, BOS positions and vocabulary.
    Check the stream totals and that streams share no document hash. Re-encode the
    source text of a fixed set of documents and compare it with the shard tokens.

Budgets count shard tokens, which include one BOS per document. The document that
crosses a quota is cut to the tokens that fill it exactly, and the cut is recorded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import multiprocessing
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

import numpy as np

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.filters import (
    ContaminationIndex,
    Document,
    FilterConfig,
    content_hash,
    document_split,
    rule_reason,
)
from scratchcoder.data.manifest import TRAINING_LICENSES, HTTPSOnlyRedirectHandler
from scratchcoder.data.shards import TokenShard, file_sha256, write_shard
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder
from tools.header_licenses import (
    REPOSITORY_ALLOWLIST,
    HeaderLicense,
    classify_file,
    repository_license_ids,
)
from tools.pilot_data import _load_tokenizer
from tools.real_samples import REPOSITORY_EVIDENCE, contamination_inputs
from tools.stack_licenses import API_HOST, HuggingFaceClient, _require_api_url, load_mapping

SCHEMA_VERSION = 1
DOMAINS = ("web", "python")
SPLITS = {"web": ("stage1", "stage2", "validation"), "python": ("train", "validation")}
TEXT_COLUMNS = {"web": "text", "python": "content"}
LICENSE_REASONS = (
    "license_copyleft",
    "license_restricted",
    "license_unrecognized",
    "license_missing_repository_license",
    "license_repository_disallowed",
)
FILTER_REASONS = (
    "kept",
    "unknown_domain",
    "too_short",
    "too_large",
    "long_line",
    "low_alphanumeric",
    "generated",
    "duplicate",
    "contamination",
)
PROVENANCE = (
    "row",
    "id",
    "sha256",
    "license",
    "license_basis",
    "repository_licenses",
    "repository",
    "path",
    "url",
)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    domain: str
    dataset: str
    revision: str
    gated: bool
    files: tuple[str, ...]
    license: str
    terms: str
    terms_accepted_on: str
    attribution: str


@dataclass(frozen=True)
class Stream:
    name: str
    domain: str
    split: str
    tokens: int | None


@dataclass(frozen=True)
class Spec:
    name: str
    tokenizer: dict[str, Any]
    sources: tuple[Source, ...]
    license_metadata: dict[str, str]
    contamination: dict[str, str]
    streams: tuple[Stream, ...]
    unit_row_groups: int
    shard_max_tokens: int
    min_free_bytes: int
    max_download_bytes: int
    sha256: str

    def source(self, domain: str) -> Source:
        return next(source for source in self.sources if source.domain == domain)

    def stream_for(self, domain: str, split: str) -> Stream:
        return next(s for s in self.streams if s.domain == domain and s.split == split)


def _exact_keys(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    return value


def _hex(value: Any, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be {length} lowercase hex characters")
    return value


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_spec(path: Path) -> Spec:
    """Parse and validate a build spec; unknown or missing keys are errors."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    top = _exact_keys(
        raw,
        {
            "schema_version",
            "name",
            "tokenizer",
            "sources",
            "license_metadata",
            "contamination",
            "streams",
            "unit_row_groups",
            "shard_max_tokens",
            "min_free_bytes",
            "max_download_bytes",
        },
        "spec",
    )
    if top["schema_version"] != SCHEMA_VERSION or not isinstance(top["name"], str):
        raise ValueError("unsupported spec schema or name")
    tokenizer = _exact_keys(
        top["tokenizer"], {"path", "identity", "file_sha256", "vocab_size", "bos_id"}, "tokenizer"
    )
    _hex(tokenizer["identity"], 64, "tokenizer identity")
    _hex(tokenizer["file_sha256"], 64, "tokenizer file_sha256")
    sources = []
    for item in top["sources"]:
        entry = _exact_keys(
            item,
            {
                "domain",
                "dataset",
                "revision",
                "gated",
                "files",
                "license",
                "terms",
                "terms_accepted_on",
                "attribution",
            },
            "source",
        )
        files = entry["files"]
        if (
            entry["domain"] not in DOMAINS
            or not isinstance(files, list)
            or not files
            or len(set(files)) != len(files)
            or any(not isinstance(f, str) or not f.endswith(".parquet") for f in files)
            or type(entry["gated"]) is not bool
        ):
            raise ValueError("invalid source domain, gating or file list")
        _hex(entry["revision"], 40, "source revision")
        if entry["domain"] == "web" and entry["license"] not in TRAINING_LICENSES:
            raise ValueError("a web source's dataset license must be on the D-05 allowlist")
        sources.append(Source(**{**entry, "files": tuple(files)}))
    if sorted(source.domain for source in sources) != sorted(DOMAINS):
        raise ValueError("the spec needs exactly one web and one Python source")
    metadata = _exact_keys(
        top["license_metadata"], {"dataset", "revision", "terms_accepted_on"}, "license_metadata"
    )
    _hex(metadata["revision"], 40, "license metadata revision")
    contamination = _exact_keys(
        top["contamination"], {"humaneval", "evalplus_directory", "arc_directory"}, "contamination"
    )
    streams = []
    for item in top["streams"]:
        entry = _exact_keys(item, {"name", "domain", "split", "tokens"}, "stream")
        if entry["domain"] not in DOMAINS or entry["split"] not in SPLITS[entry["domain"]]:
            raise ValueError(f"stream {entry['name']!r} has an unknown domain or split")
        if entry["tokens"] is not None:
            _positive(entry["tokens"], "stream tokens")
        streams.append(Stream(**entry))
    covered = sorted((s.domain, s.split) for s in streams)
    expected = sorted((d, split) for d in DOMAINS for split in SPLITS[d])
    if covered != expected or len({s.name for s in streams}) != len(streams):
        raise ValueError("streams must name every (domain, split) exactly once")
    if any((s.split == "validation") != (s.tokens is None) for s in streams):
        raise ValueError("validation streams take every document; training streams need quotas")
    return Spec(
        name=top["name"],
        tokenizer=tokenizer,
        sources=tuple(sources),
        license_metadata=metadata,
        contamination=contamination,
        streams=tuple(streams),
        unit_row_groups=_positive(top["unit_row_groups"], "unit_row_groups"),
        shard_max_tokens=_positive(top["shard_max_tokens"], "shard_max_tokens"),
        min_free_bytes=_positive(top["min_free_bytes"], "min_free_bytes"),
        max_download_bytes=_positive(top["max_download_bytes"], "max_download_bytes"),
        sha256=file_sha256(path),
    )


def require_free(path: Path, needed: int, spec: Spec, stage: str) -> int:
    """Stop before a stage when the disk would fall below the spec's headroom."""
    free = shutil.disk_usage(path).free
    if free - needed < spec.min_free_bytes:
        raise RuntimeError(
            f"{stage}: {free:,} bytes free, {needed:,} needed, headroom "
            f"{spec.min_free_bytes:,}; stopping before any write"
        )
    return free


class Hub:
    """Hugging Face HTTPS reads. A token, when given, never crosses a redirect."""

    def __init__(
        self,
        token: str | None,
        *,
        attempts: int = 6,
        timeout: float = 60,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._opener = build_opener(HTTPSOnlyRedirectHandler())
        self._attempts, self._timeout, self._sleep = attempts, timeout, sleep

    def request(self, url: str, headers: Mapping[str, str] | None = None) -> Request:
        _require_api_url(url)
        request = Request(url, headers=dict(headers or {}))
        if self._token:
            request.add_unredirected_header("Authorization", f"Bearer {self._token}")
        return request

    def open(self, url: str, headers: Mapping[str, str] | None = None) -> IO[bytes]:
        reply: IO[bytes] = self._opener.open(self.request(url, headers), timeout=self._timeout)
        return reply

    def json(self, url: str) -> Any:
        for attempt in range(self._attempts):
            try:
                with self.open(url, {"Accept": "application/json"}) as reply:
                    return json.loads(reply.read())
            except HTTPError as error:
                error.close()
                if error.code not in RETRY_STATUSES or attempt + 1 == self._attempts:
                    raise
            except (URLError, TimeoutError, ConnectionError):
                if attempt + 1 == self._attempts:
                    raise
            self._sleep(min(30.0, 2.0**attempt))
        raise RuntimeError("unreachable")

    def download(self, url: str, destination: Path, size: int, sha256: str) -> int:
        """Stream to a temporary file, resuming by range; publish only verified bytes."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        partial.unlink(missing_ok=True)
        digest = hashlib.sha256()
        have = transferred = 0
        try:
            with partial.open("xb") as stream:
                for attempt in range(self._attempts):
                    headers = {"Range": f"bytes={have}-"} if have else {}
                    try:
                        with self.open(url, headers) as reply:
                            status = getattr(reply, "status", 200)
                            if have and status != 206:
                                raise ValueError("resumed download did not answer 206")
                            while block := reply.read(CHUNK):
                                have += len(block)
                                transferred += len(block)
                                if have > size:
                                    raise ValueError("download exceeds the listed size")
                                digest.update(block)
                                stream.write(block)
                        if have == size:
                            break
                        raise ConnectionError("download ended before the listed size")
                    except HTTPError as error:
                        error.close()
                        if error.code not in RETRY_STATUSES or attempt + 1 == self._attempts:
                            raise
                    except (URLError, TimeoutError, ConnectionError):
                        if attempt + 1 == self._attempts:
                            raise
                    self._sleep(min(30.0, 2.0**attempt))
                stream.flush()
                os.fsync(stream.fileno())
            if have != size or digest.hexdigest() != sha256:
                raise ValueError(f"{destination.name}: size or SHA-256 differs from the listing")
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(destination)
        return transferred


def listing(hub: Hub, source: Source) -> dict[str, dict[str, Any]]:
    """Size and LFS SHA-256 of each spec file at the pinned revision."""
    url = f"https://{API_HOST}/api/datasets/{source.dataset}/revision/{source.revision}?blobs=true"
    info = hub.json(url)
    if info.get("sha") != source.revision:
        raise ValueError(f"{source.dataset}: the pinned revision resolved differently")
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    found = {}
    for name in source.files:
        item = siblings.get(name)
        if item is None or not item.get("lfs"):
            raise ValueError(f"{source.dataset}: {name} is not an LFS file at the pinned revision")
        found[name] = {"size_bytes": int(item["size"]), "sha256": str(item["lfs"]["sha256"])}
    return found


def _verified(path: Path, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and file_sha256(path) == sha256


def download(
    spec: Spec,
    build: Path,
    *,
    token: str | None,
    reuse: Sequence[Path] = (),
    hub_factory: Callable[[str | None], Hub] = Hub,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    receipts_path = build / "download-receipts.json"
    if receipts_path.exists() and json.loads(receipts_path.read_text())["status"] == "completed":
        raise FileExistsError("downloads are already complete for this build")
    started = time.perf_counter()
    reusable: dict[tuple[str, str, str], Path] = {}
    for path in reuse:
        for entry in json.loads(path.read_text())["files"]:
            if entry.get("verified"):
                key = (entry["source"], entry["revision"], entry["filename"])
                reusable[key] = Path(entry["path"])
    planned = []
    for source in spec.sources:
        hub = hub_factory(token if source.gated else None)
        for name, listed in listing(hub, source).items():
            planned.append((source, name, listed, hub))
    total = sum(item[2]["size_bytes"] for item in planned)
    if total > spec.max_download_bytes:
        raise RuntimeError(f"the plan lists {total:,} bytes, above the approved maximum")
    build.mkdir(parents=True, exist_ok=True)
    pending = 0
    for source, name, listed, _ in planned:
        target = build / "raw" / source.domain / name
        if not target.exists() and (source.dataset, source.revision, name) not in reusable:
            pending += listed["size_bytes"]
    free_before = require_free(build, pending, spec, "download")
    files: list[dict[str, Any]] = []
    for source, name, listed, hub in planned:
        size, sha256 = listed["size_bytes"], listed["sha256"]
        target = build / "raw" / source.domain / name
        record: dict[str, Any] = {
            "domain": source.domain,
            "source": source.dataset,
            "revision": source.revision,
            "filename": name,
            "path": str(target),
            "size_bytes": size,
            "sha256": sha256,
            "license": source.license,
            "terms": source.terms,
            "terms_accepted_on": source.terms_accepted_on,
            "attribution": source.attribution,
        }
        began = time.perf_counter()
        prior = reusable.get((source.dataset, source.revision, name))
        if target.exists():
            if not _verified(target, size, sha256):
                raise ValueError(f"{target} exists with different bytes; remove it to refetch")
            record.update(method="present", bytes_transferred=0)
        elif prior is not None and _verified(prior, size, sha256):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(prior, target)
            record.update(method="hard-linked", reused_from=str(prior), bytes_transferred=0)
        else:
            url = f"https://{API_HOST}/datasets/{source.dataset}/resolve/{source.revision}/{name}"
            moved = hub.download(url, target, size, sha256)
            record.update(method="downloaded", bytes_transferred=moved)
        if not _verified(target, size, sha256):
            raise ValueError(f"{target}: bytes differ after publication")
        record.update(verified=True, elapsed_seconds=time.perf_counter() - began)
        files.append(record)
        atomic_json(receipts_path, _receipts(spec, files, "running", started, free_before))
        progress(f"{record['method']} {source.domain} {name} ({size:,} bytes)")
    report = _receipts(spec, files, "completed", started, free_before)
    atomic_json(receipts_path, report)
    return report


def _receipts(
    spec: Spec, files: list[dict[str, Any]], status: str, started: float, free: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_at": utc_now(),
        "spec_sha256": spec.sha256,
        "files": files,
        "listed_bytes": sum(item["size_bytes"] for item in files),
        "bytes_transferred": sum(item["bytes_transferred"] for item in files),
        "free_bytes_before": free,
        "elapsed_seconds": time.perf_counter() - started,
        "paid_runs": 0,
    }


def completed_receipts(spec: Spec, build: Path) -> dict[str, Any]:
    receipts = json.loads((build / "download-receipts.json").read_text())
    if receipts.get("status") != "completed" or receipts.get("spec_sha256") != spec.sha256:
        raise ValueError("completed download receipts for this exact spec are required")
    order = [(s.domain, name) for s in spec.sources for name in s.files]
    if [(item["domain"], item["filename"]) for item in receipts["files"]] != order:
        raise ValueError("download receipts do not list the spec's files in order")
    return dict(receipts)


def licenses(
    spec: Spec,
    build: Path,
    *,
    client: HuggingFaceClient,
    workers: int = 4,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    stack_licenses = importlib.import_module("tools.stack_licenses")
    receipts = completed_receipts(spec, build)
    containers = [item for item in receipts["files"] if item["domain"] == "python"]
    report: dict[str, Any] = stack_licenses.recover(
        containers=containers,
        revision=spec.license_metadata["revision"],
        output=build / "licenses",
        terms_accepted_on=spec.license_metadata["terms_accepted_on"],
        client=client,
        workers=workers,
        progress=progress,
    )
    return report


def license_mapping(
    spec: Spec, build: Path, receipts: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    report = build / "licenses" / "report.json"
    _, identity = load_mapping(report)
    containers = [
        {"filename": item["filename"], "sha256": item["sha256"]}
        for item in receipts["files"]
        if item["domain"] == "python"
    ]
    found = [
        {"filename": item["filename"], "sha256": item["sha256"]}
        for item in identity.get("containers") or []
    ]
    if (
        found != containers
        or identity["dataset"] != spec.license_metadata["dataset"]
        or identity["revision"] != spec.license_metadata["revision"]
    ):
        raise ValueError("the license mapping was built for different containers or upstream")
    return report, identity


@dataclass(frozen=True)
class Unit:
    name: str
    domain: str
    file_index: int
    filename: str
    path: str
    first_group: int
    groups: int


def plan_units(spec: Spec, receipts: Mapping[str, Any]) -> list[Unit]:
    parquet = importlib.import_module("pyarrow.parquet")
    units = []
    for item in receipts["files"]:
        domain = item["domain"]
        index = spec.source(domain).files.index(item["filename"])
        with parquet.ParquetFile(item["path"]) as handle:
            count = int(handle.metadata.num_row_groups)
        for first in range(0, count, spec.unit_row_groups):
            groups = min(spec.unit_row_groups, count - first)
            units.append(
                Unit(
                    f"{domain}-{index:03d}-{first:05d}",
                    domain,
                    index,
                    item["filename"],
                    item["path"],
                    first,
                    groups,
                )
            )
    return units


_worker: dict[str, Any] = {}


def _initialize(
    tokenizer_path: str, contamination: list[str], license_report: str | None, web_license: str
) -> None:
    _worker["encoder"] = FastEncoder(ByteBPE.load(Path(tokenizer_path)))
    _worker["index"] = ContaminationIndex(contamination)
    _worker["license_report"] = license_report
    _worker["web_license"] = web_license
    _worker["mapping"] = None


def _mapping() -> Mapping[tuple[str, str], tuple[str, ...]]:
    if _worker["mapping"] is None:
        if _worker["license_report"] is None:
            raise RuntimeError("Python units need the license mapping")
        _worker["mapping"], _ = load_mapping(Path(_worker["license_report"]))
    mapping: Mapping[tuple[str, str], tuple[str, ...]] = _worker["mapping"]
    return mapping


def _unit_rows(unit: Unit) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = importlib.import_module("pyarrow.parquet")
    columns = (
        ["text", "id", "url"]
        if unit.domain == "web"
        else ["content", "id", "max_stars_repo_name", "max_stars_repo_path"]
    )
    with parquet.ParquetFile(unit.path, arrow_extensions_enabled=False) as handle:
        row = sum(handle.metadata.row_group(g).num_rows for g in range(unit.first_group))
        for group in range(unit.first_group, unit.first_group + unit.groups):
            table = handle.read_row_group(group, columns=columns, use_threads=False)
            for item in table.to_pylist():
                yield row, item
                row += 1


def review_unit(unit: Unit, directory: str) -> dict[str, Any]:
    """Classify, filter and encode one unit; outputs are published atomically."""
    started = time.perf_counter()
    out = Path(directory)
    encoder: FastEncoder = _worker["encoder"]
    index: ContaminationIndex = _worker["index"]
    config = FilterConfig()
    tokens_path = out / f"{unit.name}.tokens"
    rows_path = out / f"{unit.name}.jsonl"
    temporary_tokens = tokens_path.with_name(tokens_path.name + ".tmp")
    temporary_rows = rows_path.with_name(rows_path.name + ".tmp")
    offset = rows = 0
    try:
        with (
            temporary_tokens.open("wb") as tokens,
            temporary_rows.open("w", encoding="utf-8") as lines,
        ):
            for row, item in _unit_rows(unit):
                text = item[TEXT_COLUMNS[unit.domain]]
                if not isinstance(text, str):
                    raise ValueError(f"{unit.name}: row {row} has no text")
                size = len(text.encode("utf-8"))
                listed: tuple[str, ...] | None = None
                if unit.domain == "web":
                    classification = HeaderLicense(
                        "permissive", _worker["web_license"], "owner-reviewed web source terms"
                    )
                else:
                    key = (str(item["max_stars_repo_name"]), str(item["max_stars_repo_path"]))
                    listed = _mapping().get(key)
                    classification = classify_file(text, listed)
                via_repository = classification.category == "repository_disallowed" or (
                    classification.evidence == REPOSITORY_EVIDENCE
                )
                record: dict[str, Any] = {
                    "row": row,
                    "id": str(item.get("id")),
                    "sha256": content_hash(text),
                    "bytes": size,
                    "license": classification.license_id,
                    "license_class": classification.category,
                    "license_basis": (
                        "repository_metadata" if via_repository else "inline_or_source_terms"
                    ),
                    "repository_licenses": list(repository_license_ids(listed))
                    if via_repository
                    else None,
                    "metadata_found": None if unit.domain == "web" else listed is not None,
                    "repository": item.get("max_stars_repo_name"),
                    "path": item.get("max_stars_repo_path"),
                    "url": item.get("url"),
                    "contaminated": None,
                    "tokens": 0,
                    "offset": None,
                }
                if classification.category != "permissive":
                    record["reason"] = "license_" + classification.category
                else:
                    document = Document(record["id"], unit.domain, text, unit.domain)
                    record["reason"] = rule_reason(document, config)
                if record["reason"] == "kept":
                    # A contaminated document is never kept, so it is not encoded.
                    record["contaminated"] = index.contains(text)
                    if not record["contaminated"]:
                        encoded = np.asarray(encoder.encode(text), dtype="<u2")
                        tokens.write(encoded.tobytes())
                        record.update(tokens=int(encoded.size), offset=offset)
                        offset += int(encoded.size)
                lines.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
            tokens.flush()
            os.fsync(tokens.fileno())
            lines.flush()
            os.fsync(lines.fileno())
        temporary_tokens.replace(tokens_path)
        temporary_rows.replace(rows_path)
    except BaseException:
        temporary_tokens.unlink(missing_ok=True)
        temporary_rows.unlink(missing_ok=True)
        raise
    summary = {
        **asdict(unit),
        "rows": rows,
        "tokens": offset,
        "tokens_sha256": file_sha256(tokens_path),
        "rows_sha256": file_sha256(rows_path),
        "elapsed_seconds": time.perf_counter() - started,
        "process": os.getpid(),
    }
    atomic_json(out / f"{unit.name}.json", summary)
    return summary


def _complete(unit: Unit, directory: Path) -> dict[str, Any] | None:
    """A unit whose record and outputs agree; anything partial is redone."""
    record_path = directory / f"{unit.name}.json"
    if not record_path.exists():
        return None
    record: dict[str, Any] = json.loads(record_path.read_text())
    if (
        {key: record.get(key) for key in asdict(unit)} != asdict(unit)
        or file_sha256(directory / f"{unit.name}.tokens") != record["tokens_sha256"]
        or file_sha256(directory / f"{unit.name}.jsonl") != record["rows_sha256"]
    ):
        return None
    return record


def review(
    units: Sequence[Unit],
    directory: Path,
    *,
    tokenizer_path: Path,
    contamination: list[str],
    license_report: Path | None,
    web_license: str,
    workers: int,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    pending = []
    for unit in units:
        record = _complete(unit, directory)
        if record is None:
            pending.append(unit)
        else:
            done[unit.name] = record
    if pending:
        progress(f"reviewing {len(pending)} of {len(units)} units with {workers} workers")
        module = importlib.import_module("tools.rehearsal_build")
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=module._initialize,
            initargs=(
                str(tokenizer_path),
                contamination,
                None if license_report is None else str(license_report),
                web_license,
            ),
        ) as pool:
            futures = {pool.submit(module.review_unit, u, str(directory)): u for u in pending}
            try:
                for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    record = future.result()
                    done[record["name"]] = record
                    if number % 10 == 0 or number == len(pending):
                        progress(f"reviewed {number}/{len(pending)} units")
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    return [done[unit.name] for unit in units]


@dataclass
class Pick:
    unit: str
    offset: int
    content_tokens: int
    row: dict[str, Any]
    domain: str
    filename: str
    cut: bool


def _tally() -> dict[str, Any]:
    return {
        "sample_documents": 0,
        "sample_text_bytes": 0,
        "decisions": {name: Counter() for name in (*LICENSE_REASONS, *FILTER_REASONS)},
        "license_classes": {},
        "rejected_repository_license_id_counts": Counter(),
        "repository_license_id_outcomes": {},
        "repository_metadata_lookups": Counter(),
        "kept_license_id_counts": Counter(),
        "kept_license_basis": {},
        "splits": {},
    }


def assemble(
    spec: Spec, units: Sequence[Unit], records: Sequence[Mapping[str, Any]], directory: Path
) -> tuple[dict[str, list[Pick]], dict[str, Any], dict[str, Any]]:
    """Global dedup, hash splits and exact quotas, in spec order."""
    seen: set[str] = set()
    tallies = {domain: _tally() for domain in DOMAINS}
    picks: dict[str, list[Pick]] = {stream.name: [] for stream in spec.streams}
    filled = dict.fromkeys(picks, 0)
    unused: dict[str, Counter[str]] = {name: Counter() for name in picks}
    cuts: dict[str, dict[str, Any]] = {}
    for unit, record in zip(units, records, strict=True):
        count = 0
        with (directory / f"{unit.name}.jsonl").open(encoding="utf-8") as lines:
            for line in lines:
                count += 1
                row = json.loads(line)
                tally = tallies[unit.domain]
                tally["sample_documents"] += 1
                tally["sample_text_bytes"] += row["bytes"]
                if unit.domain == "python":
                    found = "found" if row["metadata_found"] else "absent"
                    tally["repository_metadata_lookups"][found] += 1
                classes = tally["license_classes"].setdefault(row["license_class"], Counter())
                classes["files"] += 1
                classes["text_bytes"] += row["bytes"]
                if row["license_basis"] == "repository_metadata":
                    outcome = "admitted" if row["license_class"] == "permissive" else "rejected"
                    for item in row["repository_licenses"] or ():
                        ids = tally["repository_license_id_outcomes"].setdefault(item, Counter())
                        ids[outcome + "_files"] += 1
                        ids[outcome + "_text_bytes"] += row["bytes"]
                        if outcome == "rejected" and item not in REPOSITORY_ALLOWLIST:
                            tally["rejected_repository_license_id_counts"][item] += 1
                reason = row["reason"]
                if reason == "kept":
                    if row["sha256"] in seen:
                        reason = "duplicate"
                    elif row["contaminated"]:
                        reason = "contamination"
                decision = tally["decisions"][reason]
                decision["files"] += 1
                decision["text_bytes"] += row["bytes"]
                if reason != "kept":
                    continue
                seen.add(row["sha256"])
                for item in (row["license"] or "unresolved").split(" AND "):
                    tally["kept_license_id_counts"][item] += 1
                basis = tally["kept_license_basis"].setdefault(row["license_basis"], Counter())
                basis["files"] += 1
                basis["text_bytes"] += row["bytes"]
                split = document_split(row["sha256"], web=unit.domain == "web")
                parts = tally["splits"].setdefault(split, Counter())
                parts["documents"] += 1
                parts["text_bytes"] += row["bytes"]
                parts["tokens"] += row["tokens"]
                stream = spec.stream_for(unit.domain, split)
                length = 1 + row["tokens"]
                room = None if stream.tokens is None else stream.tokens - filled[stream.name]
                if room is not None and room <= 0:
                    unused[stream.name]["documents"] += 1
                    unused[stream.name]["tokens"] += length
                    continue
                cut = room is not None and length > room
                take = room - 1 if cut and room is not None else row["tokens"]
                kept = {key: row[key] for key in PROVENANCE}
                picks[stream.name].append(
                    Pick(unit.name, row["offset"], take, kept, unit.domain, unit.filename, cut)
                )
                filled[stream.name] += 1 + take
                if cut:
                    cuts[stream.name] = {
                        "unit": unit.name,
                        "row": row["row"],
                        "id": row["id"],
                        "sha256": row["sha256"],
                        "document_tokens": length,
                        "kept_tokens": 1 + take,
                    }
            if count != record["rows"]:
                raise ValueError(f"{unit.name}: row count differs from its record")
    short = {
        s.name: s.tokens - filled[s.name]
        for s in spec.streams
        if s.tokens is not None and filled[s.name] != s.tokens
    }
    if short:
        raise RuntimeError(f"quotas not met; tokens missing per stream: {short}")
    streams = {
        s.name: {
            "domain": s.domain,
            "split": s.split,
            "quota_tokens": s.tokens,
            "tokens": filled[s.name],
            "documents": len(picks[s.name]),
            "bos_tokens": len(picks[s.name]),
            "cut": cuts.get(s.name),
            "unused_documents": unused[s.name]["documents"],
            "unused_tokens": unused[s.name]["tokens"],
        }
        for s in spec.streams
    }
    domains = {domain: _finish(tally) for domain, tally in tallies.items()}
    return picks, streams, domains


def _finish(tally: dict[str, Any]) -> dict[str, Any]:
    kept = tally["decisions"]["kept"]
    return {
        "sample_documents": tally["sample_documents"],
        "sample_text_bytes": tally["sample_text_bytes"],
        "primary_decisions": {
            name: {"files": c["files"], "text_bytes": c["text_bytes"]}
            for name, c in tally["decisions"].items()
        },
        "license_classes": {k: dict(v) for k, v in tally["license_classes"].items()},
        "rejected_repository_license_id_counts": dict(
            sorted(
                tally["rejected_repository_license_id_counts"].items(), key=lambda x: (-x[1], x[0])
            )
        ),
        "repository_license_id_outcomes": {
            name: dict(value)
            for name, value in sorted(
                tally["repository_license_id_outcomes"].items(),
                key=lambda x: (-(x[1]["admitted_files"] + x[1]["rejected_files"]), x[0]),
            )
        },
        "repository_metadata_lookups": dict(tally["repository_metadata_lookups"]),
        "kept_documents": kept["files"],
        "kept_text_bytes": kept["text_bytes"],
        "kept_license_id_counts": dict(
            sorted(tally["kept_license_id_counts"].items(), key=lambda x: (-x[1], x[0]))
        ),
        "kept_license_basis": {k: dict(v) for k, v in tally["kept_license_basis"].items()},
        "splits": {k: dict(v) for k, v in sorted(tally["splits"].items())},
        "retention_bytes": kept["text_bytes"] / max(tally["sample_text_bytes"], 1),
    }


def write_streams(
    spec: Spec, picks: Mapping[str, list[Pick]], units: Path, output: Path
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Write every stream's shards and the selected-document provenance list."""
    vocab, bos = spec.tokenizer["vocab_size"], spec.tokenizer["bos_id"]
    identity = spec.tokenizer["identity"]
    arrays: dict[str, np.ndarray[Any, np.dtype[np.uint16]]] = {}

    def tokens(pick: Pick) -> list[int]:
        if pick.unit not in arrays:
            path = units / f"{pick.unit}.tokens"
            arrays[pick.unit] = (
                np.memmap(path, dtype="<u2", mode="r")
                if path.stat().st_size
                else np.zeros(0, dtype="<u2")
            )
        values: list[int] = arrays[pick.unit][
            pick.offset : pick.offset + pick.content_tokens
        ].tolist()
        return values

    shards: dict[str, list[dict[str, Any]]] = {}
    documents = output / "documents.jsonl"
    with documents.open("x", encoding="utf-8") as listing_stream:
        for stream in spec.streams:
            groups: list[list[Pick]] = [[]]
            size = 0
            for pick in picks[stream.name]:
                length = 1 + pick.content_tokens
                if groups[-1] and size + length > spec.shard_max_tokens:
                    groups.append([])
                    size = 0
                groups[-1].append(pick)
                size += length
            shards[stream.name] = []
            for number, group in enumerate(groups):
                name = f"{stream.name}-{number:05d}.bin"
                path = output / "shards" / stream.name / name
                manifest = write_shard(
                    path,
                    (tokens(p) for p in group),
                    vocab_size=vocab,
                    tokenizer_hash=identity,
                    bos_id=bos,
                )
                shards[stream.name].append(
                    {
                        "file": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "manifest_sha256": file_sha256(manifest),
                        "documents": len(group),
                        "tokens": sum(1 + p.content_tokens for p in group),
                    }
                )
                for position, pick in enumerate(group):
                    row = pick.row
                    listing_stream.write(
                        json.dumps(
                            {
                                "stream": stream.name,
                                "shard": name,
                                "index": position,
                                "tokens": 1 + pick.content_tokens,
                                "cut": pick.cut,
                                "domain": pick.domain,
                                "source_file": pick.filename,
                                **row,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    return shards, file_sha256(documents)


def build_identity(spec: Spec, shards: Mapping[str, list[dict[str, Any]]]) -> str:
    """One hash naming the spec, tokenizer and every shard's bytes, in stream order."""
    value = {
        "spec_sha256": spec.sha256,
        "tokenizer": spec.tokenizer["identity"],
        "shards": [
            [stream.name, shard["file"], shard["sha256"], shard["tokens"]]
            for stream in spec.streams
            for shard in shards[stream.name]
        ],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(
    spec: Spec,
    build_dir: Path,
    *,
    workers: int,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    output = build_dir / "output"
    if output.exists():
        raise FileExistsError("the build output already exists; verify it instead")
    started = time.perf_counter()
    receipts = completed_receipts(spec, build_dir)
    for item in receipts["files"]:
        if not _verified(Path(item["path"]), item["size_bytes"], item["sha256"]):
            raise ValueError(f"{item['filename']}: raw bytes differ from the receipt")
    report_path, license_identity = license_mapping(spec, build_dir, receipts)
    tokenizer_path = Path(spec.tokenizer["path"])
    tokenizer = _load_tokenizer(
        tokenizer_path,
        expected_identity=spec.tokenizer["identity"],
        expected_file_sha256=spec.tokenizer["file_sha256"],
        expected_vocab_size=spec.tokenizer["vocab_size"],
        expected_bos_id=spec.tokenizer["bos_id"],
    )
    texts, coverage = contamination_inputs(
        Path(spec.contamination["evalplus_directory"]),
        Path(spec.contamination["arc_directory"]),
        Path(spec.contamination["humaneval"]),
    )
    units = plan_units(spec, receipts)
    raw_bytes = sum(item["size_bytes"] for item in receipts["files"])
    free_before = require_free(build_dir, 4 * raw_bytes, spec, "build")
    unit_dir = build_dir / "units"
    timings: dict[str, float] = {"verify_inputs": time.perf_counter() - started}
    began = time.perf_counter()
    records = review(
        units,
        unit_dir,
        tokenizer_path=tokenizer_path,
        contamination=texts,
        license_report=report_path,
        web_license=spec.source("web").license,
        workers=workers,
        progress=progress,
    )
    timings["review"] = time.perf_counter() - began
    began = time.perf_counter()
    picks, streams, domains = assemble(spec, units, records, unit_dir)
    timings["assemble"] = time.perf_counter() - began
    progress("assembled: " + json.dumps({k: v["tokens"] for k, v in streams.items()}))
    began = time.perf_counter()
    staging = build_dir / "output.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shards, documents_sha = write_streams(spec, picks, unit_dir, staging)
    timings["write_shards"] = time.perf_counter() - began
    for name, value in streams.items():
        value["shards"] = shards[name]
    commit, dirty = git_state()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "name": spec.name,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "spec_sha256": spec.sha256,
        "tokenizer": {**spec.tokenizer, "vocab_size": tokenizer.vocab_size},
        "token_accounting": "shard tokens, one BOS per document included",
        "boundary_rule": (
            "documents are taken whole in spec order; the one crossing a quota is cut to "
            "the tokens that fill it exactly"
        ),
        "download_receipts_sha256": file_sha256(build_dir / "download-receipts.json"),
        "sources": [asdict(source) for source in spec.sources],
        "license_mapping": license_identity,
        "license_policy": (
            "header-classifier-v3; explicit header first; absent header uses verified "
            "repository license metadata, admitted only when every listed ID is on the D-05 "
            "allowlist; web documents carry the owner-reviewed source terms"
        ),
        "contamination_coverage": coverage,
        "filter_configuration": asdict(FilterConfig()),
        "dedup": "exact content SHA-256 across the whole build, first occurrence in spec order",
        "split_rule": "scratchcoder.data.filters.document_split on the content SHA-256",
        "units": {
            "count": len(units),
            "row_groups_per_unit": spec.unit_row_groups,
            "tokens_written": sum(record["tokens"] for record in records),
        },
        "domains": domains,
        "streams": streams,
        "documents": {"file": "documents.jsonl", "sha256": documents_sha},
        "build_identity": build_identity(spec, shards),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "free_bytes_before": free_before,
        "timings_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "paid_runs": 0,
        "final_training_admission": False,
        "limitations": [
            "the final verified task-pool holdout is not built; repeat its exclusion before "
            "training and rebuild if it removes any document",
            "repository license lists are The Stack's detected licenses, not a legal warranty",
            "exact-hash deduplication only; the sources' own near-deduplication is relied on",
        ],
    }
    atomic_json(staging / "report.json", report)
    staging.replace(output)
    verification = verify(spec, build_dir)
    report["verification"] = {
        "status": verification["status"],
        "sha256": file_sha256(output / "verification.json"),
    }
    return report


def _source_text(path: Path, domain: str, row: int) -> str:
    parquet = importlib.import_module("pyarrow.parquet")
    with parquet.ParquetFile(path, arrow_extensions_enabled=False) as handle:
        start = 0
        for group in range(handle.metadata.num_row_groups):
            rows = handle.metadata.row_group(group).num_rows
            if row < start + rows:
                table = handle.read_row_group(group, columns=[TEXT_COLUMNS[domain]])
                text: str = table.column(0)[row - start].as_py()
                return text
            start += rows
    raise IndexError(f"row {row} is past the end of {path}")


def verify(spec: Spec, build_dir: Path) -> dict[str, Any]:
    """Re-check every shard and stream invariant; write verification.json."""
    output = build_dir / "output"
    report = json.loads((output / "report.json").read_text())
    if report.get("status") != "completed" or report.get("spec_sha256") != spec.sha256:
        raise ValueError("a completed build report for this exact spec is required")
    receipts = completed_receipts(spec, build_dir)
    paths = {(item["domain"], item["filename"]): Path(item["path"]) for item in receipts["files"]}
    tokenizer = ByteBPE.load(Path(spec.tokenizer["path"]))
    if tokenizer.sha256 != spec.tokenizer["identity"]:
        raise ValueError("tokenizer identity differs from the spec")
    encoder = FastEncoder(tokenizer)
    vocab, bos = spec.tokenizer["vocab_size"], spec.tokenizer["bos_id"]
    if file_sha256(output / "documents.jsonl") != report["documents"]["sha256"]:
        raise ValueError("documents.jsonl differs from the report")
    listed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with (output / "documents.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            row = json.loads(line)
            listed.setdefault((row["stream"], row["shard"]), []).append(row)
    hashes: dict[str, set[str]] = {}
    checks: Counter[str] = Counter()
    for stream in spec.streams:
        declared = report["streams"][stream.name]
        total = 0
        hashes[stream.name] = set()
        for shard in declared["shards"]:
            path = output / shard["file"]
            opened = TokenShard(
                path, tokenizer_hash=spec.tokenizer["identity"], vocab_size=vocab, bos_id=bos
            )
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"{shard['file']}: SHA-256 differs from the report")
            values = (
                np.memmap(path, dtype="<u2", mode="r")
                if path.stat().st_size
                else np.zeros(0, dtype="<u2")
            )
            starts = np.fromiter((o for o, _ in opened.offsets), dtype=np.int64)
            lengths = [length for _, length in opened.offsets]
            if starts.size and (values[starts] != bos).any():
                raise ValueError(f"{shard['file']}: a document does not start with BOS")
            if values.size and int(values.max()) >= vocab:
                raise ValueError(f"{shard['file']}: token outside the vocabulary")
            if (bos_count := int((values == bos).sum())) != len(opened):
                raise ValueError(
                    f"{shard['file']}: {bos_count} BOS tokens, {len(opened)} documents"
                )
            rows = listed.get((stream.name, Path(shard["file"]).name), [])
            if [r["tokens"] for r in rows] != lengths or len(rows) != shard["documents"]:
                raise ValueError(f"{shard['file']}: documents.jsonl disagrees with the shard")
            if sum(lengths) != shard["tokens"]:
                raise ValueError(f"{shard['file']}: token count differs from the report")
            total += shard["tokens"]
            spot = {0, len(rows) - 1} | {i for i, r in enumerate(rows) if r["cut"]}
            for position in sorted(i for i in spot if 0 <= i < len(rows)):
                row = rows[position]
                text = _source_text(
                    paths[(row["domain"], row["source_file"])], row["domain"], row["row"]
                )
                if content_hash(text) != row["sha256"]:
                    raise ValueError(f"{shard['file']}: source text hash differs at {position}")
                expected = [bos, *encoder.encode(text)][: row["tokens"]]
                offset, length = opened.offsets[position]
                if values[offset : offset + length].tolist() != expected:
                    raise ValueError(f"{shard['file']}: tokens differ from re-encoding")
                checks["re_encoded_documents"] += 1
            for row in rows:
                if row["sha256"] in hashes[stream.name]:
                    raise ValueError(f"{stream.name}: document {row['sha256']} repeats")
                hashes[stream.name].add(row["sha256"])
            checks["shards"] += 1
            checks["documents"] += len(rows)
            checks["tokens"] += sum(lengths)
        if total != declared["tokens"] or (stream.tokens is not None and total != stream.tokens):
            raise ValueError(f"{stream.name}: {total} tokens, quota {stream.tokens}")
    names = [stream.name for stream in spec.streams]
    overlaps = {
        f"{a}|{b}": len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"streams share documents: {overlaps}")
    shards = {name: report["streams"][name]["shards"] for name in names}
    if build_identity(spec, shards) != report["build_identity"]:
        raise ValueError("build identity differs from the report")
    result = {
        "status": "passed",
        "created_at": utc_now(),
        "build_identity": report["build_identity"],
        "checks": dict(checks),
        "document_hash_overlaps": overlaps,
        "stream_tokens": {name: report["streams"][name]["tokens"] for name in names},
    }
    atomic_json(output / "verification.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "licenses", "build", "verify"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True, help="build directory")
    parser.add_argument(
        "--reuse",
        type=Path,
        action="append",
        default=[],
        help="earlier verified download receipts whose files may be hard-linked",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    token = os.environ.get("HF_TOKEN") or None
    if args.stage == "download":
        result = download(spec, args.build, token=token, reuse=args.reuse)
        print(json.dumps({k: result[k] for k in ("listed_bytes", "bytes_transferred")}))
    elif args.stage == "licenses":
        recovered = licenses(
            spec, args.build, client=HuggingFaceClient(token or ""), workers=min(args.workers, 4)
        )
        print(json.dumps({k: recovered[k] for k in ("totals", "join", "mapping")}, indent=1))
    elif args.stage == "build":
        report = build(spec, args.build, workers=args.workers)
        print(
            json.dumps(
                {
                    "build_identity": report["build_identity"],
                    "streams": {k: v["tokens"] for k, v in report["streams"].items()},
                    "verification": report["verification"],
                },
                indent=1,
            )
        )
    else:
        print(json.dumps(verify(spec, args.build), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
