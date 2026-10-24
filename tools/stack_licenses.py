"""Recover upstream repository licenses for the local StarCoderData Python sample.

StarCoderData keeps repository/path provenance but no license column. The Stack
dedup v1.2, from which it was built, records ``max_stars_repo_licenses`` for the
same files. This tool reads only the join keys (``max_stars_repo_name``,
``max_stars_repo_path``) and that license column from the pinned ``data/python``
Parquet files with HTTPS range requests. Every fetched byte range is checked
against the ``content`` column chunks and the run fails if any byte overlaps
one, so file contents are never downloaded. Bytes outside the three selected
column chunks (Parquet's speculative 64 KiB footer read) are counted and reported.

The join is the exact (repository, path) key. Several upstream rows with one key
are admitted only when their license lists agree. A sample file whose path is
absent upstream but whose repository is present is reported as a diagnostic and
is never admitted. The output mapping feeds ``tools.real_samples
--repository-licenses``; the D-05 allowlist is applied there, not here.

The Hugging Face token is read from ``HF_TOKEN``, sent only to huggingface.co
(never across a redirect to the storage CDN) and never written or printed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import io
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.manifest import HTTPSOnlyRedirectHandler
from scratchcoder.data.shards import file_sha256
from tools.header_licenses import repository_license_ids
from tools.real_samples import parquet_rows, sample_rows

DATASET = "bigcode/the-stack-dedup"
PYTHON_PREFIX = "data/python/"
NAME_COLUMN = "max_stars_repo_name"
PATH_COLUMN = "max_stars_repo_path"
LICENSE_COLUMN = "max_stars_repo_licenses"
COLUMNS = (NAME_COLUMN, PATH_COLUMN, LICENSE_COLUMN)
CONTENT_COLUMN = "content"
API_HOST = "huggingface.co"
PYTHON_SAMPLE_TARGET = 1_000_000_000
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

Fetch = Callable[[str, int, int], bytes]
"""(url, first byte, last byte inclusive) -> exactly those bytes."""

Key = tuple[str, str]


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    lfs_sha256: str


@dataclass
class FileScan:
    """Join results and transfer accounting for one upstream Parquet file."""

    path: str
    rows: int = 0
    row_groups: int = 0
    footer_sha256: str = ""
    bytes_fetched: int = 0
    bytes_outside_selected_columns: int = 0
    requests: int = 0
    rows_without_license: int = 0
    license_ids: Counter[str] = field(default_factory=Counter)
    matches: dict[Key, list[tuple[tuple[str, ...], int]]] = field(default_factory=dict)
    repository_licenses: dict[str, Counter[tuple[str, ...]]] = field(default_factory=dict)


def _require_api_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != API_HOST or parsed.username is not None:
        raise ValueError(
            "Hugging Face requests must use https://huggingface.co without credentials"
        )


class HuggingFaceClient:
    """Authenticated HTTPS reads; the token never crosses a redirect."""

    def __init__(
        self,
        token: str,
        *,
        attempts: int = 6,
        timeout: float = 60,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token:
            raise ValueError("HF_TOKEN is required for the gated dataset")
        self._token = token
        self._opener = build_opener(HTTPSOnlyRedirectHandler())
        self._attempts, self._timeout, self._sleep = attempts, timeout, sleep
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        self.requests = 0

    def request(self, url: str, headers: Mapping[str, str] | None = None) -> Request:
        _require_api_url(url)
        request = Request(url, headers=dict(headers or {}))
        # Unredirected: the signed CDN URL that /resolve/ redirects to never sees it.
        request.add_unredirected_header("Authorization", f"Bearer {self._token}")
        return request

    def _read(self, url: str, headers: Mapping[str, str] | None, expected: int | None) -> bytes:
        for attempt in range(self._attempts):
            try:
                with self._opener.open(self.request(url, headers), timeout=self._timeout) as reply:
                    if expected is not None and reply.status != 206:
                        raise ValueError(f"range request answered HTTP {reply.status}, not 206")
                    body: bytes = reply.read()
            except HTTPError as error:
                error.close()
                if error.code not in RETRY_STATUSES or attempt + 1 == self._attempts:
                    raise
            except (URLError, TimeoutError, ConnectionError):
                if attempt + 1 == self._attempts:
                    raise
            else:
                if expected is not None and len(body) != expected:
                    raise ValueError("range response length differs from the requested range")
                with self._lock:
                    self.bytes_fetched += len(body)
                    self.requests += 1
                return body
            self._sleep(min(30.0, 2.0**attempt))
        raise RuntimeError("unreachable")

    def json(self, url: str) -> Any:
        return json.loads(self._read(url, {"Accept": "application/json"}, None))

    def fetch(self, url: str, first: int, last: int) -> bytes:
        if not 0 <= first <= last:
            raise ValueError("invalid byte range")
        return self._read(url, {"Range": f"bytes={first}-{last}"}, last - first + 1)


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; every read is one recorded range."""

    def __init__(self, url: str, size: int, fetch: Fetch) -> None:
        super().__init__()
        self.url, self.size, self._fetch = url, size, fetch
        self.position = 0
        self.ranges: list[tuple[int, int]] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise ValueError("negative seek position")
        self.position = base + offset
        return self.position

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        count = min(len(view), self.size - self.position)
        if count <= 0:
            return 0
        first, last = self.position, self.position + count - 1
        data = self._fetch(self.url, first, last)
        view[:count] = data
        self.ranges.append((first, last))
        self.position += count
        return count


def footer_range(metadata: Any, size: int) -> tuple[int, int]:
    """Bytes of the serialized footer, its length field and the trailing magic."""
    return size - 8 - int(metadata.serialized_size), size - 1


def column_ranges(metadata: Any, columns: Iterable[str]) -> list[tuple[int, int]]:
    """Inclusive byte ranges of every chunk of the named top-level columns."""
    wanted = set(columns)
    ranges = []
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for index in range(row_group.num_columns):
            chunk = row_group.column(index)
            if chunk.path_in_schema.split(".", 1)[0] not in wanted:
                continue
            start = chunk.data_page_offset
            if chunk.has_dictionary_page and 0 < chunk.dictionary_page_offset < start:
                start = chunk.dictionary_page_offset
            ranges.append((int(start), int(start) + int(chunk.total_compressed_size) - 1))
    return sorted(ranges)


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def overlapping(
    fetched: Iterable[tuple[int, int]], forbidden: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Fetched ranges that share at least one byte with a forbidden range."""
    return [
        (first, last)
        for first, last in fetched
        if any(first <= high and low <= last for low, high in forbidden)
    ]


def bytes_outside(fetched: Iterable[tuple[int, int]], allowed: Sequence[tuple[int, int]]) -> int:
    """Number of fetched bytes that fall outside every allowed range."""
    covered = merge_ranges(allowed)
    outside = 0
    for first, last in merge_ranges(fetched):
        inside = sum(max(0, min(last, high) - max(first, low) + 1) for low, high in covered)
        outside += last - first + 1 - inside
    return outside


def scan_file(
    remote: RemoteFile,
    url: str,
    fetch: Fetch,
    sample_keys: Mapping[Key, int],
    sample_repositories: frozenset[str],
) -> FileScan:
    """Read the key and license columns of one file and join them to the sample keys."""
    parquet = importlib.import_module("pyarrow.parquet")
    compute = importlib.import_module("pyarrow.compute")
    pyarrow = importlib.import_module("pyarrow")
    source = RangeFile(url, remote.size, fetch)
    result = FileScan(remote.path)
    reader = parquet.ParquetFile(source, pre_buffer=True)
    metadata = reader.metadata
    names = set(reader.schema_arrow.names)
    if not set(COLUMNS) <= names:
        raise ValueError(f"{remote.path} lacks the expected key/license columns")
    footer = footer_range(metadata, remote.size)
    selected_ranges = [footer, *column_ranges(metadata, COLUMNS)]
    content_ranges = column_ranges(metadata, [CONTENT_COLUMN])
    if CONTENT_COLUMN in names and not content_ranges:
        raise ValueError(f"{remote.path}: content column chunks could not be located")
    result.rows, result.row_groups = int(metadata.num_rows), int(metadata.num_row_groups)
    repositories = pyarrow.array(sorted(sample_repositories), type=pyarrow.string())
    offset = 0
    for group in range(metadata.num_row_groups):
        table = reader.read_row_group(group, columns=list(COLUMNS), use_threads=False)
        licenses = table.column(LICENSE_COLUMN)
        flat = compute.list_flatten(licenses)
        if len(flat):
            for entry in compute.value_counts(flat).to_pylist():
                if entry["values"] is not None:
                    result.license_ids[entry["values"]] += int(entry["counts"])
        result.rows_without_license += int(
            compute.sum(compute.equal(compute.list_value_length(licenses), 0)).as_py() or 0
        ) + int(licenses.null_count)
        mask = compute.is_in(table.column(NAME_COLUMN), value_set=repositories)
        selected = table.filter(mask)
        rows = zip(
            selected.column(NAME_COLUMN).to_pylist(),
            selected.column(PATH_COLUMN).to_pylist(),
            selected.column(LICENSE_COLUMN).to_pylist(),
            compute.indices_nonzero(mask).to_pylist(),
            strict=True,
        )
        for repository, path, listed, row in rows:
            ids = repository_license_ids(listed)
            result.repository_licenses.setdefault(repository, Counter())[ids] += 1
            key = (repository, path)
            if key in sample_keys:
                result.matches.setdefault(key, []).append((ids, offset + int(row)))
        offset += table.num_rows
    if offset != result.rows:
        raise ValueError(f"{remote.path} row count differs from its footer")
    leaked = overlapping(source.ranges, content_ranges)
    if leaked:
        raise RuntimeError(f"{remote.path}: fetched bytes overlap the content column: {leaked[:3]}")
    footer_bytes = fetch(url, footer[0], footer[1])
    source.ranges.append(footer)
    result.footer_sha256 = hashlib.sha256(footer_bytes).hexdigest()
    result.bytes_fetched = sum(last - first + 1 for first, last in source.ranges)
    result.bytes_outside_selected_columns = bytes_outside(source.ranges, selected_ranges)
    result.requests = len(source.ranges)
    return result


def list_python_files(client: HuggingFaceClient, revision: str) -> tuple[str, list[RemoteFile]]:
    info = client.json(f"https://{API_HOST}/api/datasets/{DATASET}/revision/{revision}?blobs=true")
    resolved = str(info["sha"])
    if resolved != revision:
        raise ValueError("pin the full 40-character revision; the API resolved a different one")
    files = sorted(
        (
            RemoteFile(item["rfilename"], int(item["size"]), str((item.get("lfs") or {})["sha256"]))
            for item in info["siblings"]
            if item["rfilename"].startswith(PYTHON_PREFIX)
            and item["rfilename"].endswith(".parquet")
        ),
        key=lambda remote: remote.path,
    )
    if not files:
        raise ValueError("the pinned revision has no data/python Parquet files")
    return resolved, files


def sample_keys_from_receipts(
    receipts_path: Path,
) -> tuple[dict[Key, int], list[dict[str, Any]], dict[str, Any]]:
    """The exact ordered Python prefix `tools.real_samples` reviews, after hash checks."""
    receipts = json.loads(receipts_path.read_text())
    if receipts.get("status") != "completed":
        raise ValueError("completed verified receipts are required")
    entry = next(item for item in receipts["files"] if item["domain"] == "python")
    path = Path(entry["path"])
    if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
        raise ValueError("raw Python sample size/hash differs before parsing")
    keys: dict[Key, int] = {}
    rows: list[dict[str, Any]] = []
    text_bytes = 0
    for index, row in enumerate(sample_rows(parquet_rows(path), "python", PYTHON_SAMPLE_TARGET)):
        key = (str(row[NAME_COLUMN]), str(row[PATH_COLUMN]))
        if key in keys:
            raise ValueError("sample (repository, path) keys must be unique")
        keys[key] = index
        size = len(row["content"].encode("utf-8"))
        text_bytes += size
        rows.append(
            {
                "index": index,
                "id": str(row.get("id", index)),
                "repository": key[0],
                "path": key[1],
                "text_bytes": size,
            }
        )
    identity = {
        "receipts": str(receipts_path),
        "receipts_sha256": file_sha256(receipts_path),
        "source": entry["source"],
        "parquet": entry["path"],
        "parquet_sha256": entry["sha256"],
        "rows": len(rows),
        "text_bytes": text_bytes,
        "target_text_bytes": PYTHON_SAMPLE_TARGET,
    }
    return keys, rows, identity


def join(
    scans: Sequence[FileScan], sample: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Exact-key mapping rows, diagnostics for everything else, and join statistics."""
    matches: dict[Key, list[tuple[tuple[str, ...], str, int]]] = {}
    repositories: dict[str, Counter[tuple[str, ...]]] = {}
    for scan in scans:
        for key, hits in scan.matches.items():
            matches.setdefault(key, []).extend((ids, scan.path, row) for ids, row in hits)
        for repository, counts in scan.repository_licenses.items():
            repositories.setdefault(repository, Counter()).update(counts)
    mapping: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    status: Counter[str] = Counter()
    status_bytes: Counter[str] = Counter()
    matched_ids: Counter[str] = Counter()
    for row in sample:
        key = (row["repository"], row["path"])
        found = matches.get(key, [])
        lists = {ids for ids, _, _ in found}
        sources = [
            {"file": name, "row": index}
            for _, name, index in sorted(found, key=lambda x: (x[1], x[2]))
        ]
        if len(found) == 1 or (len(found) > 1 and len(lists) == 1):
            kind = "matched" if len(found) == 1 else "matched_consistent_duplicates"
            (ids,) = lists
            mapping.append({**row, "licenses": list(ids), "status": kind, "upstream": sources})
            for item in ids:
                matched_ids[item] += 1
        elif found:
            kind = "conflicting_duplicates"
            diagnostics.append(
                {
                    **row,
                    "status": kind,
                    "license_lists": sorted(list(ids) for ids in lists),
                    "upstream": sources,
                }
            )
        elif key[0] in repositories:
            kind = "path_absent_repository_present"
            seen = repositories[key[0]]
            diagnostics.append(
                {
                    **row,
                    "status": kind,
                    "repository_license_lists": [
                        {"licenses": list(ids), "files": n} for ids, n in seen.most_common()
                    ],
                }
            )
        else:
            kind = "repository_absent"
            diagnostics.append({**row, "status": kind})
        status[kind] += 1
        status_bytes[kind] += row["text_bytes"]
    stats = {
        "sample_files": len(sample),
        "by_status": {
            name: {"files": status[name], "text_bytes": status_bytes[name]}
            for name in (
                "matched",
                "matched_consistent_duplicates",
                "conflicting_duplicates",
                "path_absent_repository_present",
                "repository_absent",
            )
        },
        "admissible_files": status["matched"] + status["matched_consistent_duplicates"],
        "matched_license_id_files": dict(sorted(matched_ids.items(), key=lambda x: (-x[1], x[0]))),
        "matched_license_list_files": dict(
            Counter(" AND ".join(m["licenses"]) or "<empty>" for m in mapping).most_common()
        ),
    }
    return mapping, diagnostics, stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return file_sha256(path)


def load_mapping(report_path: Path) -> tuple[dict[Key, tuple[str, ...]], dict[str, Any]]:
    """Verified (repository, path) -> license IDs for `tools.real_samples`."""
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 1 or report.get("status") != "completed":
        raise ValueError("a completed schema-1 license recovery report is required")
    if report.get("content_downloaded") is not False:
        raise ValueError("the recovery report must certify that no content was downloaded")
    mapping_path = report_path.parent / report["mapping"]["file"]
    if file_sha256(mapping_path) != report["mapping"]["sha256"]:
        raise ValueError("license mapping bytes differ from the recovery report")
    mapping: dict[Key, tuple[str, ...]] = {}
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["repository"], row["path"])
        if key in mapping:
            raise ValueError("duplicate key in license mapping")
        mapping[key] = repository_license_ids(row["licenses"])
    if len(mapping) != report["join"]["admissible_files"]:
        raise ValueError("license mapping row count differs from the recovery report")
    identity = {
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "mapping_sha256": report["mapping"]["sha256"],
        "dataset": report["dataset"],
        "revision": report["revision"],
        "columns_read": report["columns_read"],
        "sample_parquet_sha256": report["sample"]["parquet_sha256"],
        "join_rule": report["join_rule"],
    }
    return mapping, identity


def recover(
    *,
    receipts: Path,
    revision: str,
    output: Path,
    terms_accepted_on: str,
    client: HuggingFaceClient,
    workers: int = 4,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("license recovery output must be a new directory")
    started = time.perf_counter()
    keys, sample, sample_identity = sample_keys_from_receipts(receipts)
    resolved, files = list_python_files(client, revision)
    repositories = frozenset(key[0] for key in keys)
    output.mkdir(parents=True)

    def scan(remote: RemoteFile) -> FileScan:
        url = f"https://{API_HOST}/datasets/{DATASET}/resolve/{resolved}/{remote.path}"
        return scan_file(remote, url, client.fetch, keys, repositories)

    scans: list[FileScan] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(scan, files), start=1):
            scans.append(result)
            if done % 12 == 0 or done == len(files):
                progress(
                    f"scanned {done}/{len(files)} files, {client.bytes_fetched:,} bytes fetched"
                )
    mapping, diagnostics, stats = join(scans, sample)
    mapping_sha = write_jsonl(output / "mapping.jsonl", mapping)
    diagnostics_sha = write_jsonl(output / "join-diagnostics.jsonl", diagnostics)
    subset_ids: Counter[str] = Counter()
    for result in scans:
        subset_ids.update(result.license_ids)
    commit, dirty = git_state()
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "dataset": DATASET,
        "revision": resolved,
        "terms": {
            "dataset": DATASET,
            "accepted_on": terms_accepted_on,
            "basis": (
                "owner statement that gated access was granted; the same Terms of Use "
                "for The Stack as StarCoderData"
            ),
        },
        "columns_read": list(COLUMNS),
        "content_downloaded": False,
        "byte_range_guard": (
            "the run fails if any fetched byte overlaps a content column chunk; bytes outside "
            "the selected column chunks and footer are counted per file"
        ),
        "join_rule": (
            "exact (max_stars_repo_name, max_stars_repo_path); duplicate keys admitted only with "
            "identical license lists; repository-only matches are diagnostics, never admissions"
        ),
        "sample": sample_identity,
        "files": [
            {
                **asdict(remote),
                "rows": scan.rows,
                "row_groups": scan.row_groups,
                "footer_sha256": scan.footer_sha256,
                "bytes_fetched": scan.bytes_fetched,
                "bytes_outside_selected_columns": scan.bytes_outside_selected_columns,
                "range_requests": scan.requests,
                "rows_without_license": scan.rows_without_license,
            }
            for remote, scan in zip(files, sorted(scans, key=lambda s: s.path), strict=True)
        ],
        "totals": {
            "files": len(files),
            "rows_scanned": sum(scan.rows for scan in scans),
            "rows_without_license": sum(scan.rows_without_license for scan in scans),
            "upstream_file_bytes": sum(remote.size for remote in files),
            "bytes_fetched": client.bytes_fetched,
            "bytes_outside_selected_columns": sum(
                scan.bytes_outside_selected_columns for scan in scans
            ),
            "requests": client.requests,
        },
        "python_subset_license_id_files": dict(
            sorted(subset_ids.items(), key=lambda x: (-x[1], x[0]))
        ),
        "join": stats,
        "mapping": {"file": "mapping.jsonl", "sha256": mapping_sha, "rows": len(mapping)},
        "diagnostics": {
            "file": "join-diagnostics.jsonl",
            "sha256": diagnostics_sha,
            "rows": len(diagnostics),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "paid_runs": 0,
        "limitations": [
            "max_stars_repo_licenses is The Stack's detected repository license list, "
            "not a legal rights warranty",
            "StarCoderData applied PII redaction and near-deduplication after The Stack; "
            "the join uses provenance keys, not content hashes",
            "the D-05 allowlist is applied by tools.real_samples, with explicit file headers "
            "taking precedence",
        ],
    }
    atomic_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="full 40-character dataset revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terms-accepted-on", required=True, help="owner's acceptance date, YYYY-MM-DD"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    client = HuggingFaceClient(os.environ.get("HF_TOKEN", ""))
    report = recover(
        receipts=args.receipts,
        revision=args.revision,
        output=args.output,
        terms_accepted_on=args.terms_accepted_on,
        client=client,
        workers=args.workers,
    )
    summary = {key: report[key] for key in ("revision", "totals", "join", "mapping")}
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
