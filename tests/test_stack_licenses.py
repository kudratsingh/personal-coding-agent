"""Offline tests for the The Stack dedup license join; no network, no real dataset."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.shards import file_sha256
from tools import stack_licenses
from tools.real_samples import process_sample
from tools.stack_licenses import (
    FileScan,
    HuggingFaceClient,
    RemoteFile,
    bytes_outside,
    column_ranges,
    join,
    load_mapping,
    merge_ranges,
    overlapping,
    recover,
    scan_file,
    statistics_bytes_by_column,
    write_jsonl,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REVISION = "0123456789abcdef0123456789abcdef01234567"
URL = "https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/rev/data/python/x.parquet"


def upstream_parquet(rows: list[dict[str, Any]], *, row_group_size: int = 2) -> bytes:
    """An upstream-shaped file: key columns, licenses and a large content column."""
    table = pa.table(
        {
            "hexsha": [f"{i:040x}" for i in range(len(rows))],
            "max_stars_repo_path": [row["path"] for row in rows],
            "max_stars_repo_name": [row["repository"] for row in rows],
            "max_stars_repo_head_hexsha": ["f" * 40 for _ in rows],
            "max_stars_repo_licenses": pa.array(
                [row["licenses"] for row in rows], type=pa.list_(pa.string())
            ),
            "content": [row.get("content", "x = 1\n" * 2000) for row in rows],
            "alphanum_fraction": [0.5 for _ in rows],
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=row_group_size, use_dictionary=False)
    return sink.getvalue()


class Recorder:
    """Serves byte ranges from in-memory files and records every request."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[tuple[str, int, int]] = []

    def __call__(self, url: str, first: int, last: int) -> bytes:
        self.requests.append((url, first, last))
        return self.files[url][first : last + 1]


ROWS = [
    {"repository": "o/r", "path": "a.py", "licenses": ["MIT"]},
    {"repository": "o/r", "path": "b.py", "licenses": ["Apache-2.0", "MIT"]},
    {"repository": "o/r", "path": "z.py", "licenses": ["MIT"]},  # Same repository, not sampled.
    {"repository": "other/x", "path": "a.py", "licenses": ["ISC"]},
    {"repository": "o/r", "path": "c.py", "licenses": []},
]


def test_scan_reads_keys_and_licenses_but_never_the_content_column() -> None:
    data = upstream_parquet(ROWS)
    fetch = Recorder({URL: data})
    keys = {("o/r", "a.py"): 0, ("o/r", "b.py"): 1, ("o/r", "c.py"): 2}
    result = scan_file(
        RemoteFile("x.parquet", len(data), "0" * 64), URL, fetch, keys, frozenset({"o/r"})
    )
    assert (result.rows, result.row_groups) == (5, 3)
    assert result.matches == {
        ("o/r", "a.py"): [(("MIT",), 0)],
        ("o/r", "b.py"): [(("Apache-2.0", "MIT"), 1)],
        ("o/r", "c.py"): [((), 4)],
    }
    assert result.license_ids == Counter({"MIT": 3, "Apache-2.0": 1, "ISC": 1})
    assert result.rows_without_license == 1
    assert result.repository_licenses["o/r"] == Counter(
        {("MIT",): 2, ("Apache-2.0", "MIT"): 1, (): 1}
    )
    metadata = pq.ParquetFile(io.BytesIO(data)).metadata
    content = column_ranges(metadata, ["content"])
    assert content and not overlapping([(a, b) for _, a, b in fetch.requests], content)
    content_bytes = sum(b - a + 1 for a, b in content)
    assert result.bytes_fetched < len(data) - content_bytes
    # The footer is read exactly; only Parquet's coalescing of gaps under 8 KiB between
    # selected chunks can add bytes, and in this tiny fixture those are never content.
    assert result.bytes_outside_selected_columns < 1024
    footer_start = len(data) - 8 - metadata.serialized_size
    assert result.footer_sha256 == hashlib.sha256(data[footer_start:]).hexdigest()


def test_content_guard_fails_closed_if_a_read_touches_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = upstream_parquet(ROWS)
    monkeypatch.setattr(
        stack_licenses, "COLUMNS", (*stack_licenses.COLUMNS, stack_licenses.CONTENT_COLUMN)
    )
    fetch = Recorder({URL: data})
    with pytest.raises(RuntimeError, match="refusing to fetch"):
        scan_file(
            RemoteFile("x.parquet", len(data), "0" * 64),
            URL,
            fetch,
            {("o/r", "a.py"): 0},
            frozenset({"o/r"}),
        )
    # The guard refuses before sending: no content byte ever reaches the fetcher.
    content = column_ranges(pq.ParquetFile(io.BytesIO(data)).metadata, ["content"])
    assert not overlapping([(a, b) for _, a, b in fetch.requests], content)


def test_range_arithmetic() -> None:
    assert merge_ranges([(10, 19), (0, 4), (5, 9), (30, 31)]) == [(0, 19), (30, 31)]
    assert overlapping([(0, 4), (5, 9), (20, 25)], [(9, 12)]) == [(5, 9)]
    assert bytes_outside([(0, 9), (5, 14)], [(0, 4), (10, 11)]) == 8


def sample_row(index: int, repository: str, path: str, size: int = 10) -> dict[str, Any]:
    return {
        "index": index,
        "id": str(index),
        "repository": repository,
        "path": path,
        "text_bytes": size,
    }


def test_join_classifies_every_sample_row_exactly_once() -> None:
    first = FileScan("f1")
    first.matches = {("o/r", "a.py"): [(("MIT",), 0)], ("o/r", "dup.py"): [(("MIT",), 1)]}
    first.matches[("o/r", "clash.py")] = [(("MIT",), 2)]
    first.repository_licenses = {"o/r": Counter({("MIT",): 3})}
    second = FileScan("f2")
    second.matches = {("o/r", "dup.py"): [(("MIT",), 7)], ("o/r", "clash.py"): [(("ISC",), 8)]}
    sample = [
        sample_row(0, "o/r", "a.py"),
        sample_row(1, "o/r", "dup.py"),
        sample_row(2, "o/r", "clash.py"),
        sample_row(3, "o/r", "missing.py", 7),
        sample_row(4, "gone/x", "a.py", 5),
    ]
    mapping, diagnostics, stats = join([first, second], sample)
    assert [row["status"] for row in mapping] == ["matched", "matched_consistent_duplicates"]
    assert mapping[1]["upstream"] == [{"file": "f1", "row": 1}, {"file": "f2", "row": 7}]
    assert [row["status"] for row in diagnostics] == [
        "conflicting_duplicates",
        "path_absent_repository_present",
        "repository_absent",
    ]
    assert diagnostics[0]["license_lists"] == [["ISC"], ["MIT"]]
    assert diagnostics[1]["repository_license_lists"] == [{"licenses": ["MIT"], "files": 3}]
    assert stats["admissible_files"] == 2
    assert stats["by_status"]["path_absent_repository_present"] == {"files": 1, "text_bytes": 7}
    assert stats["by_status"]["repository_absent"] == {"files": 1, "text_bytes": 5}
    assert sum(item["files"] for item in stats["by_status"].values()) == len(sample)


def write_recovery(tmp_path: Path, rows: list[dict[str, Any]], **overrides: Any) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    digest = write_jsonl(tmp_path / "mapping.jsonl", rows)
    report = {
        "schema_version": 1,
        "status": "completed",
        "content_column_chunks_downloaded": False,
        "dataset": "bigcode/the-stack-dedup",
        "revision": REVISION,
        "columns_read": list(stack_licenses.COLUMNS),
        "sample": {"parquet_sha256": "a" * 64},
        "join_rule": "exact",
        "join": {"admissible_files": len(rows)},
        "mapping": {"file": "mapping.jsonl", "sha256": digest, "rows": len(rows)},
    }
    report.update(overrides)
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path / "report.json"


def test_load_mapping_verifies_identity_and_rejects_tampering(tmp_path: Path) -> None:
    rows = [{"repository": "o/r", "path": "a.py", "licenses": ["MIT", "Apache-2.0"]}]
    mapping, identity = load_mapping(write_recovery(tmp_path / "ok", rows))
    assert mapping == {("o/r", "a.py"): ("Apache-2.0", "MIT")}
    assert identity["revision"] == REVISION and identity["sample_parquet_sha256"] == "a" * 64
    tampered = write_recovery(tmp_path / "tampered", rows)
    (tmp_path / "tampered/mapping.jsonl").write_text(
        json.dumps({**rows[0], "licenses": ["GPL-3.0-only"]}) + "\n"
    )
    with pytest.raises(ValueError, match="bytes differ"):
        load_mapping(tampered)
    with pytest.raises(ValueError, match="no content column chunk"):
        load_mapping(
            write_recovery(tmp_path / "content", rows, content_column_chunks_downloaded=True)
        )
    with pytest.raises(ValueError, match="duplicate key"):
        load_mapping(write_recovery(tmp_path / "dup", rows + rows))
    with pytest.raises(ValueError, match="row count"):
        load_mapping(write_recovery(tmp_path / "count", rows, join={"admissible_files": 2}))


class FakeReply:
    def __init__(self, status: int, body: bytes) -> None:
        self.status, self._body = status, body

    def __enter__(self) -> "FakeReply":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class FakeOpener:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.seen: list[Any] = []

    def open(self, request: Any, timeout: float) -> FakeReply:
        self.seen.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        assert isinstance(reply, FakeReply)
        return reply


def test_client_keeps_the_token_off_redirects_and_checks_every_range() -> None:
    client = HuggingFaceClient("hf_fixture_token_only", sleep=lambda _: None)
    request = client.request(URL, {"Range": "bytes=0-3"})
    assert "Authorization" not in request.headers  # Copied to a redirect target.
    assert request.unredirected_hdrs["Authorization"] == "Bearer hf_fixture_token_only"
    with pytest.raises(ValueError, match="huggingface.co"):
        client.request("https://cdn.example.org/file")
    with pytest.raises(ValueError, match="huggingface.co"):
        client.request("http://huggingface.co/x")
    busy = HTTPError(URL, 503, "busy", hdrs=None, fp=io.BytesIO(b""))  # type: ignore[arg-type]
    client._opener = FakeOpener([busy, FakeReply(206, b"PAR1")])  # type: ignore[assignment]
    assert client.fetch(URL, 0, 3) == b"PAR1"
    assert (client.bytes_fetched, client.requests) == (4, 1)
    client._opener = FakeOpener([FakeReply(200, b"PAR1")])  # type: ignore[assignment]
    with pytest.raises(ValueError, match="not 206"):
        client.fetch(URL, 0, 3)
    client._opener = FakeOpener([FakeReply(206, b"PA")])  # type: ignore[assignment]
    with pytest.raises(ValueError, match="length"):
        client.fetch(URL, 0, 3)
    denied = HTTPError(URL, 403, "gated", hdrs=None, fp=io.BytesIO(b""))  # type: ignore[arg-type]
    client._opener = FakeOpener([denied])  # type: ignore[assignment]
    with pytest.raises(HTTPError):
        client.fetch(URL, 0, 3)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        HuggingFaceClient("")


class FakeHub(HuggingFaceClient):
    """Pinned API listing plus in-memory upstream files."""

    def __init__(self, files: dict[str, bytes], revision: str = REVISION) -> None:
        super().__init__("hf_fixture_token_only")
        self.files, self.revision = files, revision
        base = f"https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/{revision}"
        self.recorder = Recorder({f"{base}/{name}": data for name, data in files.items()})

    def json(self, url: str) -> Any:
        assert url.endswith(f"/revision/{REVISION}?blobs=true")
        siblings = [
            {
                "rfilename": name,
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for name, data in self.files.items()
        ]
        siblings.append(
            {
                "rfilename": "data/java/data-00000-of-00001.parquet",
                "size": 1,
                "lfs": {"sha256": "0" * 64},
            }
        )
        return {"sha": self.revision, "siblings": siblings}

    def fetch(self, url: str, first: int, last: int) -> bytes:
        data = self.recorder(url, first, last)
        self.bytes_fetched += len(data)
        self.requests += 1
        return data


def local_sample(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows = [
        {
            "max_stars_repo_path": "a.py",
            "max_stars_repo_name": "o/r",
            "content": "alpha_value = 12345678901234567890123\n",
        },
        {
            "max_stars_repo_path": "b.py",
            "max_stars_repo_name": "o/r",
            "content": "beta_value = 12345678901234567890123\n",
        },
        {
            "max_stars_repo_path": "q.py",
            "max_stars_repo_name": "o/r",
            "content": "gamma_value = 12345678901234567890123\n",
        },
        {
            "max_stars_repo_path": "a.py",
            "max_stars_repo_name": "nobody/here",
            "content": "delta_value = 12345678901234567890123\n",
        },
    ]
    table = pa.table(
        {
            "max_stars_repo_path": [row["max_stars_repo_path"] for row in rows],
            "max_stars_repo_name": [row["max_stars_repo_name"] for row in rows],
            "max_stars_count": [1 for _ in rows],
            "id": [str(i) for i in range(len(rows))],
            "content": [row["content"] for row in rows],
        }
    )
    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    receipts = tmp_path / "receipts.json"
    receipts.write_text(
        json.dumps(
            {
                "status": "completed",
                "files": [
                    {
                        "domain": "python",
                        "source": "bigcode/starcoderdata",
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                ],
            }
        )
    )
    return receipts, rows


def test_recover_end_to_end_then_the_review_admits_only_allowlisted_matches(tmp_path: Path) -> None:
    receipts, rows = local_sample(tmp_path)
    upstream = {
        "data/python/data-00000-of-00002.parquet": upstream_parquet(ROWS[:2]),
        "data/python/data-00001-of-00002.parquet": upstream_parquet(ROWS[2:]),
    }
    hub = FakeHub(upstream)
    target = sum(len(row["content"].encode()) for row in rows)
    report = recover(
        receipts=receipts,
        revision=REVISION,
        output=tmp_path / "recovery",
        terms_accepted_on="2026-10-24",
        client=hub,
        workers=2,
        progress=lambda _: None,
        target=target,
    )
    assert report["content_column_chunks_downloaded"] is False
    # 12 KB contents exceed the 4 KiB statistics limit, so these footers hold none.
    assert report["totals"]["footer_content_statistics_bytes"] == 0
    assert report["columns_read"] == [
        "max_stars_repo_name",
        "max_stars_repo_path",
        "max_stars_repo_licenses",
    ]
    assert [item["path"] for item in report["files"]] == sorted(upstream)
    assert report["totals"]["rows_scanned"] == 5
    assert report["join"]["by_status"]["matched"]["files"] == 2
    assert report["join"]["by_status"]["path_absent_repository_present"]["files"] == 1
    assert report["join"]["by_status"]["repository_absent"]["files"] == 1
    assert report["python_subset_license_id_files"] == {"MIT": 3, "Apache-2.0": 1, "ISC": 1}
    for name, data in upstream.items():
        url = f"https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/{REVISION}/{name}"
        content = column_ranges(pq.ParquetFile(io.BytesIO(data)).metadata, ["content"])
        fetched = [(a, b) for u, a, b in hub.recorder.requests if u == url]
        assert fetched and not overlapping(fetched, content)
    mapping, identity = load_mapping(tmp_path / "recovery" / "report.json")
    assert mapping == {("o/r", "a.py"): ("MIT",), ("o/r", "b.py"): ("Apache-2.0", "MIT")}
    review = process_sample(
        [dict(row, id=str(i)) for i, row in enumerate(rows)],
        domain="python",
        source="bigcode/starcoderdata",
        target=target,
        contamination=ContaminationIndex([]),
        output=tmp_path / "review",
        repository_licenses=mapping,
    )
    assert review["kept_documents"] == 2
    assert review["file_license_counts"] == {"MIT": 1, "Apache-2.0 AND MIT": 1}
    assert review["primary_decisions"]["license_missing_repository_license"]["files"] == 2
    with pytest.raises(FileExistsError):
        recover(
            receipts=receipts,
            revision=REVISION,
            output=tmp_path / "recovery",
            terms_accepted_on="2026-10-24",
            client=hub,
            target=target,
        )
    with pytest.raises(ValueError, match="different one"):
        recover(
            receipts=receipts,
            revision=REVISION,
            output=tmp_path / "moved",
            terms_accepted_on="2026-10-24",
            client=FakeHub(upstream, revision="f" * 40),
            target=target,
        )


def test_footer_statistics_bytes_are_measured_from_the_raw_thrift_footer() -> None:
    short = [
        {
            "repository": "o/r",
            "path": f"{i}.py",
            "licenses": ["MIT"],
            "content": f"value_{i} = {i}\n",
        }
        for i in range(4)
    ]
    for statistics, expect_content in ((True, True), (False, False)):
        table = pa.table(
            {"name": [r["repository"] for r in short], "content": [r["content"] for r in short]}
        )
        sink = io.BytesIO()
        pq.write_table(table, sink, row_group_size=2, write_statistics=statistics)
        data = sink.getvalue()
        length = int.from_bytes(data[-8:-4], "little")
        measured = statistics_bytes_by_column(data[-8 - length : -8])
        visible = 0
        metadata = pq.ParquetFile(io.BytesIO(data)).metadata
        for group in range(metadata.num_row_groups):
            column = metadata.row_group(group).column(1)
            if column.is_stats_set and column.statistics.has_min_max:
                visible += len(column.statistics.min) + len(column.statistics.max)
        if expect_content:
            assert measured["content"] >= visible > 0  # Legacy and current fields may both be set.
        else:
            assert measured.get("content", 0) == 0
    with pytest.raises(ValueError, match="truncated"):
        statistics_bytes_by_column(data[-8 - length : -20])


def test_legacy_recovery_reports_still_load(tmp_path: Path) -> None:
    rows = [{"repository": "o/r", "path": "a.py", "licenses": ["MIT"]}]
    report = write_recovery(tmp_path / "legacy", rows)
    raw = json.loads(report.read_text())
    raw["content_downloaded"] = raw.pop("content_column_chunks_downloaded")
    report.write_text(json.dumps(raw))
    assert load_mapping(report)[0] == {("o/r", "a.py"): ("MIT",)}
    raw.pop("content_downloaded")
    report.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no content column chunk"):
        load_mapping(report)


def test_recover_joins_every_row_of_whole_containers(tmp_path: Path) -> None:
    def container(name: str, rows: list[tuple[str, str]]) -> dict[str, Any]:
        table = pa.table(
            {
                "max_stars_repo_path": [path for _, path in rows],
                "max_stars_repo_name": [repository for repository, _ in rows],
                "max_stars_count": [1 for _ in rows],
                "id": [f"{name}-{i}" for i in range(len(rows))],
                "content": [f"value_{i} = {i}\n" for i in range(len(rows))],
            }
        )
        path = tmp_path / f"{name}.parquet"
        pq.write_table(table, path, row_group_size=1)
        return {
            "filename": f"python/{name}.parquet",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    # The same key in both containers is joined once and counted as repeated.
    containers = [
        container("train-0", [("o/r", "a.py"), ("o/r", "q.py")]),
        container("train-1", [("o/r", "b.py"), ("o/r", "a.py")]),
    ]
    hub = FakeHub({"data/python/data-00000-of-00001.parquet": upstream_parquet(ROWS)})
    report = recover(
        containers=containers,
        revision=REVISION,
        output=tmp_path / "recovery",
        terms_accepted_on="2026-10-24",
        client=hub,
        workers=1,
        progress=lambda _: None,
    )
    sample = report["sample"]
    assert sample["mode"] == "containers" and sample["parquet_sha256"] is None
    assert [item["rows"] for item in sample["containers"]] == [2, 2]
    assert sample["unique_keys"] == 3 and sample["repeated_keys"] == 1
    assert report["join"]["by_status"]["matched"]["files"] == 2
    assert report["join"]["by_status"]["path_absent_repository_present"]["files"] == 1
    mapping, identity = load_mapping(tmp_path / "recovery" / "report.json")
    assert mapping == {("o/r", "a.py"): ("MIT",), ("o/r", "b.py"): ("Apache-2.0", "MIT")}
    assert [item["sha256"] for item in identity["containers"]] == [
        item["sha256"] for item in containers
    ]
    with pytest.raises(ValueError, match="exactly one"):
        recover(
            revision=REVISION,
            output=tmp_path / "neither",
            terms_accepted_on="2026-10-24",
            client=hub,
        )
    containers[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="size/hash"):
        recover(
            containers=containers,
            revision=REVISION,
            output=tmp_path / "tampered",
            terms_accepted_on="2026-10-24",
            client=hub,
        )
