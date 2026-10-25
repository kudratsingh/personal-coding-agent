"""Offline tests for the rehearsal data build; authored fixtures only, no network."""

import hashlib
import io
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from scratchcoder.data.filters import content_hash, document_split
from scratchcoder.data.shards import TokenShard, file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import rehearsal_build
from tools.rehearsal_build import Hub, build, download, licenses, load_spec, verify
from tools.stack_licenses import HuggingFaceClient

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

WEB_REVISION = "1" * 40
PYTHON_REVISION = "2" * 40
STACK_REVISION = "3" * 40
HELD_OUT = "Implement a function that returns the sum of all distinct positive elements in a list"


def quiet(_: str) -> None:
    return None


def text_in(split: str, *, web: bool, stem: str) -> str:
    """An authored document whose content hash lands in ``split``."""
    for number in range(100_000):
        text = f"{stem} variant {number}.\n" + "\n".join(f"line_{i} = {i}" for i in range(4))
        if not web:
            text = f"def {stem}_{number}(value):\n    return value + {number}\n" + text
        if document_split(content_hash(text), web=web) == split:
            return text
    raise AssertionError("no fixture text found")


def parquet_bytes(columns: dict[str, list[Any]]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.table(columns), sink, row_group_size=2)
    return sink.getvalue()


def web_file(texts: list[str], first: int) -> bytes:
    ids = [f"<urn:uuid:{first + i}>" for i in range(len(texts))]
    urls = [f"https://example.org/{first + i}" for i in range(len(texts))]
    return parquet_bytes({"text": texts, "id": ids, "url": urls})


def python_file(rows: list[tuple[str, str, str]], first: int) -> bytes:
    return parquet_bytes(
        {
            "max_stars_repo_path": [path for _, path, _ in rows],
            "max_stars_repo_name": [repository for repository, _, _ in rows],
            "max_stars_count": [1 for _ in rows],
            "id": [str(first + i) for i in range(len(rows))],
            "content": [text for _, _, text in rows],
        }
    )


def stack_file(rows: list[tuple[str, str, list[str]]]) -> bytes:
    """An upstream-shaped The Stack dedup file with a large content column."""
    return parquet_bytes(
        {
            "hexsha": [f"{i:040x}" for i in range(len(rows))],
            "max_stars_repo_path": [path for _, path, _ in rows],
            "max_stars_repo_name": [repository for repository, _, _ in rows],
            "max_stars_repo_head_hexsha": ["f" * 40 for _ in rows],
            "max_stars_repo_licenses": pa.array(
                [ids for _, _, ids in rows], type=pa.list_(pa.string())
            ),
            "content": ["x = 1\n" * 2000 for _ in rows],
            "alphanum_fraction": [0.5 for _ in rows],
        }
    )


class Reply(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class FakeHub(Hub):
    """Pinned listings and in-memory files; records each request's headers."""

    def __init__(self, token: str | None, files: Mapping[tuple[str, str, str], bytes]) -> None:
        super().__init__(token, sleep=lambda _: None)
        self.files = files
        self.requests: list[Request] = []
        self.truncate_first = False

    def json(self, url: str) -> Any:
        self.requests.append(self.request(url))
        dataset, revision = url.split("/api/datasets/")[1].split("?")[0].split("/revision/")
        siblings = [
            {
                "rfilename": name,
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for (repo, rev, name), data in self.files.items()
            if repo == dataset and rev == revision
        ]
        return {"sha": revision, "siblings": siblings}

    def open(self, url: str, headers: Mapping[str, str] | None = None) -> Any:
        request = self.request(url, headers)
        self.requests.append(request)
        dataset, rest = url.split("/datasets/")[1].split("/resolve/")
        revision, name = rest.split("/", 1)
        data = self.files[(dataset, revision, name)]
        first = int((headers or {}).get("Range", "bytes=0-")[6:-1])
        if self.truncate_first and not first:
            self.truncate_first = False
            return Reply(data[: len(data) // 2], 200)
        return Reply(data[first:], 206 if first else 200)


class FakeStack(HuggingFaceClient):
    """The Stack dedup listing plus in-memory upstream files served by range."""

    def __init__(self, files: dict[str, bytes]) -> None:
        super().__init__("hf_fixture_token_only")
        self.files = files
        self.base = (
            f"https://huggingface.co/datasets/bigcode/the-stack-dedup/resolve/{STACK_REVISION}"
        )

    def json(self, url: str) -> Any:
        siblings = [
            {
                "rfilename": name,
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for name, data in self.files.items()
        ]
        return {"sha": STACK_REVISION, "siblings": siblings}

    def fetch(self, url: str, first: int, last: int) -> bytes:
        name = url.removeprefix(self.base + "/")
        data = self.files[name][first : last + 1]
        self.bytes_fetched += len(data)
        self.requests += 1
        return data


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.s1 = [text_in("stage1", web=True, stem=f"Stage one article {i}") for i in range(3)]
        self.s2 = [text_in("stage2", web=True, stem=f"Stage two article {i}") for i in range(2)]
        self.web_validation = text_in("validation", web=True, stem="Held validation article")
        contaminated = text_in("stage1", web=True, stem=f"Prose that quotes {HELD_OUT} at length")
        web0 = [self.s1[0], contaminated, self.s2[0], self.web_validation, self.s1[1]]
        web1 = [self.s1[0], self.s1[2], self.s2[1]]  # The first repeats an earlier document.
        self.mit = text_in("train", web=False, stem="metadata_mit")
        self.both = text_in("train", web=False, stem="metadata_both")
        self.python_validation = "# SPDX-License-Identifier: MIT\n" + text_in(
            "validation", web=False, stem="inline_validation"
        )
        while document_split(content_hash(self.python_validation)) != "validation":
            self.python_validation += "\n"
        python0 = [
            ("o/r", "mit.py", self.mit),
            ("o/r", "gpl.py", "# Licensed under the GNU General Public License v3\n" + self.mit),
            ("o/x", "ecl.py", text_in("train", web=False, stem="metadata_ecl")),
            ("nobody/none", "lost.py", text_in("train", web=False, stem="no_metadata")),
            ("o/r", "val.py", self.python_validation),
        ]
        generated = text_in("train", web=False, stem="generated") + "\n# DO NOT EDIT\n"
        python1 = [("o/r", "gen.py", generated), ("o/r", "both.py", self.both)]
        self.files = {
            ("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/000.parquet"): web_file(web0, 0),
            ("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/001.parquet"): web_file(web1, 5),
            ("bigcode/starcoderdata", PYTHON_REVISION, "python/train-0.parquet"): python_file(
                python0, 0
            ),
            ("bigcode/starcoderdata", PYTHON_REVISION, "python/train-1.parquet"): python_file(
                python1, 5
            ),
        }
        self.stack = {
            "data/python/data-0.parquet": stack_file(
                [
                    ("o/r", "mit.py", ["MIT"]),
                    ("o/r", "gpl.py", ["MIT"]),
                    ("o/x", "ecl.py", ["ECL-2.0"]),
                    ("o/r", "val.py", ["MIT"]),
                ]
            ),
            "data/python/data-1.parquet": stack_file(
                [("o/r", "gen.py", ["MIT"]), ("o/r", "both.py", ["MIT", "Apache-2.0"])]
            ),
        }
        corpus = [*web0, *web1, *(text for _, _, text in python0 + python1)]
        self.tokenizer = ByteBPE.train(corpus, vocab_size=320)
        self.tokenizer_path = root / "tokenizer.json"
        self.tokenizer.save(self.tokenizer_path)
        self.build = root / "build"

    def length(self, text: str) -> int:
        return 1 + len(self.tokenizer.encode(text))

    def spec(self, quotas: Mapping[str, int] | None = None, **overrides: Any) -> Path:
        quotas = dict(
            quotas
            or {
                # The third stage-1 document is cut to its BOS and two tokens.
                "stage1-web": self.length(self.s1[0]) + self.length(self.s1[1]) + 3,
                # The second stage-2 document is left unused.
                "stage2-web": self.length(self.s2[0]),
                "stage2-python": self.length(self.mit) + 5,
            }
        )
        value: dict[str, Any] = {
            "schema_version": 1,
            "name": "fixture",
            "tokenizer": {
                "path": str(self.tokenizer_path),
                "identity": self.tokenizer.sha256,
                "file_sha256": file_sha256(self.tokenizer_path),
                "vocab_size": self.tokenizer.vocab_size,
                "bos_id": 256,
            },
            "sources": [
                {
                    "domain": "web",
                    "dataset": "HuggingFaceFW/fineweb-edu",
                    "revision": WEB_REVISION,
                    "gated": False,
                    "files": ["sample/000.parquet", "sample/001.parquet"],
                    "license": "ODC-By-1.0",
                    "terms": "fixture terms",
                    "terms_accepted_on": "2026-10-19",
                    "attribution": "fixture",
                    "obligations": "fixture obligations",
                },
                {
                    "domain": "python",
                    "dataset": "bigcode/starcoderdata",
                    "revision": PYTHON_REVISION,
                    "gated": True,
                    "files": ["python/train-0.parquet", "python/train-1.parquet"],
                    "license": "original file licenses, per file",
                    "terms": "fixture terms",
                    "terms_accepted_on": "2026-10-19",
                    "attribution": "fixture",
                    "obligations": "fixture obligations",
                },
            ],
            "license_metadata": {
                "dataset": "bigcode/the-stack-dedup",
                "revision": STACK_REVISION,
                "terms": "fixture terms",
                "terms_accepted_on": "2026-10-24",
                "attribution": "fixture",
                "obligations": "fixture obligations",
            },
            "contamination": {"humaneval": "x", "evalplus_directory": "x", "arc_directory": "x"},
            "streams": [
                {"name": "stage1-web", "domain": "web", "split": "stage1", "tokens": None},
                {"name": "stage2-web", "domain": "web", "split": "stage2", "tokens": None},
                {"name": "stage2-python", "domain": "python", "split": "train", "tokens": None},
                {"name": "validation-web", "domain": "web", "split": "validation", "tokens": None},
                {
                    "name": "validation-python",
                    "domain": "python",
                    "split": "validation",
                    "tokens": None,
                },
            ],
            "unit_row_groups": 1,
            "shard_max_tokens": 40,
            "min_free_bytes": 1,
            "max_download_bytes": sum(len(data) for data in self.files.values()),
        }
        for stream in value["streams"]:
            stream["tokens"] = quotas.get(stream["name"])
        value.update(overrides)
        path = self.root / "spec.json"
        path.write_text(json.dumps(value))
        return path

    def hub(self, token: str | None) -> FakeHub:
        return FakeHub(token, self.files)


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setattr(
        rehearsal_build,
        "contamination_inputs",
        lambda *_: ([HELD_OUT], {"fixture": "authored held-out phrase"}),
    )
    return Fixture(tmp_path)


def prepare(fixture: Fixture, spec_path: Path) -> rehearsal_build.Spec:
    spec = load_spec(spec_path)
    download(spec, fixture.build, token="hf_fixture", hub_factory=fixture.hub, progress=quiet)
    licenses(spec, fixture.build, client=FakeStack(fixture.stack), workers=1, progress=quiet)
    return spec


def test_spec_rejects_unknown_keys_missing_streams_and_unlisted_web_licenses(
    fixture: Fixture,
) -> None:
    load_spec(fixture.spec())
    with pytest.raises(ValueError, match="exactly the keys"):
        load_spec(fixture.spec(extra=1))
    with pytest.raises(ValueError, match="every"):
        load_spec(fixture.spec(streams=[]))
    spec = json.loads(fixture.spec().read_text())
    spec["sources"][0]["license"] = "CC-BY-NC-4.0"
    fixture.spec().write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="allowlist"):
        load_spec(fixture.root / "spec.json")
    spec["sources"][0]["license"] = "ODC-By-1.0"
    spec["streams"][3]["tokens"] = 5
    (fixture.root / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="validation streams"):
        load_spec(fixture.root / "spec.json")
    spec["streams"][3]["tokens"] = None
    for section, key, value, message in [
        (spec["sources"][1], "terms_accepted_on", None, "YYYY-MM-DD"),
        (spec["sources"][1], "terms_accepted_on", "2999-01-01", "future"),
        (spec["sources"][1], "attribution", " ", "nonempty"),
        (spec["license_metadata"], "terms_accepted_on", "24/10/2026", "YYYY-MM-DD"),
    ]:
        original = section[key]
        section[key] = value
        (fixture.root / "spec.json").write_text(json.dumps(spec))
        with pytest.raises(ValueError, match=message):
            load_spec(fixture.root / "spec.json")
        section[key] = original


def test_download_verifies_hashes_keeps_the_token_off_public_sources_and_reuses(
    fixture: Fixture,
) -> None:
    hubs: list[FakeHub] = []

    def factory(token: str | None) -> FakeHub:
        hubs.append(fixture.hub(token))
        return hubs[-1]

    # An earlier verified receipt for one file is hard-linked, not fetched again.
    earlier = fixture.root / "earlier.parquet"
    key = ("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/000.parquet")
    earlier.write_bytes(fixture.files[key])
    receipt = fixture.root / "earlier.json"
    receipt.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source": key[0],
                        "revision": key[1],
                        "filename": key[2],
                        "path": str(earlier),
                        "verified": True,
                    }
                ]
            }
        )
    )
    spec = load_spec(fixture.spec())
    report = download(
        spec,
        fixture.build,
        token="hf_fixture",
        reuse=[receipt],
        hub_factory=factory,
        progress=quiet,
    )
    assert report["status"] == "completed"
    assert [item["method"] for item in report["files"]] == [
        "hard-linked",
        "downloaded",
        "downloaded",
        "downloaded",
    ]
    for item in report["files"]:
        assert file_sha256(Path(item["path"])) == item["sha256"]
    web, python = hubs
    assert all(not request.unredirected_hdrs for request in web.requests)
    assert all(
        request.unredirected_hdrs == {"Authorization": "Bearer hf_fixture"}
        and "Authorization" not in request.headers
        for request in python.requests
    )
    with pytest.raises(FileExistsError):
        download(spec, fixture.build, token="hf_fixture", hub_factory=factory, progress=quiet)


def test_download_stops_on_size_hash_or_budget_violations(fixture: Fixture) -> None:
    spec = load_spec(fixture.spec(max_download_bytes=10))
    with pytest.raises(RuntimeError, match="approved maximum"):
        download(spec, fixture.build, token="t", hub_factory=fixture.hub, progress=quiet)
    assert not fixture.build.exists()
    spec = load_spec(fixture.spec())
    hub = fixture.hub(None)
    target = fixture.root / "out.parquet"
    url = (
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/"
        f"{WEB_REVISION}/sample/000.parquet"
    )
    data = fixture.files[("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/000.parquet")]
    with pytest.raises(ValueError, match="SHA-256"):
        hub.download(url, target, len(data), "0" * 64)
    assert not target.exists() and not target.with_name("out.parquet.part").exists()
    # A short first reply resumes with a range request and still verifies.
    hub.truncate_first = True
    hub.download(url, target, len(data), hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    assert hub.requests[-1].headers["Range"] == f"bytes={len(data) // 2}-"
    del spec


def test_build_fills_exact_quotas_and_records_every_decision(fixture: Fixture) -> None:
    spec = prepare(fixture, fixture.spec())
    report = build(spec, fixture.build, workers=2, progress=quiet)
    streams = report["streams"]
    assert {name: value["tokens"] for name, value in streams.items() if value["quota_tokens"]} == {
        "stage1-web": spec.stream_for("web", "stage1").tokens,
        "stage2-web": spec.stream_for("web", "stage2").tokens,
        "stage2-python": spec.stream_for("python", "train").tokens,
    }
    assert streams["stage1-web"]["documents"] == 3
    assert streams["stage1-web"]["cut"]["kept_tokens"] == 3
    assert streams["stage1-web"]["cut"]["sha256"] == content_hash(fixture.s1[2])
    assert streams["stage2-web"]["cut"] is None
    assert streams["stage2-web"]["unused_documents"] == 1
    assert streams["stage2-python"]["cut"]["kept_tokens"] == 5
    assert streams["validation-web"]["documents"] == 1
    assert streams["validation-python"]["documents"] == 1
    web, python = report["domains"]["web"], report["domains"]["python"]
    assert web["sample_documents"] == 8
    assert web["primary_decisions"]["duplicate"]["files"] == 1
    assert web["primary_decisions"]["contamination"]["files"] == 1
    assert web["kept_documents"] == 6
    decisions = {k: v["files"] for k, v in python["primary_decisions"].items() if v["files"]}
    assert decisions == {
        "kept": 3,
        "license_copyleft": 1,
        "license_repository_disallowed": 1,
        "license_missing_repository_license": 1,
        "generated": 1,
    }
    assert python["rejected_repository_license_id_counts"] == {"ECL-2.0": 1}
    assert python["kept_license_id_counts"] == {"MIT": 3, "Apache-2.0": 1}
    assert python["repository_metadata_lookups"] == {"found": 6, "absent": 1}
    assert python["kept_license_basis"]["inline_or_source_terms"]["files"] == 1
    # Shards are standard token shards whose documents decode to the source text.
    output = fixture.build / "output"
    shard = TokenShard(
        output / streams["stage2-python"]["shards"][0]["file"],
        tokenizer_hash=fixture.tokenizer.sha256,
        vocab_size=fixture.tokenizer.vocab_size,
        bos_id=256,
    )
    assert fixture.tokenizer.decode(shard[0][1:]) == fixture.mit
    assert len(streams["stage1-web"]["shards"]) >= 2  # The 40-token shard cap.
    rows = [json.loads(line) for line in (output / "documents.jsonl").read_text().splitlines()]
    assert len(rows) == sum(value["documents"] for value in streams.values())
    both = next(row for row in rows if row["path"] == "both.py")
    assert both["cut"] and both["repository_licenses"] == ["Apache-2.0", "MIT"]
    assert both["license"] == "Apache-2.0 AND MIT"
    credits = streams["stage2-python"]["credits"]
    assert credits["license_id_documents"] == {"MIT": 2, "Apache-2.0": 1}
    assert credits["repositories_by_license_id"] == {"Apache-2.0": 1, "MIT": 1}
    assert streams["stage2-python"]["cut"]["discarded_tokens"] > 0
    excluded = [json.loads(x) for x in (output / "exclusions.jsonl").read_text().splitlines()]
    assert sorted(row["reason"] for row in excluded) == sorted(
        [
            "duplicate",
            "contamination",
            "license_copyleft",
            "license_repository_disallowed",
            "license_missing_repository_license",
            "generated",
        ]
    )
    assert report["exclusions"]["rows"] == 6
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "passed"
    assert not any(verification["document_hash_overlaps"].values())
    assert verification["checks"]["re_encoded_documents"] >= 5
    with pytest.raises(FileExistsError):
        build(spec, fixture.build, workers=1, progress=quiet)


def test_rebuild_reuses_verified_units_redoes_damaged_ones_and_verify_catches_tampering(
    fixture: Fixture,
) -> None:
    spec = prepare(fixture, fixture.spec())
    first = build(spec, fixture.build, workers=1, progress=quiet)
    units = fixture.build / "units"
    records = {path.name: json.loads(path.read_text()) for path in units.glob("*.json")}
    damaged = sorted(units.glob("python-*.tokens"))[0]
    damaged.write_bytes(damaged.read_bytes()[:-2] + b"\xff\xff")
    shutil.rmtree(fixture.build / "output")
    second = build(spec, fixture.build, workers=1, progress=quiet)
    assert second["build_identity"] == first["build_identity"]
    after = {path.name: json.loads(path.read_text()) for path in units.glob("*.json")}
    redone = {name for name in after if after[name]["process"] != records[name]["process"]}
    assert redone == {damaged.with_suffix(".json").name}
    shard = fixture.build / "output" / second["streams"]["stage1-web"]["shards"][0]["file"]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        verify(spec, fixture.build)
    assert not (fixture.build / "output" / "verification.json").exists()


def test_units_are_redone_when_their_inputs_change_or_files_go_missing(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = prepare(fixture, fixture.spec())
    first = build(spec, fixture.build, workers=1, progress=quiet)
    units = fixture.build / "units"
    before = {path.name: json.loads(path.read_text()) for path in units.glob("*.json")}
    # A different contamination set changes every unit's input fingerprint.
    monkeypatch.setattr(
        rehearsal_build, "contamination_inputs", lambda *_: ([HELD_OUT, "another phrase"], {})
    )
    shutil.rmtree(fixture.build / "output")
    second = build(spec, fixture.build, workers=1, progress=quiet)
    after = {path.name: json.loads(path.read_text()) for path in units.glob("*.json")}
    assert all(after[name]["inputs"] != before[name]["inputs"] for name in before)
    assert second["tokens_identity"] == first["tokens_identity"]
    assert second["build_identity"] != first["build_identity"]
    missing = sorted(units.glob("web-*.tokens"))[0]
    missing.unlink()
    shutil.rmtree(fixture.build / "output")
    third = build(spec, fixture.build, workers=1, progress=quiet)
    assert missing.exists() and third["tokens_identity"] == first["tokens_identity"]


def test_one_token_of_room_takes_the_crossing_documents_bos_alone(fixture: Fixture) -> None:
    quotas = {
        "stage1-web": fixture.length(fixture.s1[0]) + fixture.length(fixture.s1[1]) + 1,
        "stage2-web": fixture.length(fixture.s2[0]),
        "stage2-python": fixture.length(fixture.mit) + 5,
    }
    spec = prepare(fixture, fixture.spec(quotas))
    report = build(spec, fixture.build, workers=1, progress=quiet)
    stage1 = report["streams"]["stage1-web"]
    assert stage1["cut"]["kept_tokens"] == 1 and stage1["tokens"] == quotas["stage1-web"]
    # The BOS-only document is listed, but no license credit is claimed for it.
    assert stage1["documents"] == 3
    assert sum(stage1["credits"]["license_id_documents"].values()) == 2
    assert report["verification"]["status"] == "passed"


def test_a_failed_verification_publishes_nothing(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = prepare(fixture, fixture.spec())

    def fail(*_: Any, **__: Any) -> dict[str, Any]:
        raise ValueError("planted verification failure")

    monkeypatch.setattr(rehearsal_build, "verify", fail)
    with pytest.raises(ValueError, match="planted"):
        build(spec, fixture.build, workers=1, progress=quiet)
    assert not (fixture.build / "output").exists()
    assert (fixture.build / "output.tmp" / "report.json").exists()


def test_a_quota_the_sources_cannot_fill_stops_before_any_shard(fixture: Fixture) -> None:
    spec = prepare(
        fixture, fixture.spec({"stage1-web": 10_000, "stage2-web": 1, "stage2-python": 1})
    )
    with pytest.raises(RuntimeError, match="quotas not met"):
        build(spec, fixture.build, workers=1, progress=quiet)
    assert not (fixture.build / "output").exists()
    assert not (fixture.build / "output.tmp").exists()


def test_the_license_mapping_must_match_the_downloaded_containers(fixture: Fixture) -> None:
    spec = prepare(fixture, fixture.spec())
    receipts_path = fixture.build / "download-receipts.json"
    receipts = json.loads(receipts_path.read_text())
    receipts["files"][2]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different containers"):
        rehearsal_build.license_mapping(spec, fixture.build, receipts)
