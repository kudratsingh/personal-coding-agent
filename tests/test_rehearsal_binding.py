"""Offline tests for the rehearsal data binding: an authored admitted build and a fake R2."""

import copy
import hashlib
import io
import json
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request

import numpy as np
import pytest

from scratchcoder.data.filters import content_hash, document_split
from scratchcoder.data.onepass import combine_digests, multiset_digest, order_digest, pass_order
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import rehearsal_binding, rehearsal_build
from tools.r2_store import Credentials, Store
from tools.rehearsal_binding import (
    RUNS,
    binding_key,
    build_binding,
    fetch,
    load_binding,
    main,
    r2_credentials,
    run_files,
    staging_files,
    verify_binding,
)
from tools.stack_licenses import HuggingFaceClient

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

WEB_REVISION = "1" * 40
PYTHON_REVISION = "2" * 40
STACK_REVISION = "3" * 40
HELD_OUT = "Implement a function that returns the sum of all distinct positive elements in a list"
PREFIX = "rehearsal-s-fixture/"


def quiet(_: str) -> None:
    return None


# The build fixture below is a trimmed copy of tests/test_rehearsal_build.py's.


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


class HubReply(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class FakeHub(rehearsal_build.Hub):
    def __init__(self, token: str | None, files: Mapping[tuple[str, str, str], bytes]) -> None:
        super().__init__(token, sleep=lambda _: None)
        self.files = files

    def json(self, url: str) -> Any:
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
        dataset, rest = url.split("/datasets/")[1].split("/resolve/")
        revision, name = rest.split("/", 1)
        data = self.files[(dataset, revision, name)]
        first = int((headers or {}).get("Range", "bytes=0-")[6:-1])
        return HubReply(data[first:], 206 if first else 200)


class FakeStack(HuggingFaceClient):
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
        data = self.files[url.removeprefix(self.base + "/")][first : last + 1]
        self.bytes_fetched += len(data)
        self.requests += 1
        return data


class Fixture:
    """Three stage-1, two stage-2 and one validation web document; three Python files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.s1 = [text_in("stage1", web=True, stem=f"Stage one article {i}") for i in range(3)]
        self.s2 = [text_in("stage2", web=True, stem=f"Stage two article {i}") for i in range(2)]
        web0 = [self.s1[0], self.s2[0], text_in("validation", web=True, stem="Held"), self.s1[1]]
        web1 = [self.s1[2], self.s2[1]]
        self.mit = text_in("train", web=False, stem="metadata_mit")
        both = text_in("train", web=False, stem="metadata_both")
        validation = "# SPDX-License-Identifier: MIT\n" + text_in(
            "validation", web=False, stem="inline_validation"
        )
        while document_split(content_hash(validation)) != "validation":
            validation += "\n"
        python0 = [("o/r", "mit.py", self.mit), ("o/r", "val.py", validation)]
        python1 = [("o/r", "both.py", both)]
        self.files = {
            ("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/000.parquet"): web_file(web0, 0),
            ("HuggingFaceFW/fineweb-edu", WEB_REVISION, "sample/001.parquet"): web_file(web1, 4),
            ("bigcode/starcoderdata", PYTHON_REVISION, "python/train-0.parquet"): python_file(
                python0, 0
            ),
            ("bigcode/starcoderdata", PYTHON_REVISION, "python/train-1.parquet"): python_file(
                python1, 2
            ),
        }
        self.stack = {
            "data/python/data-0.parquet": stack_file(
                [("o/r", "mit.py", ["MIT"]), ("o/r", "val.py", ["MIT"])]
            ),
            "data/python/data-1.parquet": stack_file([("o/r", "both.py", ["MIT", "Apache-2.0"])]),
        }
        corpus = [*web0, *web1, *(text for _, _, text in python0 + python1)]
        self.tokenizer = ByteBPE.train(corpus, vocab_size=320)
        self.tokenizer_path = root / "tokenizer.json"
        self.tokenizer.save(self.tokenizer_path)
        self.build = root / "build"

    def length(self, text: str) -> int:
        return 1 + len(self.tokenizer.encode(text))

    def spec(self) -> Path:
        quotas = {
            # The third stage-1 document is cut; the second stage-2 one is left unused.
            "stage1-web": self.length(self.s1[0]) + self.length(self.s1[1]) + 3,
            "stage2-web": self.length(self.s2[0]),
            "stage2-python": self.length(self.mit) + 5,
        }
        source = {
            "license": "ODC-By-1.0",
            "terms": "fixture terms",
            "terms_accepted_on": "2026-10-19",
            "attribution": "fixture",
            "obligations": "fixture obligations",
        }
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
                    **source,
                    "domain": "web",
                    "dataset": "HuggingFaceFW/fineweb-edu",
                    "revision": WEB_REVISION,
                    "gated": False,
                    "files": ["sample/000.parquet", "sample/001.parquet"],
                },
                {
                    **source,
                    "domain": "python",
                    "dataset": "bigcode/starcoderdata",
                    "revision": PYTHON_REVISION,
                    "gated": True,
                    "files": ["python/train-0.parquet", "python/train-1.parquet"],
                    "license": "original file licenses, per file",
                },
            ],
            "license_metadata": {
                **{key: source[key] for key in ("terms", "attribution", "obligations")},
                "dataset": "bigcode/the-stack-dedup",
                "revision": STACK_REVISION,
                "terms_accepted_on": "2026-10-24",
            },
            "contamination": {"humaneval": "x", "evalplus_directory": "x", "arc_directory": "x"},
            # Spec order differs from report.json's sorted key order.
            "streams": [
                {"name": "stage1-web", "domain": "web", "split": "stage1"},
                {"name": "stage2-web", "domain": "web", "split": "stage2"},
                {"name": "stage2-python", "domain": "python", "split": "train"},
                {"name": "validation-web", "domain": "web", "split": "validation"},
                {"name": "validation-python", "domain": "python", "split": "validation"},
            ],
            "unit_row_groups": 1,
            "shard_max_tokens": 40,
            "min_free_bytes": 1,
            "max_download_bytes": sum(len(data) for data in self.files.values()),
        }
        for stream in value["streams"]:
            stream["tokens"] = quotas.get(stream["name"])
        path = self.root / "spec.json"
        path.write_text(json.dumps(value))
        return path


def write_holdout(root: Path) -> Path:
    path = root / "holdout.jsonl"
    prompt = "Write a function that counts the vowels in the string it is given as its argument"
    save_tasks(
        path,
        [
            Task(
                id="pool/held-1",
                prompt=prompt,
                public_tests="assert True",
                hidden_tests="assert True",
                source="fixture",
                license="MIT",
                written_by="model",
                split="dev",
                solution="x = 1",
            )
        ],
    )
    return path


@dataclass
class Built:
    fixture: Fixture
    spec_path: Path
    build_dir: Path
    binding_path: Path
    binding: dict[str, Any]
    sha256: str


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Built]:
    """One admitted build (holdout applied) and its binding, shared by every test."""
    root = tmp_path_factory.mktemp("rehearsal")
    fixture = Fixture(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            rehearsal_build, "contamination_inputs", lambda *_: ([HELD_OUT], {"fixture": "held"})
        )
        # The tree under test may be dirty; the fixture build records a clean commit.
        patch.setattr(rehearsal_build, "git_state", lambda: ("f" * 40, False))
        # The fixture tokenizer is small; the S model's 65,536 is tested separately.
        patch.setattr(rehearsal_binding, "VOCAB_SIZE", fixture.tokenizer.vocab_size)
        spec_path = fixture.spec()
        spec = rehearsal_build.load_spec(spec_path)
        rehearsal_build.download(
            spec,
            fixture.build,
            token="hf",
            hub_factory=lambda t: FakeHub(t, fixture.files),
            progress=quiet,
        )
        rehearsal_build.licenses(
            spec, fixture.build, client=FakeStack(fixture.stack), workers=1, progress=quiet
        )
        rehearsal_build.build(
            spec,
            fixture.build,
            workers=1,
            holdout=write_holdout(root),
            admitted_on="2026-10-25",
            progress=quiet,
        )
        path = root / "binding.json"
        binding = build_binding(fixture.build, spec_path, path, r2_prefix=PREFIX)
        yield Built(fixture, spec_path, fixture.build, path, binding, file_sha256(path))


def copy_build(built: Built, tmp_path: Path) -> Path:
    """A private copy of the build records and shards, safe to tamper with."""
    build_dir = tmp_path / "build"
    shutil.copytree(built.build_dir / "output", build_dir / "output")
    return build_dir


def edit_json(path: Path, change: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2))
    return value


def reverify(output: Path, report: Mapping[str, Any]) -> None:
    """Make verification.json agree with a tampered report, as a careless builder might."""
    edit_json(
        output / "verification.json",
        lambda v: v.update(
            report_sha256=file_sha256(output / "report.json"),
            tokens_identity=report["tokens_identity"],
            build_identity=report["build_identity"],
        ),
    )


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


class R2Reply(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class FakeR2:
    """Objects by key, served through ``Store``'s opener; records each GET."""

    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)
        self.gets: list[str] = []

    def __call__(self, request: Request) -> R2Reply:
        assert request.get_method() == "GET" and "Authorization" in dict(request.header_items())
        key = urlsplit(request.full_url).path.split("/", 2)[2]
        self.gets.append(key)
        return R2Reply(self.objects.get(key, b""), 200 if key in self.objects else 404)

    def store(self, credentials: Credentials | None = None) -> Store:
        credentials = credentials or Credentials(
            "https://acct.r2.cloudflarestorage.com", "id", "secret", "bucket-a"
        )
        return Store(credentials, opener=self, sleep=lambda _: None)


def staged(built: Built) -> FakeR2:
    """A bucket holding exactly what ``stage`` would upload."""
    return FakeR2(
        {key: path.read_bytes() for key, path in staging_files(built.binding, built.build_dir)}
    )


def pod_root(built: Built, tmp_path: Path, run: str = "A") -> Path:
    root = tmp_path / "pod"
    fetch(
        built.binding,
        root,
        staged(built).store(),
        run=run,
        receipt=tmp_path / "r.json",
        progress=quiet,
    )
    return root


def forge(
    built: Built, tmp_path: Path, change: Callable[[dict[str, Any]], object]
) -> tuple[Path, str]:
    """A canonically written binding with one change, and its SHA-256."""
    binding = copy.deepcopy(built.binding)
    change(binding)
    path = tmp_path / "forged.json"
    path.write_bytes(rehearsal_binding._serialized(binding))
    return path, file_sha256(path)


def test_the_walkthrough_example_by_hand() -> None:
    shards = [("a" * 64, 3), ("b" * 64, 2)]
    order = pass_order(shards, 17)
    assert order.tolist() == [(0, 0), (0, 1), (1, 1), (1, 0), (0, 2)]
    assert order_digest(order).startswith("fbf104b0") and order.dtype.itemsize == 6
    assert pass_order(shards, 18).tolist() == [(1, 0), (1, 1), (0, 0), (0, 2), (0, 1)]
    digests = [f"{value:064x}" for value in (0x05, 0x03, 0x0A)]
    assert multiset_digest(digests) == multiset_digest(reversed(digests)) == f"{0x12:064x}"
    assert rehearsal_binding._packed(1_000_000_000) == {
        "tokens": 1_000_000_000,
        "targets": 1_000_000_000,
        "windows": 976_563,
        "planned_updates": 7_630,
    }
    assert 976_563 - 7_629 * 128 == 51
    assert rehearsal_binding._packed(2_000_000_000)["windows"] == 1_953_125
    assert rehearsal_binding._packed(2_000_000_000)["planned_updates"] == 15_259
    assert rehearsal_binding.PACKING["tokens_per_update"] == 131_072


def test_binding_pins_files_in_spec_order_with_stage_orders_and_one_multiset(
    built: Built,
) -> None:
    binding = built.binding
    spec = rehearsal_build.load_spec(built.spec_path)
    output = built.build_dir / "output"
    report = json.loads((output / "report.json").read_text())
    assert binding["stream_order"] == [stream.name for stream in spec.streams]
    assert binding["stream_order"] != list(report["streams"])
    assert binding["build"]["tokens_identity"] == report["tokens_identity"]
    assert binding["build"]["report_sha256"] == file_sha256(output / "report.json")
    assert binding["build"]["holdout_sha256"] == report["holdout"]["sha256"]
    assert load_binding(built.binding_path, expected_sha256=built.sha256) == binding
    for name in binding["stream_order"]:
        for shard in binding["streams"][name]["shards"]:
            path = output / shard["file"]
            assert shard["file"].startswith(f"shards/{name}/")
            assert shard["key"] == PREFIX + shard["file"]
            assert shard["size_bytes"] == path.stat().st_size
            assert shard["sha256"] == file_sha256(path)
            manifest = output / (shard["file"] + ".json")
            assert shard["manifest_sha256"] == file_sha256(manifest)
            assert shard["manifest_size_bytes"] == manifest.stat().st_size
    assert len(binding["streams"]["stage1-web"]["shards"]) >= 2
    tokenizer = binding["tokenizer"]
    assert tokenizer["path"] == "tokenizer/tokenizer.json"
    assert tokenizer["key"] == PREFIX + "tokenizer/tokenizer.json"
    assert tokenizer["sha256"] == file_sha256(built.fixture.tokenizer_path)
    rows = [json.loads(line) for line in (output / "documents.jsonl").read_text().splitlines()]
    for run, stages in RUNS.items():
        for (name, streams), stage in zip(stages, binding["runs"][run]["stages"], strict=True):
            assert stage["name"] == name and stage["streams"] == list(streams)
            assert stage["order_seed"] == 17
            shards = [s for n in streams for s in binding["streams"][n]["shards"]]
            assert stage["shards"] == [shard["file"] for shard in shards]
            order = pass_order([(s["sha256"], s["documents"]) for s in shards], 17)
            assert stage["order_sha256"] == order_digest(order)
            selected = [row for row in rows if row["stream"] in streams]
            assert stage["documents"] == len(selected) == len(order)
            assert stage["tokens"] == stage["targets"] == sum(row["tokens"] for row in selected)
            assert stage["windows"] == -(-stage["tokens"] // 1024)
            assert stage["planned_updates"] == -(-stage["windows"] // 128)
            assert stage["multiset_sha256"] == multiset_digest(r["tokens_sha256"] for r in selected)
    two_stage = [stage["multiset_sha256"] for stage in binding["runs"]["A"]["stages"]]
    mixed = binding["runs"]["B"]["stages"][0]["multiset_sha256"]
    assert combine_digests(*two_stage) == mixed == binding["same_multiset"]["A"]
    assert binding["runs"]["A"]["tokens"] == binding["runs"]["B"]["tokens"]
    assert binding["validation"] == {
        "streams": ["validation-web", "validation-python"],
        "subset_rule": {"target_tokens_per_domain": 1_000_000, "sequence_length": 1024},
    }
    with pytest.raises(FileExistsError):
        build_binding(built.build_dir, built.spec_path, built.binding_path, r2_prefix=PREFIX)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(status="running"), "not completed"),
        (lambda r: r.update(dirty=True), "clean"),
        (lambda r: r.update(spec_sha256="0" * 64), "different spec"),
        (lambda r: r.update(final_training_admission=False, admission=None), "not admitted"),
        (lambda r: r.pop("holdout"), "holdout"),
    ],
)
def test_unfinished_dirty_or_unadmitted_builds_are_refused(
    built: Built, tmp_path: Path, change: Callable[[dict[str, Any]], object], message: str
) -> None:
    build_dir = copy_build(built, tmp_path)
    edit_json(build_dir / "output" / "report.json", change)
    with pytest.raises(ValueError, match=message):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    assert not (tmp_path / "b.json").exists()


def test_unverified_builds_and_stale_verifications_are_refused(
    built: Built, tmp_path: Path
) -> None:
    build_dir = copy_build(built, tmp_path)
    output = build_dir / "output"
    original = (output / "verification.json").read_bytes()
    for change, message in [
        (lambda v: v.update(status="failed"), "did not pass"),
        (lambda v: v.update(tokens_identity="0" * 64), "other identities"),
        (lambda v: v.update(final_training_admission=False), "no admission"),
    ]:
        edit_json(output / "verification.json", change)
        with pytest.raises(ValueError, match=message):
            build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
        (output / "verification.json").write_bytes(original)
    # A report edited after verification no longer matches the hash verification recorded.
    edit_json(output / "report.json", lambda r: r.update(note="edited"))
    with pytest.raises(ValueError, match="different report.json"):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    (output / "verification.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)


def test_tokens_identity_is_recomputed_in_spec_stream_order(built: Built, tmp_path: Path) -> None:
    build_dir = copy_build(built, tmp_path)
    output = build_dir / "output"
    report = json.loads((output / "report.json").read_text())
    spec = rehearsal_build.load_spec(built.spec_path)
    assert rehearsal_build.identities(spec, report)["tokens_identity"] == report["tokens_identity"]
    # The same hash over report.json's sorted stream keys names other tokens.
    assert list(report["streams"]) == sorted(report["streams"]) != [s.name for s in spec.streams]
    shards = [
        [name, shard["file"], shard["sha256"], shard["tokens"]]
        for name in report["streams"]
        for shard in report["streams"][name]["shards"]
    ]
    key_order = rehearsal_build._sha256_json(
        {"tokenizer": spec.tokenizer["identity"], "shards": shards}
    )
    assert key_order != report["tokens_identity"]
    # A report carrying that identity, even with a matching verification, is refused.
    report = edit_json(output / "report.json", lambda r: r.update(tokens_identity=key_order))
    reverify(output, report)
    with pytest.raises(ValueError, match="spec order"):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)


def test_build_refuses_tampered_shards_manifests_listings_and_vocabularies(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = copy_build(built, tmp_path)
    output = build_dir / "output"
    shard = output / built.binding["streams"]["stage2-web"]["shards"][0]["file"]
    original = shard.read_bytes()
    for target in (shard, shard.with_name(shard.name + ".json")):
        saved = target.read_bytes()
        flip_last_byte(target)
        with pytest.raises(ValueError, match="SHA-256 differs from the report"):
            build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
        target.write_bytes(saved)
    # A symlinked shard is refused even when it points at the right bytes.
    outside = tmp_path / "outside.bin"
    outside.write_bytes(original)
    shard.unlink()
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    shard.unlink()
    shard.write_bytes(original)
    # A token hash that disagrees with the shard bytes, even in a re-verified report.
    listing = output / "documents.jsonl"
    lines = listing.read_text().splitlines()
    row = json.loads(lines[0])
    row["tokens_sha256"] = hashlib.sha256(b"other tokens").hexdigest()
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    listing.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="documents.jsonl differs"):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    spec = rehearsal_build.load_spec(built.spec_path)

    def relisted(report: dict[str, Any]) -> None:
        report["documents"]["sha256"] = file_sha256(listing)
        report.update(rehearsal_build.identities(spec, report))

    reverify(output, edit_json(output / "report.json", relisted))
    with pytest.raises(ValueError, match="differ from the shard bytes"):
        build_binding(build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    # The S model's vocabulary is 65,536; the fixture tokenizer's is not.
    monkeypatch.setattr(rehearsal_binding, "VOCAB_SIZE", 65_536)
    with pytest.raises(ValueError, match="vocabulary 65536"):
        build_binding(built.build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    with pytest.raises(ValueError, match="ending in '/'"):
        build_binding(built.build_dir, built.spec_path, tmp_path / "b.json", r2_prefix="../x")
    assert not (tmp_path / "b.json").exists()


def test_run_a_must_read_run_bs_multiset(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(RUNS, "B", (("mixed", ("stage1-web", "stage2-web")),))
    with pytest.raises(ValueError, match="do not combine"):
        build_binding(built.build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX)
    assert not (tmp_path / "b.json").exists()


def test_order_digests_are_deterministic_and_seed_sensitive(built: Built, tmp_path: Path) -> None:
    again = build_binding(built.build_dir, built.spec_path, tmp_path / "a.json", r2_prefix=PREFIX)
    other = build_binding(
        built.build_dir, built.spec_path, tmp_path / "b.json", r2_prefix=PREFIX, order_seed=18
    )
    for run in RUNS:
        stages = zip(
            built.binding["runs"][run]["stages"],
            again["runs"][run]["stages"],
            other["runs"][run]["stages"],
            strict=True,
        )
        for first, second, reseeded in stages:
            assert first["order_sha256"] == second["order_sha256"]
            assert first["multiset_sha256"] == reseeded["multiset_sha256"]
            assert reseeded["order_seed"] == 18
            shards = [s for n in first["streams"] for s in built.binding["streams"][n]["shards"]]
            order = pass_order([(s["sha256"], s["documents"]) for s in shards], 18)
            assert reseeded["order_sha256"] == order_digest(order)
    # Six documents: another seed gives another order.
    mixed = built.binding["runs"]["B"]["stages"][0]
    assert other["runs"]["B"]["stages"][0]["order_sha256"] != mixed["order_sha256"]


def test_fetch_downloads_pinned_files_then_skips_present_ones(built: Built, tmp_path: Path) -> None:
    bucket = staged(built)
    root = tmp_path / "pod"

    def fetch_run(run: str = "A", receipt: str = "r.json") -> dict[str, Any]:
        return fetch(
            built.binding, root, bucket.store(), run=run, receipt=tmp_path / receipt, progress=quiet
        )

    needed = run_files(built.binding, "A")
    first = fetch_run(receipt="first.json")
    assert [row["method"] for row in first["files"]] == ["downloaded"] * len(needed)
    assert bucket.gets == [item["key"] for item in needed]
    assert needed[-1]["path"] == "tokenizer/tokenizer.json"
    receipt = json.loads((tmp_path / "first.json").read_text())
    assert receipt["binding_sha256"] == built.sha256 and receipt["run"] == "A"
    assert receipt["bytes_transferred"] == sum(item["size_bytes"] for item in needed)
    bucket.gets.clear()
    second = fetch_run()
    assert {row["method"] for row in second["files"]} == {"present"} and bucket.gets == []
    verify_binding(built.binding_path, root, expected_sha256=built.sha256, run="A")
    # A corrupted object is never published; a present file with other bytes stops the fetch.
    item = needed[0]
    target = root / item["path"]
    data = bytearray(target.read_bytes())
    data[0] ^= 1
    target.unlink()
    bucket.objects[item["key"]] = bytes(data)
    with pytest.raises(ValueError, match="pinned value"):
        fetch_run()
    assert not target.exists() and not target.with_name(target.name + ".part").exists()
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="other bytes"):
        fetch_run()
    with pytest.raises(ValueError, match="unknown run"):
        fetch_run("C")


def test_verify_returns_open_shards_recomputed_orders_and_the_tokenizer(
    built: Built, tmp_path: Path
) -> None:
    root = pod_root(built, tmp_path, run="B")
    before = os.stat(built.binding_path)
    result = verify_binding(built.binding_path, root, expected_sha256=built.sha256)
    after = os.stat(built.binding_path)
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    assert file_sha256(built.binding_path) == built.sha256
    assert list(result["streams"]) == built.binding["stream_order"]
    for name, opened in result["streams"].items():
        recorded = built.binding["streams"][name]["shards"]
        assert [sha for sha, _ in opened] == [shard["sha256"] for shard in recorded]
        assert [len(shard) for _, shard in opened] == [shard["documents"] for shard in recorded]
    assert result["tokenizer"].sha256 == built.binding["tokenizer"]["identity"]
    assert [stage["name"] for stage in result["stages"]["A"]] == ["stage1", "stage2"]
    mixed = result["stages"]["B"][0]
    assert order_digest(mixed["order"]) == mixed["order_sha256"]
    assert len(mixed["order"]) == mixed["documents"] == 6
    # Reading the documents in order gives back the binding's multiset of token hashes.
    digests = []
    for ordinal, index in mixed["order"].tolist():
        tokens = mixed["shards"][ordinal][1][index]
        digests.append(hashlib.sha256(np.asarray(tokens, dtype="<u2").tobytes()).hexdigest())
    assert multiset_digest(digests) == mixed["multiset_sha256"]
    only_a = verify_binding(built.binding_path, root, expected_sha256=built.sha256, run="A")
    assert list(only_a["stages"]) == ["A"]
    with pytest.raises(ValueError, match="expected value"):
        verify_binding(built.binding_path, root, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "target",
    [
        "shards/stage1-web/stage1-web-00000.bin",
        "shards/stage2-python/stage2-python-00000.bin.json",
        "shards/validation-python/validation-python-00000.bin",
        "tokenizer/tokenizer.json",
    ],
)
def test_verify_rejects_tampered_or_missing_files(
    built: Built, tmp_path: Path, target: str
) -> None:
    root = pod_root(built, tmp_path)
    flip_last_byte(root / target)
    with pytest.raises(ValueError, match="size or SHA-256 differs"):
        verify_binding(built.binding_path, root, expected_sha256=built.sha256)
    (root / target).unlink()
    with pytest.raises(FileNotFoundError):
        verify_binding(built.binding_path, root, expected_sha256=built.sha256)


def _stage(binding: dict[str, Any], run: str = "A", index: int = 0) -> dict[str, Any]:
    stage: dict[str, Any] = binding["runs"][run]["stages"][index]
    return stage


def _shard(binding: dict[str, Any], stream: str = "stage1-web") -> dict[str, Any]:
    shard: dict[str, Any] = binding["streams"][stream]["shards"][0]
    return shard


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: _stage(b).update(order_sha256="0" * 64), "document order differs"),
        (lambda b: _stage(b, "B").update(order_seed=18), "document order differs"),
        (lambda b: _shard(b).update(sha256="0" * 64), "size or SHA-256 differs"),
        (lambda b: _shard(b).update(manifest_size_bytes=1), "size or SHA-256 differs"),
        (lambda b: _shard(b).update(documents=_shard(b)["documents"] + 1), "totals differ"),
        (lambda b: _shard(b).update(file="shards/stage1-web/../x.bin"), "plain components"),
        (lambda b: _shard(b).update(key="other/" + _shard(b)["file"]), "R2 keys"),
        (lambda b: b["tokenizer"].update(identity="0" * 64), "tokenizer's identity"),
        (lambda b: b["tokenizer"].update(path="tokenizer/../../x.json"), "plain components"),
        (lambda b: _stage(b).update(multiset_sha256="0" * 64), "counts or multiset"),
        (lambda b: _stage(b).update(planned_updates=2), "counts or multiset"),
        (lambda b: b["same_multiset"].update(B="0" * 64), "same token multiset"),
        (lambda b: _stage(b, "B")["streams"].reverse(), "run design"),
        (lambda b: b["packing"].update(microbatch=32), "packing"),
        (lambda b: b["validation"]["subset_rule"].update(sequence_length=2048), "subset rule"),
    ],
)
def test_verify_rejects_tampered_binding_fields(
    built: Built, tmp_path: Path, change: Callable[[dict[str, Any]], object], message: str
) -> None:
    root = pod_root(built, tmp_path)
    path, sha256 = forge(built, tmp_path, change)
    with pytest.raises(ValueError, match=message):
        verify_binding(path, root, expected_sha256=sha256)


def test_a_reformatted_binding_is_refused_even_with_its_own_hash(
    built: Built, tmp_path: Path
) -> None:
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(built.binding, sort_keys=True, indent=4))
    with pytest.raises(ValueError, match="canonical"):
        load_binding(path, expected_sha256=file_sha256(path))


def test_symlinks_and_paths_outside_the_root_are_refused(built: Built, tmp_path: Path) -> None:
    root = pod_root(built, tmp_path)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        verify_binding(built.binding_path, linked_root, expected_sha256=built.sha256)
    # A shard replaced by a symlink to identical bytes.
    shard = root / _shard(built.binding)["file"]
    copied = tmp_path / "copied.bin"
    shutil.copyfile(shard, copied)
    shard.unlink()
    shard.symlink_to(copied)
    with pytest.raises(ValueError, match="not a symlink"):
        verify_binding(built.binding_path, root, expected_sha256=built.sha256)
    shard.unlink()
    shutil.copyfile(copied, shard)
    # A directory on the way that is a symlink, for verify and for fetch.
    moved = tmp_path / "moved-shards"
    (root / "shards").rename(moved)
    (root / "shards").symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="not a real directory"):
        verify_binding(built.binding_path, root, expected_sha256=built.sha256)
    bucket = staged(built)
    with pytest.raises(ValueError, match="not a real directory"):
        fetch(
            built.binding,
            root,
            bucket.store(),
            run="A",
            receipt=tmp_path / "r.json",
            progress=quiet,
        )
    # A traversal path is refused before anything is requested.
    forged = copy.deepcopy(built.binding)
    forged["tokenizer"]["path"] = "tokenizer/../../escape.json"
    with pytest.raises(ValueError, match="plain components"):
        fetch(forged, tmp_path / "fresh", bucket.store(), run="A", receipt=tmp_path / "r.json")
    assert bucket.gets == [] and not (tmp_path / "escape.json").exists()
    link = tmp_path / "binding-link.json"
    link.symlink_to(built.binding_path)
    with pytest.raises(ValueError, match="not a symlink"):
        load_binding(link, expected_sha256=built.sha256)


def test_staging_lists_every_pinned_file_and_the_binding_last(built: Built, tmp_path: Path) -> None:
    pairs = staging_files(built.binding, built.build_dir)
    keys = [key for key, _ in pairs]
    shards = [
        s
        for name in built.binding["stream_order"]
        for s in built.binding["streams"][name]["shards"]
    ]
    expected = {
        *(shard["key"] for shard in shards),
        *(shard["manifest_key"] for shard in shards),
        PREFIX + "tokenizer/tokenizer.json",
        *(PREFIX + name for name in ("documents.jsonl", "exclusions.jsonl")),
        *(PREFIX + name for name in ("report.json", "verification.json")),
    }
    assert set(keys[:-1]) == expected and len(keys) == len(set(keys))
    assert pairs[-1] == (f"{PREFIX}bindings/{built.sha256}.json", built.binding_path.absolute())
    assert binding_key(built.binding) == pairs[-1][0]
    assert dict(pairs)[PREFIX + "tokenizer/tokenizer.json"] == built.fixture.tokenizer_path
    # Every pinned file is re-hashed before upload.
    build_dir = copy_build(built, tmp_path)
    flip_last_byte(build_dir / "output" / shards[-1]["file"])
    with pytest.raises(ValueError, match="bytes differ"):
        staging_files(built.binding, build_dir)
    # The binding file on disk must be this binding.
    edited = copy.deepcopy(built.binding)
    edited["created_at"] = "2026-10-26T00:00:00+00:00"
    with pytest.raises(ValueError, match="is not this binding"):
        staging_files(edited, built.build_dir)


def test_credentials_use_the_named_key_pair_and_are_never_printed(
    built: Built,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environ = {
        "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
        "R2_BUCKET": "scratch-coder-artifacts",
        "R2_ACCESS_KEY_ID": "owner-id",
        "R2_SECRET_ACCESS_KEY": "owner-secret-value",
        "R2_POD_ACCESS_KEY_ID": "pod-id",
        "R2_POD_SECRET_ACCESS_KEY": "pod-secret-value",
    }
    assert r2_credentials(environ).access_key_id == "owner-id"
    pod = r2_credentials(environ, key_prefix="R2_POD_")
    assert (pod.access_key_id, pod.secret_access_key) == ("pod-id", "pod-secret-value")
    with pytest.raises(ValueError, match="R2_POD_SECRET_ACCESS_KEY") as missing:
        r2_credentials({**environ, "R2_POD_SECRET_ACCESS_KEY": ""}, key_prefix="R2_POD_")
    assert "secret-value" not in str(missing.value)
    with pytest.raises(ValueError, match="key prefix"):
        r2_credentials(environ, key_prefix="r2-pod")
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    bucket = staged(built)
    used: list[Credentials] = []

    def store(credentials: Credentials) -> Store:
        used.append(credentials)
        return bucket.store(credentials)

    monkeypatch.setattr(rehearsal_binding, "Store", store)
    root, receipt = tmp_path / "pod", tmp_path / "fetch.json"
    common = ["--binding", str(built.binding_path), "--expected-sha256", built.sha256]
    args = ["fetch", *common, "--root", str(root), "--run", "B", "--receipt", str(receipt)]
    assert main([*args, "--key-prefix", "R2_POD_"]) == 0
    assert used[-1].access_key_id == "pod-id"
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{") :])
    assert [stage["name"] for stage in summary["stages"]["B"]] == ["mixed"]
    assert summary["binding_sha256"] == built.sha256
    assert main(["verify", *common, "--root", str(root)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert (
        verified["stages"]["A"][1]["order_sha256"] == _stage(built.binding, "A", 1)["order_sha256"]
    )
    uploaded: list[list[tuple[str, Path]]] = []

    def upload(_: Store, files: Any, **__: Any) -> dict[str, Any]:
        uploaded.append(list(files))
        return {"bucket_bytes_after": 1, "monthly_storage_usd_after": 0.0}

    monkeypatch.setattr(rehearsal_binding, "upload", upload)
    stage_args = [
        "stage",
        *common,
        "--build",
        str(built.build_dir),
        "--receipt",
        str(tmp_path / "s.json"),
    ]
    assert main(stage_args) == 0
    assert used[-1].access_key_id == "owner-id"
    assert uploaded[0][-1][0] == json.loads(capsys.readouterr().out)["binding_key"]
    output = tmp_path / "cli.json"
    build_args = ["build", "--build", str(built.build_dir), "--spec", str(built.spec_path)]
    assert main([*build_args, "--output", str(output), "--r2-prefix", PREFIX]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["sha256"] == file_sha256(output)
    everything = out + printed.out + printed.err + receipt.read_text()
    assert "secret-value" not in everything
