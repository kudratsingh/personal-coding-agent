"""Offline tests for the rehearsal task pool; authored fixtures and a fake sandbox only."""

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.sandbox.runner import Result
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import task_pool
from tools.rehearsal_build import Hub

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REVISION = {"opencode": "1" * 40, "acecode": "2" * 40, "taco": "3" * 40, "mbpp": "4" * 40}
MBPP_BYTES = b"authored mbpp fixture bytes\n"


def quiet(_: str) -> None:
    return None


def parquet(columns: dict[str, Any]) -> bytes:
    sink = io.BytesIO()
    pq.write_table(pa.table(columns), sink, row_group_size=3)
    return sink.getvalue()


def function_task(name: str, value: int) -> tuple[str, str, list[str]]:
    prompt = f"Write {name}(x) returning x plus {value} for the integer x given."
    solution = f"def {name}(x):\n    return x + {value}\n"
    tests = [f"assert {name}({i}) == {i + value}" for i in range(3)]
    return prompt, solution, tests


def opencode_file(count: int) -> bytes:
    rows = [function_task(f"opencode_{i}", i) for i in range(count)]
    # One row whose reference is wrong, one whose tests are not assertions.
    rows[1] = (rows[1][0], rows[1][1].replace("+", "-"), rows[1][2])
    return parquet(
        {
            "id": [f"oc{i}" for i in range(count)],
            "input": [prompt for prompt, _, _ in rows],
            "output": [f"```python\n{solution}```" for _, solution, _ in rows],
            "unit_tests": [
                json.dumps(tests if i != 2 else ["print('not an assertion')"] * 3)
                for i, (_, _, tests) in enumerate(rows)
            ],
        }
    )


def acecode_file(count: int, first: int) -> bytes:
    rows = [function_task(f"acecode_{first + i}", first + i) for i in range(count)]
    return parquet(
        {
            "id": [f"ac{first + i}" for i in range(count)],
            "source": ["oss"] * count,
            "question": [prompt for prompt, _, _ in rows],
            "test_cases": [tests for _, _, tests in rows],
            "inferences": [
                [
                    {
                        "completion": solution,
                        "completion_id": 0,
                        "model_name": "fixture",
                        # The first row has no attempt that passed every test.
                        "pass_rate": 0.5 if first + i == 0 else 1.0,
                    }
                ]
                for i, (_, solution, _) in enumerate(rows)
            ],
        }
    )


def taco_file(rows: list[dict[str, Any]]) -> bytes:
    keys = [
        "question",
        "solutions",
        "starter_code",
        "input_output",
        "difficulty",
        "source",
        "url",
        "picture_num",
    ]
    return parquet({key: [row[key] for row in rows] for key in keys})


def taco_row(index: int, *, source: str = "codeforces", difficulty: str = "EASY") -> dict[str, Any]:
    inputs = [f"{index} {j}\n" for j in range(3)]
    return {
        "question": f"Problem {index} asks you to read integers {index} and print their sum.",
        "solutions": json.dumps(["a, b = map(int, input().split())\nprint(a + b)"]),
        "starter_code": "",
        "input_output": json.dumps(
            {"inputs": inputs, "outputs": [f"{index + j}\n" for j in range(3)]}
        ),
        "difficulty": difficulty,
        "source": source,
        "url": f"https://example.invalid/taco/{index}",
        "picture_num": None,
    }


def mbpp_rows() -> list[dict[str, Any]]:
    rows = []
    for task_id in range(1, 9):
        prompt, solution, tests = function_task(f"mbpp_{task_id}", task_id)
        rows.append(
            {
                "task_id": task_id,
                "text": prompt,
                "code": solution,
                "test_list": tests,
                "test_setup_code": "",
                "challenge_test_list": [],
            }
        )
    return rows


class FakeHub(Hub):
    def __init__(self, files: Mapping[tuple[str, str], bytes]) -> None:
        super().__init__(None, sleep=lambda _: None)
        self.files = files

    def json(self, url: str) -> Any:
        dataset, revision = url.split("/api/datasets/")[1].split("?")[0].split("/revision/")
        return {
            "sha": revision,
            "siblings": [
                {
                    "rfilename": name,
                    "size": len(data),
                    "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                }
                for (repo, name), data in self.files.items()
                if repo == dataset
            ],
        }

    def download(self, url: str, destination: Path, size: int, sha256: str) -> int:
        dataset, rest = url.split("/datasets/")[1].split("/resolve/")
        data = self.files[(dataset, rest.split("/", 1)[1])]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return len(data)


class Sandbox:
    """Runs nothing: passes unless the code subtracts, and counts calls."""

    def __init__(self, status: str | None = None) -> None:
        self.calls = 0
        self.status = status

    def __call__(self, code: str, tests: str, mode: str) -> Result:
        self.calls += 1
        if self.status is not None:
            return Result(self.status)
        if "-" in code.split("return", 1)[-1]:
            return Result("failed", "AssertionError\n", 1)
        return Result("passed")


class Fixture:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        taco_train = [taco_row(i) for i in range(6)] + [
            taco_row(50, source="HackerRank"),
            taco_row(51, difficulty="HARD"),
        ]
        taco_test = [taco_row(100), taco_row(101)]
        self.files = {
            ("nvidia/OpenCodeInstruct", "data/oc-0.parquet"): opencode_file(12),
            ("TIGER-Lab/AceCode-87K", "data/ac-0.parquet"): acecode_file(5, 0),
            ("TIGER-Lab/AceCode-87K", "data/ac-1.parquet"): acecode_file(5, 5),
            ("BAAI/TACO", "ALL/train-00000-of-00001.parquet"): taco_file(taco_train),
            ("BAAI/TACO", "ALL/test-00000-of-00001.parquet"): taco_file(taco_test),
        }
        self.tokenizer = ByteBPE.train(["def f(x):\n    return x + 1\n"], vocab_size=300)
        self.tokenizer_path = root / "tokenizer.json"
        self.tokenizer.save(self.tokenizer_path)
        self.mbpp = root / "mbpp.jsonl"
        self.mbpp.write_bytes(MBPP_BYTES)
        protected = Task(
            id="Mbpp/3",
            prompt="Protected MBPP+ prompt about something else entirely different",
            public_tests="assert True",
            hidden_tests="assert True",
            source="evalplus/mbpp-plus",
            license="Apache-2.0",
            written_by="human",
            split="test",
        )
        choice = ChoiceTask(
            id="arc/1",
            question="Which planet is closest to the sun in our solar system today?",
            labels=("A", "B"),
            choices=("Mercury", "Venus"),
            answer="A",
            source="allenai/ai2_arc",
            source_revision="5" * 40,
            license="CC-BY-SA-4.0",
            split="test",
        )
        monkeypatch.setattr(
            task_pool, "protection", lambda _: ([protected], [choice], {"fixture": True})
        )
        monkeypatch.setattr(task_pool, "read_mbpp", lambda path, spec: mbpp_rows())
        self.directory = root / "pool-build"

    def spec(self, **splits: int) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "name": "fixture-pool",
            "tokenizer": {
                "path": str(self.tokenizer_path),
                "identity": self.tokenizer.sha256,
                "file_sha256": file_sha256(self.tokenizer_path),
                "vocab_size": self.tokenizer.vocab_size,
            },
            "context_tokens": 400,
            "context_basis": "fixture",
            "sources": [
                self.source("opencode", ["data/oc-0.parquet"], 10, "CC-BY-4.0"),
                self.source("acecode", ["data/ac-0.parquet", "data/ac-1.parquet"], 8, "MIT"),
                {
                    **self.source(
                        "taco",
                        ["ALL/train-00000-of-00001.parquet", "ALL/test-00000-of-00001.parquet"],
                        None,
                        "Apache-2.0",
                    ),
                    "difficulty": "EASY",
                    "excluded_sources": ["hackerrank"],
                    "truncate_cases": True,
                },
                {
                    **self.source("mbpp", [], None, "CC-BY-4.0"),
                    "url": "https://example.invalid/mbpp.jsonl",
                    "sha256": hashlib.sha256(MBPP_BYTES).hexdigest(),
                    "size_bytes": len(MBPP_BYTES),
                },
            ],
            "protection": {"humaneval": "x", "evalplus_directory": "x", "arc_directory": "x"},
            "splits": {"dev": 3, "test": 3, "human_holdout_minimum": 4, **splits},
            "verification": {
                "timeout_seconds": 2.0,
                "io_seconds_per_case": 0.25,
                "memory_mb": 128,
                "workers": 3,
            },
            "local_copies": {"google-research-datasets/mbpp:mbpp.jsonl": str(self.mbpp)},
            "max_download_bytes": 10**9,
        }
        return value

    def source(self, kind: str, files: list[str], rows: int | None, license: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "dataset": task_pool.DATASETS[kind],
            "revision": REVISION[kind],
            "files": files,
            "rows": rows,
            "license": license,
            "attribution": "fixture",
            "terms": "fixture terms",
            "terms_reviewed_on": "2026-10-25",
            "obligations": "fixture obligations",
        }

    def load(self, value: Mapping[str, Any]) -> dict[str, Any]:
        path = self.root / "spec.json"
        path.write_text(json.dumps(value))
        return task_pool.load_spec(path)

    def fetch(self, spec: Mapping[str, Any]) -> None:
        task_pool.fetch(
            spec, self.directory, hub_factory=lambda _: FakeHub(self.files), progress=quiet
        )


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return Fixture(tmp_path, monkeypatch)


def test_spec_requires_every_source_kind_allowlisted_licenses_and_explicit_taco_rules(
    fixture: Fixture,
) -> None:
    fixture.load(fixture.spec())
    value = fixture.spec()
    value["sources"] = value["sources"][:3]
    with pytest.raises(ValueError, match="one source of each kind"):
        fixture.load(value)
    value = fixture.spec()
    value["sources"][1]["license"] = "CC-BY-NC-4.0"
    with pytest.raises(ValueError, match="allowlist"):
        fixture.load(value)
    value = fixture.spec()
    value["sources"][2]["difficulty"] = "MEDIUM"
    with pytest.raises(ValueError, match="EASY"):
        fixture.load(value)


def test_download_verifies_hashes_links_local_copies_and_enforces_the_budget(
    fixture: Fixture,
) -> None:
    spec = fixture.load(fixture.spec())
    fixture.fetch(spec)
    receipts = json.loads((fixture.directory / "download-receipts.json").read_text())
    methods = {item["filename"]: item["method"] for item in receipts["files"]}
    assert methods["mbpp.jsonl"] == "hard-linked"
    assert set(methods.values()) == {"downloaded", "hard-linked"}
    with pytest.raises(FileExistsError):
        fixture.fetch(spec)
    small = fixture.load({**fixture.spec(), "max_download_bytes": 10})
    with pytest.raises(RuntimeError, match="approved maximum"):
        task_pool.fetch(small, fixture.root / "other", hub_factory=lambda _: FakeHub(fixture.files))


def test_build_verifies_references_records_every_drop_and_assigns_exact_holdouts(
    fixture: Fixture,
) -> None:
    spec = fixture.load(fixture.spec())
    fixture.fetch(spec)
    sandbox = Sandbox()
    report = task_pool.build(
        spec, fixture.directory, execute=sandbox, runtime={"image": "fixture"}, progress=quiet
    )
    intake = report["intake"]
    assert intake["opencode"]["selected_rows"] == 10
    assert intake["opencode"]["unsupported_schema"] == 1
    assert intake["opencode"]["invalid_solution"] == 1
    assert intake["acecode"]["selected_rows"] == 8
    assert intake["acecode"]["missing_solution"] == 1
    assert intake["taco"]["excluded_component"] == 1 and intake["taco"]["not_easy"] == 1
    assert intake["mbpp"]["mbpp_plus_id_excluded"] == 1
    assert report["kept_by_source"] == {"opencode": 8, "acecode": 7, "taco": 8, "mbpp": 7}
    pool = fixture.directory / "pool"
    splits = {name: list(load_tasks(pool / f"{name}.jsonl")) for name in ("train", "dev", "test")}
    assert len(splits["dev"]) == len(splits["test"]) == 3
    # Two TACO test tasks plus two MBPP tasks reach the human minimum of four.
    assert report["holdouts"]["taco_test_holdouts"] == 2
    assert report["holdouts"]["mbpp_holdouts_added"] == 2
    held = [t for name in ("dev", "test") for t in splits[name]]
    assert sum(t.written_by == "human" for t in held) == 4
    assert not any(t.prompt.startswith(("Problem 100 ", "Problem 101 ")) for t in splits["train"])
    assert {t.id for t in load_tasks(pool / "holdout.jsonl")} == {t.id for t in held}
    assert sandbox.calls == 2 * (8 + 1 + 7 + 8 + 7)


def test_verdicts_are_cached_for_resume_and_unavailable_isolation_stops_the_build(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = fixture.load(fixture.spec())
    fixture.fetch(spec)
    first = task_pool.build(spec, fixture.directory, execute=Sandbox(), runtime={}, progress=quiet)
    (fixture.directory / "pool").rename(fixture.directory / "pool-first")
    again = Sandbox()
    second = task_pool.build(spec, fixture.directory, execute=again, runtime={}, progress=quiet)
    assert again.calls == 0 and second["files"] == first["files"]
    monkeypatch.setattr(task_pool, "RETRY_DELAY_SECONDS", 0.0)
    broken = fixture.root / "broken"
    spec_two = fixture.load(fixture.spec())
    task_pool.fetch(spec_two, broken, hub_factory=lambda _: FakeHub(fixture.files), progress=quiet)
    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        task_pool.build(
            spec_two, broken, execute=Sandbox("unavailable"), runtime={}, progress=quiet
        )
    assert not (broken / "pool").exists()


def test_io_deadlines_grow_with_the_case_count_up_to_the_runner_maximum() -> None:
    verification = {"timeout_seconds": 2.0, "io_seconds_per_case": 0.25, "memory_mb": 128}
    cases = [{"input": "1\n", "output": "1\n"}]
    assert task_pool.limits_for(verification, "assert True", "function").timeout_seconds == 2.0
    assert task_pool.limits_for(verification, json.dumps(cases * 4), "io").timeout_seconds == 3.0
    assert task_pool.limits_for(verification, json.dumps(cases * 400), "io").timeout_seconds == 60
    assert task_pool.policy_hash({**verification, "workers": 1}) == task_pool.policy_hash(
        {**verification, "workers": 8}
    )
