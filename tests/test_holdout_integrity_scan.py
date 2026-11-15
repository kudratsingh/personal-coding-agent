"""The holdout integrity scan: frozen family rules against a sealed split, logged, leak-free.

Every fixture here is synthetic. The "sealed" file is a temporary ``test.jsonl`` whose IDs,
prompts, references and tests carry marker strings, so a test can assert that none of them
reaches any output, the ledger or the console.
"""

import ast
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import scratchcoder.data.integrity_scan as integrity_scan
import tools.holdout_integrity_scan as tool
from scratchcoder.data.clusters import canonical_key, solution_key
from scratchcoder.data.integrity_scan import (
    SealedContentError,
    item_of,
    parse_sealed_tasks,
    scan,
)
from scratchcoder.data.pool import shingles
from scratchcoder.eval.seal import SealError, SealLedger
from scratchcoder.eval.tasks import Task

WORDS = [f"alpha{index}" for index in range(20)]
OTHER = [f"omega{index}" for index in range(20)]


def prompt(shared: int, *, tail: str = "omega") -> str:
    """The first ``shared`` of the test prompt's 20 words, then distinct filler words."""
    return " ".join(WORDS[:shared] + [f"{tail}{index}" for index in range(20 - shared)])


TEST_PROMPT = " ".join(WORDS) + " SECRETPROMPTMARK"
TEST_SOLUTION = (
    "def secretfnmark(values):\n    total = 0\n    for value in values:\n"
    "        total += value * 3\n    return total\n"
)
OTHER_SOLUTION = "def unrelated(a, b):\n    return a - b\n"


def task(identifier: str, text: str, solution: str, *, split: str = "train") -> Task:
    return Task(
        id=identifier,
        prompt=text,
        public_tests="assert True",
        hidden_tests="assert HIDDENTESTMARK == 1" if split == "test" else "assert 1 == 1",
        source="example/source",
        license="CC-BY-4.0",
        written_by="model",
        split=split,
        entry_point=None,
        solution=solution,
    )


def sealed_tasks() -> list[Task]:
    return [
        task("SECRETIDMARK-0", TEST_PROMPT, TEST_SOLUTION, split="test"),
        task(
            "SECRETIDMARK-1",
            " ".join(f"zeta{index}" for index in range(20)) + " SECRETPROMPTMARK",
            "def secondsecret(x):\n    return x ** 5 + 11\n",
            split="test",
        ),
    ]


RENAMED = TEST_SOLUTION.replace("total", "acc").replace("value", "item")
SUBJECTS = {
    "copy": task("s-copy", prompt(0, tail="copy"), TEST_SOLUTION),
    "renamed": task("s-renamed", prompt(0, tail="beta"), RENAMED.replace("secretfnmark", "g")),
    "para": task("s-para", prompt(16, tail="para"), OTHER_SOLUTION),
    "named": task(
        "s-named", prompt(10, tail="named"), "def secretfnmark(v):\n    return sorted(v)\n"
    ),
    "flag": task("s-flag", prompt(6, tail="flag"), "def secretfnmark(v):\n    return max(v)\n"),
    "bridge": task("s-bridge", prompt(16, tail="gamma"), "def bridgefn(q):\n    return q[::-1]\n"),
    "chain": task("s-chain", prompt(0, tail="delta"), "def bridgefn(q):\n    return q[::-1]\n"),
    "clear": task("s-clear", " ".join(OTHER), "def fine(s):\n    return s.upper()\n"),
}


def test_fixture_prompts_sit_where_the_rules_put_them() -> None:
    def jac(a: str, b: str) -> float:
        x, y = shingles(a, 3), shingles(b, 3)
        return len(x & y) / len(x | y)

    assert jac(prompt(16), TEST_PROMPT) >= 0.5
    assert 0.25 <= jac(prompt(10), TEST_PROMPT) < 0.5
    assert 0.10 <= jac(prompt(6), TEST_PROMPT) < 0.25
    assert canonical_key(RENAMED.replace("secretfnmark", "g")) == canonical_key(TEST_SOLUTION)
    assert solution_key(RENAMED) != solution_key(TEST_SOLUTION)


def statuses() -> dict[str, tuple[str, tuple[str, ...]]]:
    results, _ = scan(
        [item_of(item) for item in sealed_tasks()],
        {"pilot": [item_of(item) for item in SUBJECTS.values()]},
    )
    return {result.id: (result.status, result.signals) for result in results}


def test_frozen_family_rules_decide_each_subject() -> None:
    found = statuses()
    assert found["s-copy"][0] == "excluded" and "solution_ast" in found["s-copy"][1]
    assert found["s-renamed"] == ("excluded", ("canonical_ast",))
    assert found["s-para"] == ("excluded", ("prompt_paraphrase",))
    assert found["s-named"] == ("excluded", ("prompt_named_paraphrase",))
    assert found["s-flag"] == ("flagged", ("prompt_named_flag",))
    assert found["s-bridge"] == ("excluded", ("prompt_paraphrase",))
    # In the test task's family only through another subject (equal reference ASTs).
    assert found["s-chain"] == ("excluded", ("family_chain",))
    assert found["s-clear"] == ("clear", ())


def test_a_subject_with_a_test_id_is_excluded() -> None:
    twin = task("SECRETIDMARK-0", " ".join(OTHER), OTHER_SOLUTION)
    results, counts = scan([item_of(item) for item in sealed_tasks()], {"pilot": [item_of(twin)]})
    assert [(r.status, r.signals) for r in results] == [("excluded", ("same_id",))]
    assert counts["subjects"]["pilot"]["excluded"] == 1


def test_subject_ids_must_be_unique_across_sets() -> None:
    clear = item_of(SUBJECTS["clear"])
    with pytest.raises(ValueError, match="unique"):
        scan([item_of(item) for item in sealed_tasks()], {"a": [clear], "b": [clear]})


def test_sealed_parse_errors_never_quote_the_file() -> None:
    good = json.dumps(asdict(sealed_tasks()[0]))
    for data in (
        b"\xff\xfe SECRETPROMPTMARK",
        (good + '\n{"id": "SECRETIDMARK-9"}\n').encode(),
        (good + "\n" + good + "\n").encode(),
        json.dumps(asdict(SUBJECTS["copy"])).encode(),  # a train task in the sealed file
    ):
        with pytest.raises(SealedContentError) as raised:
            parse_sealed_tasks(data)
        assert "SECRET" not in str(raised.value) and "s-copy" not in str(raised.value)


# The command --------------------------------------------------------------------------


def write_tasks(path: Path, tasks: list[Task]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(item)) + "\n" for item in tasks))
    return path


def setup(root: Path, *, pinned_sha: str | None = None) -> dict[str, Path]:
    sealed = write_tasks(root / "pool" / "test.jsonl", sealed_tasks())
    report = root / "pool" / "report.json"
    digest = pinned_sha or tool._sha_file(sealed)
    report.write_text(json.dumps({"files": {"test": {"file_sha256": digest, "tasks": 2}}}))
    pilot = write_tasks(root / "subjects" / "pilot.jsonl", list(SUBJECTS.values()))
    ledger = root / "seal" / "ledger.jsonl"
    return {"sealed": sealed, "report": report, "pilot": pilot, "ledger": ledger}


def request(root: Path, paths: dict[str, Path], out: Path) -> Path:
    argv = ["request", "--sealed", str(paths["sealed"]), "--pool-report", str(paths["report"])]
    argv += ["--subject", f"pilot={paths['pilot']}", "--ledger", str(paths["ledger"])]
    assert tool.main([*argv, "--out", str(out)]) == 0
    return out / tool.REQUEST_NAME


def approve(path: Path, destination: Path | None = None) -> Path:
    document = json.loads(path.read_text())
    document["approval"] = {
        "approved_by": "owner",
        "approved_at": "2026-11-14T00:00:00Z",
        "statement": "approved for this synthetic test",
    }
    target = destination or path.with_name("approval.json")
    target.write_text(json.dumps(document, indent=2, sort_keys=True))
    return target


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every read of the sealed file through the tool."""
    reads: list[str] = []
    original = tool._read_sealed

    def recording(path: Path) -> bytes:
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(tool, "_read_sealed", recording)
    return reads


def markers(paths: dict[str, Path]) -> list[str]:
    found = ["SECRETIDMARK", "SECRETPROMPTMARK", "HIDDENTESTMARK", "secretfnmark", "secondsecret"]
    for item in sealed_tasks():
        for key in (solution_key(item.solution), canonical_key(item.solution)):
            assert key is not None
            found.append(key[:16])
    return found


def test_request_opens_nothing_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = setup(tmp_path)
    real_open = Path.open
    real_read_bytes = Path.read_bytes

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        assert self.name != "test.jsonl", "request opened the sealed file"
        return real_open(self, *args, **kwargs)

    def guarded_read_bytes(self: Path) -> bytes:
        assert self.name != "test.jsonl", "request read the sealed file"
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    made = request(tmp_path, paths, tmp_path / "scan")
    document = json.loads(made.read_text())
    assert document["approval"] == dict.fromkeys(tool.APPROVAL_FIELDS)
    assert document["request"]["sealed"]["tasks"] == 2
    assert not paths["ledger"].exists()


def test_approved_scan_logs_reads_once_and_leaks_nothing(
    tmp_path: Path, opened: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    paths = setup(tmp_path)
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    capsys.readouterr()
    assert tool.main(["run", "--approval", str(approval)]) == 0
    printed = capsys.readouterr()
    assert opened == [str(paths["sealed"].resolve())]
    exclusions = json.loads((tmp_path / "scan" / "exclusions.json").read_text())
    assert exclusions["excluded"]["pilot"] == sorted(
        ["s-copy", "s-renamed", "s-para", "s-named", "s-bridge", "s-chain"]
    )
    assert exclusions["flagged"]["pilot"] == ["s-flag"]
    events = SealLedger(paths["ledger"]).events()
    assert [event["event"] for event in events] == [
        "integrity_scan_opened",
        "integrity_scan_closed",
    ]
    assert events[1]["outcome"] == "completed"
    texts = [printed.out, printed.err, paths["ledger"].read_text()]
    texts += [(tmp_path / "scan" / name).read_text() for name in tool.OUTPUTS]
    for marker in markers(paths):
        for text in texts:
            assert marker not in text, marker
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["counts"]["subjects"]["pilot"] == {
        "excluded": 6,
        "flagged": 1,
        "clear": 1,
        "total": 8,
    }
    assert summary["counts"]["test_tasks_with_any_subject_edge"] == 1


def test_one_approval_allows_one_scan(tmp_path: Path, opened: list[str]) -> None:
    paths = setup(tmp_path)
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    assert tool.main(["run", "--approval", str(approval)]) == 0
    for name in tool.OUTPUTS:
        (tmp_path / "scan" / name).unlink()
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert len(opened) == 1
    assert len(SealLedger(paths["ledger"]).events()) == 2


def edited(approval: Path, change) -> Path:  # type: ignore[no-untyped-def]
    document = json.loads(approval.read_text())
    change(document)
    approval.write_text(json.dumps(document))
    return approval


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["approval"].update(approved_by=""),
        lambda d: d["approval"].update(extra="x"),
        lambda d: d["request"]["code"].update({"scratchcoder/data/clusters.py": "0" * 64}),
        lambda d: d["request"].update(rules_version="task-clusters-v0"),
        lambda d: d["request"]["performs"].update(scoring=True),
        lambda d: d["request"]["subjects"][0].update(sha256="0" * 64),
        lambda d: d.update(request="not a request"),
    ],
)
def test_refusals_before_opening_log_nothing(tmp_path: Path, opened: list[str], change) -> None:  # type: ignore[no-untyped-def]
    paths = setup(tmp_path)
    approval = edited(approve(request(tmp_path, paths, tmp_path / "scan")), change)
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert opened == [] and not paths["ledger"].exists()


def test_a_linked_or_misnamed_sealed_path_is_refused(tmp_path: Path, opened: list[str]) -> None:
    paths = setup(tmp_path)
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    real = paths["sealed"].with_name("real.jsonl")
    paths["sealed"].rename(real)
    os.symlink(real, paths["sealed"])
    assert tool.main(["run", "--approval", str(approval)]) == 2
    holdout = tmp_path / "pool" / "holdout.jsonl"
    holdout.write_text("")
    argv = ["request", "--sealed", str(holdout), "--pool-report", str(paths["report"])]
    argv += ["--subject", f"pilot={paths['pilot']}", "--ledger", str(paths["ledger"])]
    assert tool.main([*argv, "--out", str(tmp_path / "other")]) == 2
    assert opened == [] and not paths["ledger"].exists()


def test_a_sealed_subject_is_refused(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    argv = ["request", "--sealed", str(paths["sealed"]), "--pool-report", str(paths["report"])]
    argv += ["--subject", f"pilot={paths['sealed']}", "--ledger", str(paths["ledger"])]
    assert tool.main([*argv, "--out", str(tmp_path / "scan")]) == 2


def test_existing_outputs_are_refused_before_opening(tmp_path: Path, opened: list[str]) -> None:
    paths = setup(tmp_path)
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    (tmp_path / "scan" / "summary.json").write_text("{}")
    assert tool.main(["run", "--approval", str(approval)]) == 3
    assert opened == [] and not paths["ledger"].exists()


def test_a_sealed_hash_mismatch_is_logged_as_failed(tmp_path: Path, opened: list[str]) -> None:
    paths = setup(tmp_path, pinned_sha="0" * 64)
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    assert tool.main(["run", "--approval", str(approval)]) == 4
    events = SealLedger(paths["ledger"]).events()
    assert [(e["event"], e.get("outcome"), e.get("error")) for e in events] == [
        ("integrity_scan_opened", None, None),
        ("integrity_scan_closed", "failed", "sealed_sha256_mismatch"),
    ]
    assert not (tmp_path / "scan" / "exclusions.json").exists()


def test_a_parse_failure_is_logged_without_quoting(
    tmp_path: Path, opened: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    paths = setup(tmp_path)
    paths["sealed"].write_text(json.dumps(asdict(sealed_tasks()[0])) + "\n{SECRETIDMARK}\n")
    report = json.loads(paths["report"].read_text())
    report["files"]["test"]["file_sha256"] = tool._sha_file(paths["sealed"])
    paths["report"].write_text(json.dumps(report))
    approval = approve(request(tmp_path, paths, tmp_path / "scan"))
    capsys.readouterr()
    assert tool.main(["run", "--approval", str(approval)]) == 5
    printed = capsys.readouterr()
    assert "SECRET" not in printed.out + printed.err + paths["ledger"].read_text()
    assert SealLedger(paths["ledger"]).events()[-1]["error"] == "SealedContentError"


def test_scan_events_leave_the_seal_rules_alone(tmp_path: Path) -> None:
    ledger = SealLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(SealError, match="no integrity scan"):
        ledger.close_integrity_scan("a" * 64, "completed")
    ledger.open_integrity_scan("a" * 64, tool="x")
    with pytest.raises(SealError, match="already used"):
        ledger.open_integrity_scan("a" * 64, tool="x")
    with pytest.raises(ValueError, match="outcome"):
        ledger.close_integrity_scan("a" * 64, "maybe")
    ledger.close_integrity_scan("a" * 64, "completed")
    with pytest.raises(SealError, match="already closed"):
        ledger.close_integrity_scan("a" * 64, "failed")
    events = ledger.events()
    assert not any(event["event"] in ("opened", "completed") for event in events)


def test_the_scan_imports_no_generation_execution_or_scoring_code() -> None:
    forbidden = (
        "scratchcoder.sandbox",
        "scratchcoder.engine",
        "scratchcoder.rl",
        "torch",
        "scratchcoder.eval.harness",
        "scratchcoder.eval.executor",
        "subprocess",
    )
    for path in (Path(tool.__file__), Path(str(integrity_scan.__file__))):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"
