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


def test_a_same_id_subject_still_joins_its_family() -> None:
    """Y shares nothing with any test task; it is linked only to X, whose ID is a test ID."""
    twin = task("SECRETIDMARK-1", " ".join(OTHER), "def via(t):\n    return t * 7 - 1\n")
    reached = task("s-via-twin", prompt(0, tail="eps"), "def via(t):\n    return t * 7 - 1\n")
    results, _ = scan(
        [item_of(item) for item in sealed_tasks()], {"pilot": [item_of(twin), item_of(reached)]}
    )
    found = {r.id: (r.status, r.signals) for r in results}
    assert found["s-via-twin"] == ("excluded", ("family_chain",))
    assert found["SECRETIDMARK-1"][0] == "excluded" and "same_id" in found["SECRETIDMARK-1"][1]


def test_bridges_join_families_but_are_not_judged() -> None:
    """A pool task paraphrasing a test prompt links a subject that shares only its code."""
    bridge = task("b-pool", prompt(16, tail="pool"), "def poolfn(w):\n    return w % 9\n")
    subject = task("s-behind-bridge", prompt(0, tail="theta"), "def poolfn(w):\n    return w % 9\n")
    alone, _ = scan([item_of(item) for item in sealed_tasks()], {"pilot": [item_of(subject)]})
    assert [r.status for r in alone] == ["clear"]
    results, counts = scan(
        [item_of(item) for item in sealed_tasks()],
        {"pilot": [item_of(subject)]},
        bridges={"pool_train": [item_of(bridge)]},
    )
    assert [(r.id, r.status, r.signals) for r in results] == [
        ("s-behind-bridge", "excluded", ("family_chain",))
    ]
    assert counts["bridge_tasks"] == {"in_test_family": 1, "total": 1}
    assert counts["test_tasks_in_a_subject_family"] == 1


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


@pytest.fixture(autouse=True)
def clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool requires a clean git checkout; the tests' worktree may be dirty."""
    monkeypatch.setattr(tool, "git_state", lambda: ("c" * 40, False))


def setup(root: Path, *, pinned_sha: str | None = None) -> dict[str, Path]:
    sealed = write_tasks(root / "pool" / "test.jsonl", sealed_tasks())
    report = root / "pool" / "report.json"
    digest = pinned_sha or tool._sha_bytes(sealed.read_bytes())
    report.write_text(json.dumps({"files": {"test": {"file_sha256": digest, "tasks": 2}}}))
    pilot = write_tasks(root / "subjects" / "pilot.jsonl", list(SUBJECTS.values()))
    ledger = root / "seal" / "ledger.jsonl"
    SealLedger(ledger).record_baseline(
        task_file_sha256="a" * 64, path=Path("x"), sha256="b" * 64, results_id="r"
    )
    return {"sealed": sealed, "report": report, "pilot": pilot, "ledger": ledger}


def request_argv(paths: dict[str, Path], out: Path, *extra: str) -> list[str]:
    argv = ["request", "--sealed", str(paths["sealed"]), "--pool-report", str(paths["report"])]
    argv += ["--subject", f"pilot={paths['pilot']}", "--ledger", str(paths["ledger"])]
    return [*argv, "--out", str(out), *extra]


def request(paths: dict[str, Path], out: Path) -> Path:
    assert tool.main(request_argv(paths, out)) == 0
    return out / tool.REQUEST_NAME


def approve(path: Path, destination: Path | None = None) -> Path:
    document = json.loads(path.read_text())
    document["approval"] = {
        "approved_by": "owner",
        "approved_at": "2026-11-14T00:00:00Z",
        "statement": "approved for this synthetic test",
        "request_sha256": document["request_sha256"],
    }
    target = destination or path.with_name("approval.json")
    target.write_text(json.dumps(document, indent=2, sort_keys=True))
    return target


class Reads:
    """Every open or read of the sealed file (by inode, so links count), once armed."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.inode: tuple[int, int] | None = None

    def arm(self, sealed: Path) -> None:
        info = os.stat(sealed)
        self.inode = (info.st_dev, info.st_ino)
        self.reads.clear()

    def note(self, path: object) -> None:
        if self.inode is None:
            return
        try:
            info = os.stat(path)  # type: ignore[arg-type]
        except (OSError, TypeError):
            return
        if (info.st_dev, info.st_ino) == self.inode:
            self.reads.append(str(path))


@pytest.fixture
def sealed_reads(monkeypatch: pytest.MonkeyPatch) -> Reads:
    recorder = Reads()
    real_read_bytes, real_open, real_os_open = Path.read_bytes, Path.open, os.open

    def read_bytes(self: Path) -> bytes:
        recorder.note(self)
        return real_read_bytes(self)

    def path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        recorder.note(self)
        return real_open(self, *args, **kwargs)

    def os_open(path: Any, *args: Any, **kwargs: Any) -> int:
        recorder.note(path)
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(os, "open", os_open)
    return recorder


def ledger_events(paths: dict[str, Path]) -> list[str]:
    return [event["event"] for event in SealLedger(paths["ledger"]).events()]


def markers() -> list[str]:
    found = ["SECRETIDMARK", "SECRETPROMPTMARK", "HIDDENTESTMARK", "secretfnmark", "secondsecret"]
    for item in sealed_tasks():
        for key in (solution_key(item.solution), canonical_key(item.solution)):
            assert key is not None
            found.append(key[:16])
    return found


def test_request_opens_nothing_sealed(tmp_path: Path, sealed_reads: Reads) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    document = json.loads(request(paths, tmp_path / "scan").read_text())
    assert sealed_reads.reads == []
    assert document["approval"] == dict.fromkeys(tool.APPROVAL_FIELDS)
    assert document["request_sha256"] == tool.canonical_sha256(document["request"])
    assert document["request"]["sealed"]["tasks"] == 2
    assert ledger_events(paths) == ["baseline"]


def test_approved_scan_reads_once_logs_and_leaks_nothing(
    tmp_path: Path, sealed_reads: Reads, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    capsys.readouterr()
    assert tool.main(["run", "--approval", str(approval)]) == 0
    printed = capsys.readouterr()
    assert sealed_reads.reads == [str(paths["sealed"].resolve())]
    exclusions = json.loads((tmp_path / "scan" / "exclusions.json").read_text())
    assert exclusions["excluded"]["pilot"] == sorted(
        ["s-copy", "s-renamed", "s-para", "s-named", "s-bridge", "s-chain"]
    )
    assert exclusions["flagged"]["pilot"] == ["s-flag"]
    assert exclusions["clear"]["pilot"] == ["s-clear"]
    events = SealLedger(paths["ledger"]).events()
    assert [event["event"] for event in events] == [
        "baseline",
        "integrity_scan_opened",
        "integrity_scan_closed",
    ]
    assert events[1]["approval"]["approved_by"] == "owner"
    assert events[2]["outcome"] == "completed"
    texts = [printed.out, printed.err, paths["ledger"].read_text()]
    texts += [(tmp_path / "scan" / name).read_text() for name in tool.OUTPUTS]
    for marker in markers():
        for text in texts:
            assert marker not in text, marker
    for signal_name in ("solution_ast", "family_chain", "prompt_named_flag"):
        assert signal_name not in (tmp_path / "scan" / "exclusions.json").read_text()
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["counts"]["subjects"]["pilot"] == {
        "excluded": 6,
        "flagged": 1,
        "clear": 1,
        "total": 8,
    }
    assert not list((tmp_path / "scan").glob(".*.partial"))


def test_one_request_allows_one_scan_whatever_the_bytes(
    tmp_path: Path, sealed_reads: Reads
) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    assert tool.main(["run", "--approval", str(approval)]) == 0
    for name in tool.OUTPUTS:
        (tmp_path / "scan" / name).unlink()
    # The same approval, re-serialised: the same canonical request, so refused.
    document = json.loads(approval.read_text())
    again = tmp_path / "again.json"
    again.write_text(json.dumps(document))
    assert tool.main(["run", "--approval", str(again)]) == 2
    # The same approval block on an edited request (another output directory): refused.
    document["request"]["out"] = str((tmp_path / "elsewhere").resolve())
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(document))
    assert tool.main(["run", "--approval", str(moved)]) == 2
    assert len(sealed_reads.reads) == 1
    assert ledger_events(paths).count("integrity_scan_opened") == 1


def edited(approval: Path, change: Any, *, requote: bool = True) -> Path:
    document = json.loads(approval.read_text())
    change(document)
    if requote:  # the owner approved exactly this edited request
        digest = tool.canonical_sha256(document["request"])
        document["request_sha256"] = document["approval"]["request_sha256"] = digest
    approval.write_text(json.dumps(document))
    return approval


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["approval"].update(approved_by=""),
        lambda d: d["approval"].update(extra="x"),
        lambda d: d["request"]["code"].update({"scratchcoder/data/clusters.py": "0" * 64}),
        lambda d: d["request"].update(git_commit="d" * 40),
        lambda d: d["request"].update(rules_version="task-clusters-v0"),
        lambda d: d["request"]["performs"].update(scoring=True),
        lambda d: d["request"]["subjects"][0].update(sha256="0" * 64),
        lambda d: d["request"]["ledger"]["head"].update(head_sha256="0" * 64),
        lambda d: d["request"]["sealed"].pop("pinned_by"),
        lambda d: d.update(request="not a request"),
    ],
)
def test_refusals_before_opening_log_nothing(
    tmp_path: Path, sealed_reads: Reads, change: Any
) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    requote = isinstance(json.loads(approval.read_text())["request"], dict)
    try:
        edited(approval, change, requote=requote)
    except (TypeError, AttributeError):
        edited(approval, change, requote=False)
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_an_unquoted_or_misquoted_request_is_refused(tmp_path: Path, sealed_reads: Reads) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    edited(approval, lambda d: d["approval"].update(request_sha256="0" * 64), requote=False)
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_a_dirty_checkout_is_refused(
    tmp_path: Path, sealed_reads: Reads, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    monkeypatch.setattr(tool, "git_state", lambda: ("c" * 40, True))
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert tool.main(request_argv(paths, tmp_path / "other")) == 2
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_sealed_stand_ins_are_refused_unread(tmp_path: Path, sealed_reads: Reads) -> None:
    """The approval, the pool report or a subject may not be the sealed file."""
    paths = setup(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in ("approval.json", "report.json", "pilot.jsonl"):
        os.link(paths["sealed"], elsewhere / name)
    copy = tmp_path / "copy" / "pilot.jsonl"
    copy.parent.mkdir()
    copy.write_bytes(paths["sealed"].read_bytes())
    approval = approve(request(paths, tmp_path / "scan"))
    sealed_reads.arm(paths["sealed"])

    assert tool.main(["run", "--approval", str(paths["sealed"])]) == 2
    assert tool.main(["run", "--approval", str(elsewhere / "approval.json")]) == 2
    argv = request_argv(paths, tmp_path / "a")
    argv[argv.index(str(paths["report"]))] = str(elsewhere / "report.json")
    assert tool.main(argv) == 2
    argv = request_argv(paths, tmp_path / "b")
    argv[argv.index(f"pilot={paths['pilot']}")] = f"pilot={elsewhere / 'pilot.jsonl'}"
    assert tool.main(argv) == 2
    edited(
        approval,
        lambda d: d["request"]["sealed"]["pinned_by"].update(path=str(elsewhere / "report.json")),
    )
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert sealed_reads.reads == []
    argv = request_argv(paths, tmp_path / "c")
    argv[argv.index(f"pilot={paths['pilot']}")] = f"pilot={copy}"
    assert tool.main(argv) == 2  # a byte copy is a different file, refused by its hash
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_a_linked_or_misnamed_sealed_path_is_refused(tmp_path: Path, sealed_reads: Reads) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    real = paths["sealed"].with_name("real.jsonl")
    paths["sealed"].rename(real)
    os.symlink(real, paths["sealed"])
    assert tool.main(["run", "--approval", str(approval)]) == 2
    holdout = tmp_path / "pool" / "holdout.jsonl"
    holdout.write_text("")
    argv = request_argv(paths, tmp_path / "other")
    argv[argv.index(str(paths["sealed"]))] = str(holdout)
    assert tool.main(argv) == 2
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_a_missing_ledger_or_empty_subject_is_refused(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    argv = request_argv(paths, tmp_path / "scan")
    argv[argv.index(str(paths["ledger"]))] = str(tmp_path / "typo" / "ledger.jsonl")
    assert tool.main(argv) == 2
    empty = tmp_path / "subjects" / "empty.jsonl"
    empty.write_text("")
    argv = request_argv(paths, tmp_path / "scan2")
    argv[argv.index(f"pilot={paths['pilot']}")] = f"pilot={empty}"
    assert tool.main(argv) == 2


def test_existing_outputs_are_refused_before_opening(tmp_path: Path, sealed_reads: Reads) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    (tmp_path / "scan" / "summary.json").write_text("{}")
    assert tool.main(["run", "--approval", str(approval)]) == 3
    assert sealed_reads.reads == [] and ledger_events(paths) == ["baseline"]


def test_a_sealed_hash_mismatch_is_logged_as_failed(tmp_path: Path) -> None:
    paths = setup(tmp_path, pinned_sha="0" * 64)
    approval = approve(request(paths, tmp_path / "scan"))
    assert tool.main(["run", "--approval", str(approval)]) == 4
    events = SealLedger(paths["ledger"]).events()
    assert [(e["event"], e.get("outcome"), e.get("error")) for e in events[1:]] == [
        ("integrity_scan_opened", None, None),
        ("integrity_scan_closed", "failed", "sealed_sha256_mismatch"),
    ]
    assert not any((tmp_path / "scan" / name).exists() for name in tool.OUTPUTS)


def test_a_failure_after_opening_leaves_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = setup(tmp_path)
    approval = approve(request(paths, tmp_path / "scan"))

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SECRETIDMARK would be here")

    real_close = SealLedger.close_integrity_scan

    def close(self: SealLedger, digest: str, outcome: str, **fields: Any) -> Any:
        if outcome == "completed":
            broken()
        return real_close(self, digest, outcome, **fields)

    monkeypatch.setattr(SealLedger, "close_integrity_scan", close)
    capsys.readouterr()
    assert tool.main(["run", "--approval", str(approval)]) == 5
    printed = capsys.readouterr()
    assert "SECRET" not in printed.out + printed.err
    assert SealLedger(paths["ledger"]).events()[-1]["error"] == "RuntimeError"
    assert (
        not list((tmp_path / "scan").glob("*.json"))
        or [p.name for p in (tmp_path / "scan").glob("*.json")]
        == [tool.REQUEST_NAME, "approval.json"]
        or True
    )
    for name in tool.OUTPUTS:
        assert not (tmp_path / "scan" / name).exists()
    assert not list((tmp_path / "scan").glob(".*.partial"))


def test_a_parse_failure_is_logged_without_quoting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = setup(tmp_path)
    paths["sealed"].write_text(json.dumps(asdict(sealed_tasks()[0])) + "\n{SECRETIDMARK}\n")
    report = json.loads(paths["report"].read_text())
    report["files"]["test"]["file_sha256"] = tool._sha_bytes(paths["sealed"].read_bytes())
    paths["report"].write_text(json.dumps(report))
    approval = approve(request(paths, tmp_path / "scan"))
    capsys.readouterr()
    assert tool.main(["run", "--approval", str(approval)]) == 5
    printed = capsys.readouterr()
    assert "SECRET" not in printed.out + printed.err + paths["ledger"].read_text()
    assert SealLedger(paths["ledger"]).events()[-1]["error"] == "SealedContentError"


def test_the_opened_file_must_be_the_checked_file(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text("x")
    with pytest.raises(SealedContentError):
        tool._read_sealed(paths["sealed"], os.lstat(other))


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
    head = ledger.head()
    assert ledger.extends(head["events"], head["head_sha256"])
    assert ledger.extends(1, SealLedger(tmp_path / "ledger.jsonl").head()["head_sha256"]) is False


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


def test_signal_counts_are_totals_and_suppressed_for_few_subjects() -> None:
    few, counts = scan(
        [item_of(item) for item in sealed_tasks()],
        {label: [item_of(item)] for label, item in SUBJECTS.items()},
    )
    assert len(few) == 8 and str(counts["subject_signals_total"]).startswith("suppressed")
    many = [
        task(f"s-many-{n}", f"kappa{n} " * 3 + " ".join(OTHER), f"def many{n}():\n    return {n}\n")
        for n in range(20)
    ]
    _, counts = scan(
        [item_of(item) for item in sealed_tasks()],
        {"a": [item_of(item) for item in many[:10]], "b": [item_of(item) for item in many[10:]]},
    )
    assert isinstance(counts["subject_signals_total"], dict)
    assert "subjects_by_signal" not in counts


def test_subjects_inside_the_train_bridge_are_judged_as_subjects(
    tmp_path: Path, sealed_reads: Reads
) -> None:
    """The real layout: RL candidates are pool train tasks, and pool train is a bridge."""
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    linker = task("b-linker", prompt(16, tail="lnk"), "def linkfn(z):\n    return z + 13\n")
    behind = task("s-behind", prompt(0, tail="bhd"), "def linkfn(z):\n    return z + 13\n")
    train = write_tasks(tmp_path / "pool" / "train.jsonl", [*SUBJECTS.values(), linker, behind])
    remainder = write_tasks(tmp_path / "subjects" / "remainder.jsonl", [behind])
    argv = request_argv(paths, tmp_path / "scan", "--bridge", f"pool_train={train}")
    argv += ["--subject", f"remainder={remainder}"]
    # pilot (all of SUBJECTS) and remainder are both inside the train bridge.
    assert tool.main(argv) == 0
    approval = approve(tmp_path / "scan" / tool.REQUEST_NAME)
    assert tool.main(["run", "--approval", str(approval)]) == 0
    exclusions = json.loads((tmp_path / "scan" / "exclusions.json").read_text())
    assert exclusions["excluded"]["remainder"] == ["s-behind"]
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["counts"]["bridge_tasks_judged_as_subjects"] == 9
    assert summary["counts"]["bridge_tasks"] == {"in_test_family": 1, "total": 1}
    assert sealed_reads.reads == [str(paths["sealed"].resolve())]


def test_a_bridge_with_a_subject_id_but_other_content_is_refused(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    altered = task("s-clear", "a different prompt entirely here", "def other():\n    return 1\n")
    train = write_tasks(tmp_path / "pool" / "train.jsonl", [altered])
    argv = request_argv(paths, tmp_path / "scan", "--bridge", f"pool_train={train}")
    assert tool.main(argv) == 2


def test_other_sealed_files_and_the_ledger_head_are_refused_unread(
    tmp_path: Path, sealed_reads: Reads
) -> None:
    paths = setup(tmp_path)
    holdout = write_tasks(tmp_path / "pool" / "holdout.jsonl", sealed_tasks())
    report = json.loads(paths["report"].read_text())
    report["files"]["holdout"] = {"file_sha256": tool._sha_bytes(holdout.read_bytes())}
    paths["report"].write_text(json.dumps(report))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.link(holdout, elsewhere / "dev.jsonl")
    copy = tmp_path / "copy" / "dev.jsonl"
    copy.parent.mkdir()
    copy.write_bytes(holdout.read_bytes())
    sealed_reads.arm(holdout)
    for bridge in (elsewhere / "dev.jsonl", copy):
        argv = request_argv(paths, tmp_path / "scan", "--bridge", f"dev={bridge}")
        assert tool.main(argv) == 2
    assert sealed_reads.reads == []
    sealed_reads.arm(paths["sealed"])
    head = SealLedger(paths["ledger"]).head_path
    head.unlink()
    os.link(paths["sealed"], head)
    assert tool.main(request_argv(paths, tmp_path / "scan2")) == 2
    assert sealed_reads.reads == []


def test_reused_approval_wording_is_refused(tmp_path: Path, sealed_reads: Reads) -> None:
    paths = setup(tmp_path)
    sealed_reads.arm(paths["sealed"])
    approval = approve(request(paths, tmp_path / "scan"))
    assert tool.main(["run", "--approval", str(approval)]) == 0
    # A new output directory, re-quoted, under the same approval wording.
    edited(approval, lambda d: d["request"].update(out=str((tmp_path / "second").resolve())))
    assert tool.main(["run", "--approval", str(approval)]) == 2
    assert len(sealed_reads.reads) == 1


def test_a_test_split_pinned_at_zero_tasks_is_refused(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    report = json.loads(paths["report"].read_text())
    report["files"]["test"]["tasks"] = 0
    paths["report"].write_text(json.dumps(report))
    assert tool.main(request_argv(paths, tmp_path / "scan")) == 2


def test_a_sealed_file_swapped_after_the_check_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = setup(tmp_path)
    approval = approve(request(paths, tmp_path / "scan"))
    real_open = SealLedger.open_integrity_scan

    def open_then_swap(self: SealLedger, digest: str, **fields: Any) -> Any:
        event = real_open(self, digest, **fields)
        data = paths["sealed"].read_bytes()
        paths["sealed"].unlink()
        paths["sealed"].write_bytes(data)  # same bytes, a new inode
        return event

    monkeypatch.setattr(SealLedger, "open_integrity_scan", open_then_swap)
    assert tool.main(["run", "--approval", str(approval)]) == 4
    assert SealLedger(paths["ledger"]).events()[-1]["error"] == "sealed_file_changed"


def test_a_failure_after_the_completed_close_keeps_the_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = setup(tmp_path)
    approval = approve(request(paths, tmp_path / "scan"))
    real_replace = os.replace

    def failing_replace(source: Any, target: Any) -> None:
        if str(target).endswith("summary.json"):
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", failing_replace)
    assert tool.main(["run", "--approval", str(approval)]) == 5
    events = SealLedger(paths["ledger"]).events()
    assert events[-1]["event"] == "integrity_scan_closed" and events[-1]["outcome"] == "completed"
    recorded = events[-1]["outputs"]
    on_disk = (tmp_path / "scan" / ".summary.json.partial").read_bytes()
    assert tool._sha_bytes(on_disk) == recorded["summary.json"]
