"""Pilot intake: the row offset, and every drop reason in order, on authored fixtures only."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

import tools.pilot_intake as tool
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task, load_tasks
from tools import task_pool

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REVISION = "1" * 40
FILE = "data/oc.parquet"


def write_parquet(path: Path, columns: dict[str, list[Any]], group: int = 700) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=group)
    return path


@pytest.mark.parametrize("start", [0, 1, 511, 512, 513, 1023, 1024, 1499, 1500])
def test_rows_start_at_the_offset_in_file_order(tmp_path: Path, start: int) -> None:
    ids = [f"r{i}" for i in range(1500)]
    path = write_parquet(tmp_path / "rows.parquet", {"id": ids})
    assert [row["id"] for row in task_pool._parquet_rows(path, ["id"], start=start)] == ids[start:]


def test_a_negative_offset_is_refused(tmp_path: Path) -> None:
    path = write_parquet(tmp_path / "rows.parquet", {"id": ["a"]})
    with pytest.raises(ValueError, match="nonnegative"):
        list(task_pool._parquet_rows(path, ["id"], start=-1))


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def assertions_for(name: str) -> list[str]:
    return [f"assert {name}({i}) == {i * 2}" for i in range(3)]


def row(identifier: str, prompt: str, name: str, body: str | None = None) -> dict[str, Any]:
    # A distinct program per row: equal programs would form one task family.
    body = body or f"x * 2 + {sum(map(ord, identifier)) * 1000 + len(identifier)}"
    return {
        "id": identifier,
        "input": prompt,
        "output": f"```python\ndef {name}(x):\n    return {body}\n```",
        "unit_tests": json.dumps(assertions_for(name)),
    }


def pool_task(identifier: str, prompt: str, solution: str, split: str) -> Task:
    return Task(
        id=identifier,
        prompt=prompt,
        public_tests="assert True",
        hidden_tests="assert 1 == 1",
        source="nvidia/OpenCodeInstruct",
        license="CC-BY-4.0",
        written_by="model",
        split=split,
        solution=solution,
    )


ARC_TEXT = words("arcword", 16)
BENCH_PROMPT = words("benchword", 20)
DEV_PROMPT = words("devword", 20)
DEV_TEXT_PROMPT = words("devtext", 20)
TRAIN_PROMPT = words("trainword", 20)
DEV_SOLUTION = (
    "def devfamily(values):\n    total = 0\n    for v in values:\n"
    "        total += v * 9\n    return total\n"
)

# (row, expected reason or "kept"); rows 0-1 precede the range and row 15 follows it.
ROWS: list[tuple[dict[str, Any], str | None]] = [
    (row("before0", words("early", 14), "early0"), None),
    (row("before1", words("early", 15), "early1"), None),
    (row("keep1", words("alpha", 14), "alpha_fn"), "kept"),
    (
        {**row("bad", words("bad", 14), "bad_fn"), "unit_tests": json.dumps(["print(1)"])},
        "unsupported_schema",
    ),
    (
        row("huge", words("huge", 14), "huge_fn", body="x * 2  # " + words("pad", 1000)),
        "over_context",
    ),
    (row("long", words("long", 60), "long_fn"), "over_rl_prompt_tokens"),
    (row("arc", "Solve this " + ARC_TEXT, "arc_fn"), "choice_contamination"),
    (row("bench", BENCH_PROMPT + " now", "bench_fn"), "benchmark_contamination"),
    (row("pooled", words("pooled", 14), "pooled_fn"), "id_in_pool"),
    (row("keep1", words("again", 14), "again_fn"), "duplicate_id"),
    (row("devtext", "Please " + DEV_TEXT_PROMPT, "devtext_fn"), "dev_contamination"),
    (row("near", TRAIN_PROMPT + " ok", "near_fn"), "near_duplicate"),
    (
        row("para", " ".join(DEV_PROMPT.split()[:10]) + " " + words("para", 10), "devname"),
        "dev_paraphrase",
    ),
    (
        {
            **row("fam", words("famword", 14), "famfn"),
            "output": "```python\n" + DEV_SOLUTION.replace("devfamily", "famfn") + "```",
            "unit_tests": json.dumps(
                ["assert famfn([1]) == 9", "assert famfn([]) == 0", "assert famfn([2]) == 18"]
            ),
        },
        "dev_family",
    ),
    (row("keep2", words("omega", 14), "omega_fn"), "kept"),
    (row("after", words("late", 14), "late_fn"), None),
]
START, COUNT = 2, len(ROWS) - 3


class Encoder:
    def __init__(self, tokenizer: object) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[dict[str, Any], str | None]] = ROWS,
) -> list[str]:
    source = write_parquet(
        tmp_path / "raw" / "oc.parquet",
        {key: [entry[key] for entry, _ in rows] for key in ("id", "input", "output", "unit_tests")},
        group=4,
    )
    digest = tool._sha(source)
    spec = {
        "sha256": "s" * 64,
        "context_tokens": 960,
        "verification": {"memory_mb": 128},
        "protection": {"min_alpha_tokens": 3},
        "splits": {"rl_prompt_tokens": 40},
        "tokenizer": {"path": "t", "identity": "i", "file_sha256": "f", "vocab_size": 1},
        "sources": [
            {
                "kind": "opencode",
                "dataset": "nvidia/OpenCodeInstruct",
                "revision": REVISION,
                "files": [FILE],
                "rows": START,
                "license": "CC-BY-4.0",
                "attribution": "NVIDIA",
                "terms": "CC BY 4.0",
                "terms_reviewed_on": "2026-10-25",
                "obligations": "credit",
            }
        ],
    }
    receipts = {
        "status": "completed",
        "spec_sha256": "s" * 64,
        "files": [
            {
                "kind": "opencode",
                "filename": FILE,
                "path": str(source),
                "sha256": digest,
                "size_bytes": source.stat().st_size,
                "url": f"https://huggingface.co/datasets/nvidia/OpenCodeInstruct/resolve/{REVISION}/{FILE}",
            }
        ],
    }
    (tmp_path / "spec.json").write_text("{}")
    (tmp_path / "receipts.json").write_text(json.dumps(receipts))
    benchmark = pool_task("HumanEval/0", BENCH_PROMPT, "def b(x):\n    return x\n", "test")
    question = ChoiceTask(
        id="arc-1",
        question=ARC_TEXT,
        labels=("A", "B"),
        choices=("yes", "no"),
        answer="A",
        source="allenai/ai2_arc",
        source_revision="2" * 40,
        license="CC-BY-SA-4.0",
    )
    monkeypatch.setattr(tool, "load_spec", lambda path: spec)
    monkeypatch.setattr(tool, "_load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(tool, "FastEncoder", Encoder)
    monkeypatch.setattr(tool, "humaneval_tasks", lambda path: [benchmark])
    monkeypatch.setattr(tool, "read_release", lambda path, spec: [])
    monkeypatch.setattr(tool, "arc_questions", lambda path, record: [question])
    train = [
        pool_task(
            "nvidia/OpenCodeInstruct/pooled",
            words("other", 14),
            "def p(x):\n    return 1\n",
            "train",
        ),
        pool_task(
            "nvidia/OpenCodeInstruct/t1", TRAIN_PROMPT, "def t1(x):\n    return 3\n", "train"
        ),
        pool_task(
            "nvidia/OpenCodeInstruct/r1",
            words("remainder", 14),
            "def r1(x):\n    return 4\n",
            "train",
        ),
    ]
    dev = [
        pool_task(
            "nvidia/OpenCodeInstruct/d1", DEV_PROMPT, "def devname(x):\n    return 5\n", "dev"
        ),
        pool_task(
            "nvidia/OpenCodeInstruct/d2", DEV_TEXT_PROMPT, "def d2(x):\n    return 6\n", "dev"
        ),
        pool_task("nvidia/OpenCodeInstruct/d3", words("devfam", 14), DEV_SOLUTION, "dev"),
    ]
    for name, tasks in (("train.jsonl", train), ("dev.jsonl", dev)):
        (tmp_path / "pool").mkdir(exist_ok=True)
        (tmp_path / "pool" / name).write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))
    (tmp_path / "remainder.json").write_text(
        json.dumps({"admitted": ["nvidia/OpenCodeInstruct/r1"]})
    )
    for name in ("HumanEval.jsonl.gz", "plus.jsonl.gz", "mbpp.jsonl.gz", "arc.jsonl", "arc.json"):
        (tmp_path / "bench").mkdir(exist_ok=True)
        (tmp_path / "bench" / name).write_text("x")
    return [
        "--spec", str(tmp_path / "spec.json"),
        "--receipts", str(tmp_path / "receipts.json"),
        "--pool-train", str(tmp_path / "pool" / "train.jsonl"),
        "--pool-dev", str(tmp_path / "pool" / "dev.jsonl"),
        "--humaneval", str(tmp_path / "bench" / "HumanEval.jsonl.gz"),
        "--humaneval-plus", str(tmp_path / "bench" / "plus.jsonl.gz"),
        "--mbpp-plus", str(tmp_path / "bench" / "mbpp.jsonl.gz"),
        "--arc-export", str(tmp_path / "bench" / "arc.jsonl"),
        "--arc-record", str(tmp_path / "bench" / "arc.json"),
        "--file", FILE,
        "--start-row", str(START),
        "--rows", str(len(rows) - 3),
        "--remainder", str(tmp_path / "remainder.json"),
        "--out", str(tmp_path / "out"),
    ]  # fmt: skip


def test_every_reason_fires_in_order_and_only_the_range_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = fixture(tmp_path, monkeypatch)
    assert tool.main(argv) == 0
    out = tmp_path / "out"
    kept = list(load_tasks(out / "candidates.jsonl"))
    assert [task.id for task in kept] == [
        "nvidia/OpenCodeInstruct/keep1",
        "nvidia/OpenCodeInstruct/keep2",
    ]
    assert all(task.split == "train" and task.solution for task in kept)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    expected = [
        (START + offset, reason) for offset, (_, reason) in enumerate(ROWS[START : START + COUNT])
    ]
    assert [(entry["row"], entry["reason"]) for entry in rows] == expected
    record = json.loads((out / "record.json").read_text())
    counts = record["counts"]
    assert counts["rows"] == COUNT and counts["kept"] == 2
    assert sum(counts[reason] for reason in tool.REASONS) + counts["kept"] == COUNT
    assert all(counts[reason] == 1 for reason in tool.REASONS if reason != "oversized_for_sandbox")
    assert record["source"]["start_row"] == START and record["source"]["rows"] == COUNT
    assert record["executed_code"] is False and record["sealed_files_opened"] is False
    remainder = list(load_tasks(out / "remainder.jsonl"))
    assert [task.id for task in remainder] == ["nvidia/OpenCodeInstruct/r1"]
    assert set(record["outputs"]) == {
        "candidates.jsonl",
        "rows.jsonl",
        "remainder.jsonl",
        "scan-bridges.jsonl",
    }
    # Every adapted row that was dropped is a scan bridge, in row order, as a train task;
    # the row the adapter refused has no task to link.
    bridges = list(load_tasks(out / "scan-bridges.jsonl"))
    adapted_drops = [
        f"nvidia/OpenCodeInstruct/{entry['id']}"
        for entry, reason in ROWS[START : START + COUNT]
        if reason not in ("kept", "unsupported_schema")
    ]
    assert [task.id for task in bridges] == adapted_drops
    assert all(task.split == "train" for task in bridges)
    # Two of them repeat an ID (a pool task's and a candidate's) with other content.
    assert record["scan_bridges"] == {
        "tasks": len(adapted_drops),
        "identical_repeats_omitted": 0,
        "repeated_ids": 2,
    }


def test_a_range_past_the_end_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = fixture(tmp_path, monkeypatch)
    argv[argv.index("--rows") + 1] = str(len(ROWS))
    assert tool.main(argv) == 2


def test_existing_outputs_and_sealed_inputs_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = fixture(tmp_path, monkeypatch)
    sealed = tmp_path / "pool" / "test.jsonl"
    sealed.write_text("")
    refused = list(argv)
    refused[refused.index("--pool-dev") + 1] = str(sealed)
    assert tool.main(refused) == 2
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "rows.jsonl").write_text("")
    assert tool.main(argv) == 3


def test_a_source_file_that_differs_from_its_receipt_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = fixture(tmp_path, monkeypatch)
    receipts = json.loads((tmp_path / "receipts.json").read_text())
    receipts["files"][0]["sha256"] = "0" * 64
    (tmp_path / "receipts.json").write_text(json.dumps(receipts))
    assert tool.main(argv) == 2


def test_a_dropped_row_still_bridges_its_family_to_dev() -> None:
    """A paraphrase of a dev prompt is dropped, but a candidate with the same program is in
    that dev task's family and must be dropped too (the reviewer's reproduction)."""
    solution = "def zeta_fn(x):\n    return x * 7 + 1\n"
    dev = [pool_task("d1", DEV_PROMPT, "def devname(x):\n    return 5\n", "dev")]
    paraphrase = pool_task("ca", DEV_PROMPT.replace("devword12", "zz1"), solution, "train")
    candidate = pool_task("cb", words("beta", 14), solution, "train")
    assert tool.dev_family_ids([candidate], dev, []) == set()
    assert tool.dev_family_ids([candidate], dev, [], [paraphrase]) == {"cb"}


def test_a_bridge_row_repeating_a_pool_id_is_joined_to_it() -> None:
    linked = pool_task("t1", words("trainonly", 14), "def tq(x):\n    return x - 3\n", "train")
    dev = [pool_task("d1", DEV_PROMPT, "def tq(x):\n    return x - 3\n", "dev")]
    # The range row repeats train task t1's ID with other content; t1 shares dev's program.
    repeat = pool_task("t1", words("repeat", 14), "def other(x):\n    return x // 4\n", "train")
    candidate = pool_task("cc", words("gamma", 14), "def other(x):\n    return x // 4\n", "train")
    assert tool.dev_family_ids([candidate], dev, [linked], [repeat]) == {"cc"}


def test_rows_the_pool_build_read_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = fixture(tmp_path, monkeypatch)
    spec = tool.load_spec(Path("unused"))  # type: ignore[attr-defined]
    spec["sources"][0]["rows"] = START + 1  # the build read rows 0..START
    assert tool.main(argv) == 2
    spec["sources"][0]["rows"] = START
    assert tool.main(argv) == 0


def arc_record(tmp_path: Path, path: Path, **changes: Any) -> Path:
    from scratchcoder.data.arc import arc_splits

    record = {
        "kind": "pod-task-export",
        "export": "arc",
        "output": {"path": str(path), "bytes": path.stat().st_size, "sha256": tool._sha(path)},
        "questions": arc_splits()["test"].task_count,
        "source": {"manifests": {"test": asdict(arc_splits()["test"].manifest)}},
    }
    for key, value in changes.items():
        record[key] = value
    target = tmp_path / "arc-export.json"
    target.write_text(json.dumps(record))
    return target


def test_the_arc_export_must_match_its_record_and_the_reviewed_split(tmp_path: Path) -> None:
    export = tmp_path / "arc-easy-test.jsonl"
    export.write_text('{"id": "x"}\n')
    other = tmp_path / "other.jsonl"
    other.write_text('{"id": "y"}\n')
    with pytest.raises(tool.Refused, match="file its record names"):
        tool.arc_questions(other, arc_record(tmp_path, export))
    with pytest.raises(tool.Refused, match="reviewed ARC-Easy"):
        tool.arc_questions(export, arc_record(tmp_path, export, questions=1))
    record = json.loads(arc_record(tmp_path, export).read_text())
    record["output"]["bytes"] += 1
    (tmp_path / "arc-export.json").write_text(json.dumps(record))
    with pytest.raises(tool.Refused, match="size"):
        tool.arc_questions(export, tmp_path / "arc-export.json")
    record["output"]["bytes"] -= 1
    record["output"]["sha256"] = "0" * 64
    (tmp_path / "arc-export.json").write_text(json.dumps(record))
    with pytest.raises(tool.Refused, match="SHA-256"):
        tool.arc_questions(export, tmp_path / "arc-export.json")


def test_scan_bridges_omit_only_identical_repeats() -> None:
    held = pool_task("p1", words("held", 14), "def h(x):\n    return 1\n", "dev")
    same = replace(held, split="train", public_tests="assert 2 == 2")  # same family fields
    other = replace(held, prompt=words("changed", 14))
    first = pool_task("n1", words("new", 14), "def n(x):\n    return 2\n", "dev")
    again = replace(first, hidden_tests="assert 3 == 3")
    linked, counts = tool.scan_bridges([same, other, first, again], [held])
    assert [(task.id, task.prompt) for task in linked] == [
        ("p1", other.prompt),
        ("n1", first.prompt),
    ]
    assert all(task.split == "train" for task in linked)
    assert counts == {"tasks": 2, "identical_repeats_omitted": 2, "repeated_ids": 1}


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": words("changed", 14)},
        {"solution": "def h(x):\n    return 99\n"},
        {"entry_point": "h"},
        {"source": "other/source"},
        {
            "mode": "io",
            "solution": "print(input())\n",
            "public_tests": json.dumps([{"input": "1", "output": "1"}]),
            "hidden_tests": json.dumps([{"input": "2", "output": "2"}]),
        },
    ],
)
def test_a_repeat_differing_in_any_family_field_is_kept_and_counted(
    change: dict[str, Any],
) -> None:
    held = pool_task("p1", words("held", 14), "def h(x):\n    return 1\n", "dev")
    changed = replace(held, **change)
    linked, counts = tool.scan_bridges([changed], [held])
    assert [task.id for task in linked] == ["p1"]
    assert counts == {"tasks": 1, "identical_repeats_omitted": 0, "repeated_ids": 1}


def test_scan_bridges_keep_row_order_across_the_family_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev-family drop is decided last, but its row comes before a later drop."""
    family_row = next(entry for entry in ROWS if entry[1] == "dev_family")
    long_row = next(entry for entry in ROWS if entry[1] == "over_rl_prompt_tokens")
    rows = [*ROWS[:START], family_row, long_row, ROWS[-1]]
    assert tool.main(fixture(tmp_path, monkeypatch, rows)) == 0
    bridges = list(load_tasks(tmp_path / "out" / "scan-bridges.jsonl"))
    assert [task.id for task in bridges] == [
        "nvidia/OpenCodeInstruct/fam",
        "nvidia/OpenCodeInstruct/long",
    ]


# The integrity scan over the intake's outputs --------------------------------------------

SECRET_PROMPT = words("sealedword", 20) + " SEALEDPROMPTMARK"
SECRET_SOLUTION = (
    "def sealedfnmark(values):\n    out = []\n    for v in values:\n"
    "        out.append(v * 13 - 1)\n    return out\n"
)
CHAIN_PROMPT = words("chainword", 20)
CHAIN_ROWS: list[tuple[dict[str, Any], str | None]] = [
    *ROWS[:START],
    (row("survivor", CHAIN_PROMPT, "survivor_fn"), "kept"),
    (
        {
            **row("dropped", CHAIN_PROMPT + " again", "sealedfnmark"),
            "output": "```python\n" + SECRET_SOLUTION + "```",
            "unit_tests": json.dumps(
                [
                    "assert sealedfnmark([1]) == [12]",
                    "assert sealedfnmark([]) == []",
                    "assert sealedfnmark([2]) == [25]",
                ]
            ),
        },
        "near_duplicate",
    ),
    ROWS[-1],
]


def sealed_fixture(root: Path) -> tuple[Path, Path, Path]:
    from scratchcoder.eval.seal import SealLedger
    from tools import holdout_integrity_scan as scan_tool

    sealed = root / "sealed" / "test.jsonl"
    sealed.parent.mkdir(parents=True)
    test_task = replace(
        pool_task("SEALEDIDMARK", SECRET_PROMPT, SECRET_SOLUTION, "test"),
        hidden_tests="assert SEALEDTESTMARK == 1",
    )
    sealed.write_text(json.dumps(asdict(test_task)) + "\n")
    report = root / "sealed" / "report.json"
    digest = scan_tool._sha_bytes(sealed.read_bytes())
    report.write_text(json.dumps({"files": {"test": {"file_sha256": digest, "tasks": 1}}}))
    ledger = root / "seal" / "ledger.jsonl"
    SealLedger(ledger).record_baseline(
        task_file_sha256="a" * 64, path=Path("x"), sha256="b" * 64, results_id="r"
    )
    return sealed, report, ledger


def test_a_candidate_linked_to_a_test_task_only_through_a_dropped_row_is_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Surviving candidate -> dropped intake row -> sealed test task: the dropped row is a
    scan bridge, so the candidate is excluded, and no output names the row or the test."""
    from scratchcoder.data.integrity_scan import item_of, scan
    from tools import holdout_integrity_scan as scan_tool

    argv = fixture(tmp_path, monkeypatch, CHAIN_ROWS)
    assert tool.main(argv) == 0
    out = tmp_path / "out"
    survivor, dropped = "nvidia/OpenCodeInstruct/survivor", "nvidia/OpenCodeInstruct/dropped"
    assert [task.id for task in load_tasks(out / "candidates.jsonl")] == [survivor]
    assert [task.id for task in load_tasks(out / "scan-bridges.jsonl")] == [dropped]
    sealed, report, ledger = sealed_fixture(tmp_path)
    test_items = [item_of(task) for task in load_tasks(sealed)]
    pilot = [item_of(task) for task in load_tasks(out / "candidates.jsonl")]
    linked = [item_of(task) for task in load_tasks(out / "scan-bridges.jsonl")]
    # Without the dropped row the candidate looks clear; with it, it is in the family.
    assert [r.status for r in scan(test_items, {"pilot": pilot})[0]] == ["clear"]
    with_range = scan(test_items, {"pilot": pilot}, bridges={"range_dropped": linked})[0]
    assert [(r.id, r.status) for r in with_range] == [(survivor, "excluded")]

    # The command, end to end: request (nothing sealed opened), approval, one logged scan.
    monkeypatch.setattr(scan_tool, "git_state", lambda: ("c" * 40, False))
    scan_out = tmp_path / "scan"
    request_argv = [
        "request", "--sealed", str(sealed), "--pool-report", str(report),
        "--subject", f"pilot={out / 'candidates.jsonl'}",
        "--subject", f"remainder={out / 'remainder.jsonl'}",
        "--bridge", f"pool_train={tmp_path / 'pool' / 'train.jsonl'}",
        "--bridge", f"pool_dev={tmp_path / 'pool' / 'dev.jsonl'}",
        "--bridge", f"range_dropped={out / 'scan-bridges.jsonl'}",
        "--ledger", str(ledger), "--out", str(scan_out),
    ]  # fmt: skip
    assert scan_tool.main(request_argv) == 0
    document = json.loads((scan_out / scan_tool.REQUEST_NAME).read_text())
    document["approval"] = {
        "approved_by": "owner",
        "approved_at": "2026-11-15T00:00:00Z",
        "statement": "approved for this synthetic test",
        "request_sha256": document["request_sha256"],
    }
    approval = scan_out / "approval.json"
    approval.write_text(json.dumps(document))
    capsys.readouterr()
    assert scan_tool.main(["run", "--approval", str(approval)]) == 0
    printed = capsys.readouterr()
    exclusions = json.loads((scan_out / "exclusions.json").read_text())
    assert exclusions["excluded"] == {"pilot": [survivor], "remainder": []}
    assert exclusions["clear"] == {"pilot": [], "remainder": ["nvidia/OpenCodeInstruct/r1"]}
    summary = json.loads((scan_out / "summary.json").read_text())
    assert summary["counts"] == {
        "subjects": {
            "pilot": {"excluded": 1, "flagged": 0, "clear": 0, "total": 1},
            "remainder": {"excluded": 0, "flagged": 0, "clear": 1, "total": 1},
        }
    }
    texts = [printed.out, printed.err, ledger.read_text()]
    texts += [(scan_out / name).read_text() for name in scan_tool.OUTPUTS]
    for text in texts:
        for marker in ("SEALEDIDMARK", "SEALEDPROMPTMARK", "SEALEDTESTMARK", "sealedfnmark"):
            assert marker not in text, marker
        assert dropped not in text and "chainword" not in text


def test_a_range_bridge_repeating_a_pool_id_is_refused_by_the_scan_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repeated ID with other content cannot be scanned silently: the request refuses."""
    from tools import holdout_integrity_scan as scan_tool

    argv = fixture(tmp_path, monkeypatch)
    assert tool.main(argv) == 0
    sealed, report, ledger = sealed_fixture(tmp_path)
    monkeypatch.setattr(scan_tool, "git_state", lambda: ("c" * 40, False))
    request_argv = [
        "request", "--sealed", str(sealed), "--pool-report", str(report),
        "--subject", f"pilot={tmp_path / 'out' / 'candidates.jsonl'}",
        "--bridge", f"pool_train={tmp_path / 'pool' / 'train.jsonl'}",
        "--bridge", f"range_dropped={tmp_path / 'out' / 'scan-bridges.jsonl'}",
        "--ledger", str(ledger), "--out", str(tmp_path / "scan"),
    ]  # fmt: skip
    capsys.readouterr()
    assert scan_tool.main(request_argv) == 2
    assert "bridge IDs must be unique" in capsys.readouterr().err
    assert not (tmp_path / "scan" / scan_tool.REQUEST_NAME).exists()
