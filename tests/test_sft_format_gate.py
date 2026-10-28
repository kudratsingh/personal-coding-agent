"""Reference tests of the SFT format gate on tiny authored models (never real pool data).

A tiny model trained on authored chat-v1 code rows passes all four checks. The same model
with the special tokens' embedding rows made identical, the failure run 1's SFT weights
showed, fails them. Each threshold is tested at its edge.
"""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data import chat
from scratchcoder.data.onepass import IGNORE
from scratchcoder.data.sft import SFTRow, write_rows
from scratchcoder.engine.generation import Completion
from scratchcoder.eval import format_gate as gate
from scratchcoder.eval.tasks import Task, load_tasks, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import AdamW, cross_entropy_sum
from scratchcoder.tokenizer.bpe import ByteBPE

TOKENIZER = ByteBPE()  # No merges: every ordinary token is one UTF-8 byte.
CONFIG = ModelConfig(vocab_size=272, width=32, layers=2, heads=2, context=96)


def task(index: int) -> Task:
    return Task(
        id=f"synthetic/{index:02d}",
        prompt=f"Task {index}: write f.",
        public_tests="assert f() == 1",
        hidden_tests="assert f() == 1",
        source="synthetic",
        license="fixture",
        written_by="human",
        split="dev",
        solution="def f():\n    return 1",
    )


TASKS = [task(index) for index in range(40)]
ROWS = [SFTRow.of(item.id, item.source, chat.render_code_task(TOKENIZER, item)) for item in TASKS]


def train(rows: list[SFTRow], steps: int = 80) -> GPT:
    """Assistant-token loss on every row at once; the format is learned in a few steps."""
    torch.manual_seed(0)
    model = GPT(CONFIG)
    width = max(len(row.ids) for row in rows) - 1
    inputs, targets = [], []
    for row in rows:
        row_inputs, row_targets = chat.Rendered(row.ids, row.mask).inputs_targets()
        inputs.append(row_inputs + [chat.BOS] * (width - len(row_inputs)))
        targets.append(row_targets + [IGNORE] * (width - len(row_targets)))
    x, y = torch.tensor(inputs), torch.tensor(targets)
    optimizer = AdamW(model.parameters(), learning_rate=1e-2, weight_decay=0.0)
    for _ in range(steps):
        total, count = cross_entropy_sum(model(x), y)
        optimizer.zero_grad()
        (total / count).backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
        optimizer.step()
    model.eval()
    return model


@pytest.fixture(scope="module")
def trained() -> GPT:
    return train(ROWS)


def collinear(model: GPT, tokens: tuple[int, ...] = gate.CHAT_SPECIALS) -> GPT:
    """The model with the rows of ``tokens`` (default: every chat-v1 special) set to their mean."""
    broken = copy.deepcopy(model)
    with torch.no_grad():
        rows = broken.embedding.weight[list(tokens)]
        broken.embedding.weight[list(tokens)] = rows.mean(dim=0)
    return broken


def test_a_model_that_learned_the_format_passes_every_check(trained: GPT) -> None:
    result = gate.evaluate_gate(trained, TOKENIZER, TOKENIZER, ROWS, TASKS)
    checks = result["checks"]
    assert result["passed"] and result["failed_checks"] == []
    assert checks["first_token"]["rows_argmax_python_start"] == gate.ROWS
    assert checks["first_token"]["mean_p_python_start"] >= gate.MIN_P_PYTHON_START
    assert checks["close"]["mean_p_python_end"] >= gate.MIN_P_PYTHON_END
    assert checks["close"]["mean_p_assistant_end_after_python_end"] >= gate.MIN_P_ASSISTANT_END
    assert checks["separation"]["max_cosine"] < gate.MAX_SPECIAL_COSINE
    assert checks["separation"]["tokens"] == [
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
    ]
    every = checks["separation"]["all_specials"]
    assert every["min_pair_distance"] > 0.1 and len(every["norms"]) == len(gate.CHAT_SPECIALS)
    assert len(every["cosines"]) == len(gate.CHAT_SPECIALS)
    assert checks["greedy_smoke"]["programs"] == gate.SMOKE_PROMPTS
    assert len(result["row_ids"]) == gate.ROWS and len(result["smoke_task_ids"]) == 16
    assert result["rows_matching_task_prompts"] == gate.ROWS
    # Both are chosen by the same key, so the smoke tasks are the first chosen rows.
    assert result["smoke_task_ids"] == result["row_ids"][:16]
    assert all(row["program_sha256"] for row in result["smoke"])


def test_collinear_special_rows_fail_first_token_and_separation(trained: GPT) -> None:
    result = gate.evaluate_gate(collinear(trained), TOKENIZER, TOKENIZER, ROWS, TASKS)
    checks = result["checks"]
    assert not result["passed"]
    assert not checks["first_token"]["passed"] and not checks["separation"]["passed"]
    # Identical rows give identical logits: <|python_start|> can never beat the others.
    assert checks["first_token"]["rows_argmax_python_start"] == 0
    assert checks["first_token"]["mean_p_python_start"] < 1 / len(gate.CHAT_SPECIALS)
    assert checks["separation"]["max_cosine"] == pytest.approx(1.0, abs=1e-6)
    every = checks["separation"]["all_specials"]
    assert every["max_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert every["min_pair_distance"] == pytest.approx(0.0, abs=1e-5)
    assert checks["greedy_smoke"]["programs"] == 0 and not checks["greedy_smoke"]["passed"]


def test_collinear_input_only_specials_are_recorded_but_not_judged(trained: GPT) -> None:
    # Under assistant-only loss <|user_start|> and <|user_end|> are never targets, so
    # nothing in SFT separates them; only the three output specials are judged.
    inputs_only = collinear(trained, (chat.USER_START, chat.USER_END))
    separation = gate.evaluate_gate(inputs_only, TOKENIZER, TOKENIZER, ROWS, TASKS)["checks"][
        "separation"
    ]
    assert separation["passed"] and separation["max_cosine"] < gate.MAX_SPECIAL_COSINE
    assert separation["all_specials"]["max_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert separation["all_specials"]["max_pair"] == ["<|user_start|>", "<|user_end|>"]


def test_rows_are_chosen_by_key_not_by_file_order() -> None:
    forward = gate.select_rows(ROWS, 8, CONFIG.context)
    backward = gate.select_rows(list(reversed(ROWS)), 8, CONFIG.context)
    assert forward == backward
    expected = sorted((row.id for row in ROWS), key=gate.selection_key)
    assert [row.id for row in forward] == expected[:8]
    tasks = gate.select_tasks(list(reversed(TASKS)), 8, TOKENIZER, CONFIG.context)
    assert [item.id for item in tasks] == [row.id for row in forward]


def test_rows_that_do_not_fit_the_context_are_skipped_and_too_few_refused() -> None:
    code = "x = 1\n" * 20
    long = SFTRow.of("long", "synthetic", chat.render_code_task(TOKENIZER, TASKS[0], code))
    assert len(long.ids) - 1 > CONFIG.context
    chosen = gate.select_rows([long, *ROWS[:3]], 3, CONFIG.context)
    assert {row.id for row in chosen} == {row.id for row in ROWS[:3]}
    with pytest.raises(ValueError, match="needs 4 code rows"):
        gate.select_rows([long, *ROWS[:3]], 4, CONFIG.context)


def test_a_row_that_is_not_a_chat_v1_code_row_is_refused() -> None:
    language = chat.render_conversation(
        TOKENIZER, [chat.Message("user", "hi"), chat.Message("assistant", "hello")]
    )
    with pytest.raises(ValueError, match="not a one-exchange chat-v1 code row"):
        gate.select_rows([SFTRow.of("language", "oasst", language), *ROWS], 2, CONFIG.context)


@pytest.mark.parametrize("position", [0, gate.ROWS - 1])
def test_a_row_whose_prompt_differs_from_its_task_is_refused(trained: GPT, position: int) -> None:
    # Position 31 is a measured row that is not a smoke task: every chosen row is compared.
    chosen = gate.select_rows(ROWS, gate.ROWS, CONFIG.context)[position].id
    assert position < 16 or chosen not in [
        item.id for item in gate.select_tasks(TASKS, 16, TOKENIZER, CONFIG.context)
    ]
    edited = replace(next(item for item in TASKS if item.id == chosen), prompt="Other text.")
    rows = [row for row in ROWS if row.id != chosen]
    rows.append(SFTRow.of(chosen, "synthetic", chat.render_code_task(TOKENIZER, edited)))
    with pytest.raises(ValueError, match="render different prompts"):
        gate.evaluate_gate(trained, TOKENIZER, TOKENIZER, rows, TASKS)


def test_a_chosen_row_without_a_task_is_refused(trained: GPT) -> None:
    last = gate.select_rows(ROWS, gate.ROWS, CONFIG.context)[-1].id
    tasks = [item for item in TASKS if item.id != last]
    with pytest.raises(ValueError, match="has no task"):
        gate.evaluate_gate(trained, TOKENIZER, TOKENIZER, ROWS, tasks)


def test_a_code_row_with_an_empty_block_is_refused() -> None:
    row = ROWS[0]
    start = row.ids.index(chat.ASSISTANT_START)
    ids = (*row.ids[: start + 2], chat.PYTHON_END, chat.ASSISTANT_END)  # The renderer refuses this.
    empty = SFTRow("empty", "synthetic", ids, (*row.mask[: start + 2], True, True))
    with pytest.raises(ValueError, match="not a one-exchange chat-v1 code row"):
        gate.assistant_index(empty)
    assert gate.assistant_index(row) == start


def test_first_token_threshold_edges() -> None:
    start, end = chat.PYTHON_START, chat.ASSISTANT_END
    assert gate.judge_first_token([start, start], [0.9, 0.9])["passed"]
    assert not gate.judge_first_token([start, start], [0.9, 0.8999999])["passed"]
    # One row whose argmax is another token fails even with a high mean.
    assert not gate.judge_first_token([start, end], [1.0, 0.95])["passed"]
    assert gate.judge_first_token([start], [1.0])["rows_argmax_python_start"] == 1


def test_close_threshold_edges() -> None:
    assert gate.MIN_P_PYTHON_END == 0.2 and gate.MIN_P_ASSISTANT_END == 0.5
    assert gate.judge_close([0.2, 0.2], [0.5, 0.5])["passed"]
    assert not gate.judge_close([0.2, 0.1999999], [1.0, 1.0])["passed"]
    assert not gate.judge_close([1.0, 1.0], [0.5, 0.4999999])["passed"]
    # The bar is on the mean: one low row is allowed when the others make up for it.
    result = gate.judge_close([0.1, 0.3], [1.0, 0.0])
    assert result["mean_p_python_end"] == 0.2 and result["passed"]
    assert result["min_p_assistant_end_after_python_end"] == 0.0


def test_separation_threshold_edge_is_strict() -> None:
    size = len(gate.OUTPUT_SPECIALS)

    def matrix(value: float) -> list[list[float]]:
        rows = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        rows[1][2] = rows[2][1] = value  # python_start and python_end
        return rows

    assert gate.judge_separation(matrix(0.8999999))["passed"]
    below = gate.judge_separation(matrix(0.9))
    assert not below["passed"]
    assert below["max_pair"] == ["<|python_start|>", "<|python_end|>"]
    with pytest.raises(ValueError):
        gate.judge_separation([[1.0]])
    with pytest.raises(ValueError):  # The seven-token matrix is recorded, not judged.
        gate.judge_separation([[1.0] * 7] * 7)


def test_smoke_threshold_edges() -> None:
    assert gate.judge_smoke([True] * 12 + [False] * 4)["passed"]
    assert not gate.judge_smoke([True] * 11 + [False] * 5)["passed"]
    assert gate.judge_smoke([True] * 12 + [False] * 4)["required"] == 12
    assert gate.judge_smoke([True] * 3 + [False])["required"] == 3


def test_an_empty_python_block_is_not_a_program(
    trained: GPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    def empty_blocks(model: GPT, prompts: list[list[int]], **_: object) -> list[Completion]:
        return [Completion([chat.PYTHON_START, chat.PYTHON_END], "stop", 8) for _ in prompts]

    monkeypatch.setattr(gate, "generate_many", empty_blocks)
    result = gate.greedy_smoke(trained, TOKENIZER, TOKENIZER, TASKS[:2])
    assert [(row["opened"], row["closed"], row["program_extracted"]) for row in result] == [
        (True, True, False)
    ] * 2


def checkpoint(root: Path, model: GPT) -> tuple[Path, Path]:
    tokenizer = root / "tokenizer.json"
    TOKENIZER.save(tokenizer)
    path = root / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(model.config)},
            "tokenizer_hash": TOKENIZER.sha256,
        },
        path,
    )
    return path, tokenizer


def gate_arguments(root: Path, model: GPT) -> list[str]:
    path, tokenizer = checkpoint(root, model)
    write_rows(root / "code_dev.jsonl", ROWS)
    save_tasks(root / "dev.jsonl", TASKS)
    return [
        "sft-format-gate",
        "--checkpoint",
        str(path),
        "--tokenizer",
        str(tokenizer),
        "--rows",
        str(root / "code_dev.jsonl"),
        "--tasks",
        str(root / "dev.jsonl"),
        "--output",
        str(root / "gate.json"),
        "--threads",
        "1",
    ]


def test_cli_writes_the_record_and_exits_zero_on_pass(
    trained: GPT, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = gate_arguments(tmp_path, trained)
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    record = json.loads((tmp_path / "gate.json").read_text())
    assert record["format"] == gate.FORMAT and record["passed"] is True
    assert record["thresholds"] == gate.thresholds()
    inputs = record["inputs"]
    assert inputs["tokenizer_identity"] == TOKENIZER.sha256
    for name in ("checkpoint_sha256", "tokenizer_sha256", "rows_sha256", "tasks_sha256"):
        assert len(inputs[name]) == 64
    assert record["torch_threads"] == 1
    capsys.readouterr()
    assert main(arguments) == 2  # Not measured: an existing record is never replaced.
    assert "not measured: FileExistsError" in capsys.readouterr().err


def test_cli_exits_nonzero_when_the_gate_fails(
    trained: GPT, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(gate_arguments(tmp_path, collinear(trained))) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["passed"] is False and "first_token" in summary["failed_checks"]
    assert json.loads((tmp_path / "gate.json").read_text())["passed"] is False


def test_the_gate_refuses_a_task_file_that_is_not_the_dev_split(
    trained: GPT, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = gate_arguments(tmp_path, trained)
    other = tmp_path / "test.jsonl"
    save_tasks(other, [replace(TASKS[0], split="test")])
    arguments[arguments.index("--tasks") + 1] = str(other)
    # Exit 2, not 1: a gate that could not measure is not a measured failure.
    assert main(arguments) == 2
    assert "only dev tasks" in capsys.readouterr().err
    assert not (tmp_path / "gate.json").exists()


def omitted_rows() -> list[SFTRow]:
    """The dev rows with every third (by position) rendered without its public tests."""
    return [
        SFTRow.of(
            item.id,
            item.source,
            chat.render_code_task(TOKENIZER, item, public_tests=bool(index % 3)),
        )
        for index, item in enumerate(TASKS)
    ]


def test_rows_without_public_tests_are_skipped_not_measured(trained: GPT) -> None:
    rows = omitted_rows()
    without = {row.id for index, row in enumerate(rows) if index % 3 == 0}
    many = TASKS + [task(index) for index in range(40, 60)]
    rows += [
        SFTRow.of(item.id, item.source, chat.render_code_task(TOKENIZER, item))
        for item in many[40:]
    ]
    result = gate.evaluate_gate(trained, TOKENIZER, TOKENIZER, rows, many)
    assert len(result["row_ids"]) == gate.ROWS and not set(result["row_ids"]) & without
    skipped = result["rows_without_public_tests_skipped"]
    assert skipped and set(skipped) <= without
    # Every chosen row is the next with-tests row in key order.
    ordered = sorted((row.id for row in rows), key=gate.selection_key)
    with_tests = [identifier for identifier in ordered if identifier not in without]
    assert result["row_ids"] == with_tests[: gate.ROWS]
    assert set(skipped) == {
        i for i in ordered[: ordered.index(result["row_ids"][-1])] if i in without
    }


def test_too_few_rows_with_public_tests_are_refused(trained: GPT) -> None:
    with pytest.raises(ValueError, match="carry their public tests"):
        gate.evaluate_gate(trained, TOKENIZER, TOKENIZER, omitted_rows(), TASKS)


def test_an_all_with_tests_file_chooses_version_ones_rows(trained: GPT) -> None:
    chosen, skipped = gate.choose_rows(
        ROWS, {item.id: item for item in TASKS}, TOKENIZER, gate.ROWS, CONFIG.context
    )
    assert chosen == gate.select_rows(ROWS, gate.ROWS, CONFIG.context) and skipped == []


def benchmark_tasks() -> dict[str, list[Task]]:
    common = {"source": "fixture", "license": "fixture", "written_by": "human", "split": "eval"}
    humaneval = [
        Task(
            id=f"humaneval/{index}",
            prompt=f"def f{index}():\n",
            public_tests=f"assert callable(f{index})",
            hidden_tests="assert True",
            entry_point=f"f{index}",
            **common,
        )
        for index in range(10)
    ]
    mbpp = [
        Task(
            id=f"mbpp/{index}",
            prompt=f"Task {index}: write f.",
            public_tests="\nassert f() == 1",
            hidden_tests="assert f() == 1",
            **common,
        )
        for index in range(10)
    ]
    return {"humaneval": humaneval, "mbpp-plus": mbpp}


def test_the_benchmark_smoke_is_recorded_and_never_judged(trained: GPT) -> None:
    result = gate.evaluate_gate(
        collinear(trained, (chat.USER_START, chat.USER_END)),
        TOKENIZER,
        TOKENIZER,
        ROWS,
        TASKS,
        benchmarks=benchmark_tasks(),
    )
    recorded = result["benchmark_smoke"]
    assert set(recorded) == {"humaneval", "mbpp-plus"}
    for suite, entry in recorded.items():
        assert entry["judged"] is False and entry["prompts"] == gate.BENCHMARK_SMOKE_PROMPTS
        expected = sorted((t.id for t in benchmark_tasks()[suite]), key=gate.selection_key)
        assert [row["task_id"] for row in entry["results"]] == expected[:8]
    assert "benchmark_smoke" not in result["checks"] and result["passed"]
    # A broken model fails the gate on its four checks, never on the recorded smoke.
    broken = gate.evaluate_gate(
        collinear(trained), TOKENIZER, TOKENIZER, ROWS, TASKS, benchmarks=benchmark_tasks()
    )
    assert set(broken["failed_checks"]) <= {"first_token", "close", "separation", "greedy_smoke"}
    assert broken["benchmark_smoke"]["mbpp-plus"]["programs"] == 0


def test_the_benchmark_smoke_reads_only_humaneval_and_mbpp_plus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="benchmark smoke reads"):
        gate.load_benchmarks({"humaneval-plus": tmp_path / "x"})


def test_cli_records_the_benchmark_smoke(
    trained: GPT,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scratchcoder.eval import protocols

    arguments = gate_arguments(tmp_path, trained)
    files = {}
    for suite, items in benchmark_tasks().items():
        files[suite] = tmp_path / f"{suite}.jsonl"
        save_tasks(files[suite], items)
        arguments += ["--benchmark", f"{suite}={files[suite]}"]
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: list(load_tasks(path)))
    monkeypatch.setattr(protocols, "read_release", lambda path, release: list(load_tasks(path)))
    assert main(arguments) == 0
    record = json.loads((tmp_path / "gate.json").read_text())
    assert record["version"] == gate.VERSION == 2
    assert set(record["benchmark_smoke"]) == {"humaneval", "mbpp-plus"}
    assert set(record["inputs"]["benchmarks_sha256"]) == {"humaneval", "mbpp-plus"}
    capsys.readouterr()
    arguments[-1] = "mbpp-plus"  # No path: refused before anything is measured.
    (tmp_path / "gate.json").unlink()
    assert main(arguments) == 2
    assert "SUITE=PATH" in capsys.readouterr().err
