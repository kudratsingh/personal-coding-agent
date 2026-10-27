"""Scorecard arithmetic by hand: metrics, slices, baselines, pairs, the leak flag, the CLI."""

import json
from pathlib import Path
from typing import Any

import pytest
from eval_fixtures import (
    TOKENIZER,
    benchmark_tasks,
    pool_task,
    save_task_file,
    write_benchmark_completions,
    write_fixture_completions,
)

from scratchcoder.cli import main
from scratchcoder.eval import protocols
from scratchcoder.eval.baseline import pool_baseline_report, save_baseline
from scratchcoder.eval.benchmarks import score_benchmark
from scratchcoder.eval.choices import ChoiceTask, evaluate_choices
from scratchcoder.eval.completions import Sampling, read_completions
from scratchcoder.eval.executor import execute_completions
from scratchcoder.eval.metrics import paired_bootstrap
from scratchcoder.eval.scorecard import build_scorecard, render_markdown
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result

BASE, SFT, RL = "a" * 64, "b" * 64, "c" * 64
CHECKPOINTS = {"base": BASE, "sft": SFT, "rl": RL}
GREEDY_PASSES = {"base": {"d/1"}, "sft": {"d/1", "d/2", "d/3"}, "rl": {"d/1", "d/2", "d/3"}}
SAMPLED_PASSES = {"base": {"d/1": 1}, "sft": {"d/1": 4, "d/2": 2, "d/3": 1}}
SAMPLED = Sampling(samples=4, temperature=0.8, top_p=0.95)


def sandbox(code: str, tests: str, mode: str) -> Result:
    if "# down" in code:
        return Result("unavailable")
    return Result("passed" if "# good" in code else "failed", elapsed_seconds=30.0)


def dev_tasks() -> list[Task]:
    return [
        pool_task("d/1", source="src-a", written_by="model"),
        pool_task("d/2", source="src-a", written_by="model"),
        pool_task("d/3", source="src-b", written_by="human"),
        pool_task("d/4", source="src-b", written_by="human"),
    ]


def pool_results(root: Path) -> tuple[list[Path], Path]:
    tasks = dev_tasks()
    tasks_path = save_task_file(root / "dev.jsonl", tasks)
    files = []
    for name, sha in CHECKPOINTS.items():
        protocol = "base-pool-completion-v1" if name == "base" else "chat-v1"

        def greedy(task: Task, sample: int, name: str = name) -> str:
            if name == "rl" and task.id == "d/4":
                return "x = 0  # down"
            return "x = 1  # good" if task.id in GREEDY_PASSES[name] else "x = 0"

        runs = [("greedy", Sampling(), greedy)]
        if name in SAMPLED_PASSES:

            def sampled(task: Task, sample: int, name: str = name) -> str:
                return "x = 1  # good" if sample < SAMPLED_PASSES[name].get(task.id, 0) else "x = 0"

            runs.append(("sampled", SAMPLED, sampled))
        for label, sampling, answer in runs:
            completions = root / f"{name}-{label}.jsonl"
            write_fixture_completions(
                completions,
                tasks_path,
                tasks,
                answer,
                sampling=sampling,
                protocol=protocol,
                checkpoint=sha,
            )
            results = root / f"{name}-{label}-results.jsonl"
            execute_completions(
                completions,
                tasks_path,
                results,
                tokenizer=TOKENIZER,
                execute=sandbox,
                sleep=lambda _: None,
            )
            files.append(results)
    baseline = root / "dev-baseline.json"
    save_baseline(baseline, pool_baseline_report(tasks_path, split="dev"))
    return files, baseline


def benchmark_artifacts(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)

    def suite(code: str, tests: str, mode: str) -> Result:
        return Result("passed" if code.rstrip().endswith(("return x", "# good")) else "failed")

    paths = []
    for name, sha, good in (("base", BASE, 1), ("sft", SFT, 0)):
        texts = {
            task.id: "\n    return x  # good\n" if index < good else "\n    return 0\n"
            for index, task in enumerate(tasks)
        }
        completions, _ = write_benchmark_completions(
            root, tasks, texts, checkpoint=sha, name=f"{name}-he.jsonl"
        )
        report = score_benchmark(read_completions(completions), tasks, execute=suite)
        report["request_id"] = "0" * 32
        path = root / f"{name}-he-results.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def arc_inputs(root: Path) -> tuple[Path, list[Path]]:
    tasks = [
        ChoiceTask(
            id=f"arc/{index}",
            question=f"q{index}",
            labels=("A", "B", "C", "D"),
            choices=("w", "x", "y", "z"),
            answer="ABBA"[index],
            source="allenai/ai2_arc",
            source_revision="f" * 40,
            license="CC-BY-SA-4.0",
            split="test",
        )
        for index in range(4)
    ]
    root.mkdir(parents=True, exist_ok=True)
    tasks_path = root / "arc.jsonl"
    tasks_path.write_text(
        "".join(
            json.dumps(
                {**task.__dict__, "labels": list(task.labels), "choices": list(task.choices)}
            )
            + "\n"
            for task in tasks
        )
    )
    reports = []
    for name, sha, picks in (("base", BASE, "AAAA"), ("sft", SFT, "ABBA")):
        chosen = dict(zip((task.public_prompt() for task in tasks), picks, strict=True))

        def score(prompt: str, labels: list[str], chosen: dict[str, str] = chosen) -> list[float]:
            return [1.0 if label.strip() == chosen[prompt] else 0.0 for label in labels]

        report = evaluate_choices(
            tasks, score, checkpoint_hash=sha, tokenizer_hash=TOKENIZER.sha256, bootstrap_samples=10
        )
        path = root / f"{name}-arc.json"
        path.write_text(json.dumps(report))
        reports.append(path)
    return tasks_path, reports


def build(root: Path, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> dict[str, Any]:
    results, baseline = pool_results(root)
    arc_tasks, arc_reports = arc_inputs(root)
    options: dict[str, Any] = {
        "checkpoints": CHECKPOINTS,
        "results": results,
        "benchmarks": benchmark_artifacts(root, monkeypatch),
        "arc_reports": arc_reports,
        "arc_tasks": arc_tasks,
        "pool_baselines": [baseline],
    }
    options.update(changes)
    return build_scorecard(**options)


def row(card: dict[str, Any], checkpoint: str, suite: str, decoding: str) -> dict[str, Any]:
    found = [
        item
        for item in card["rows"]
        if item["checkpoint"] == checkpoint
        and item["suite"] == suite
        and item["decoding"] == decoding
    ]
    assert len(found) == 1
    result: dict[str, Any] = found[0]
    return result


def test_pool_metrics_slices_and_baselines_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = build(tmp_path, monkeypatch)
    sft = row(card, "sft", "pool", "greedy")
    assert sft["metrics"] == {"tasks": 4, "pass_at_1": 0.75}
    assert sft["slices"]["by_source"]["src-a"]["pass_at_1"] == 1.0
    assert sft["slices"]["by_source"]["src-b"]["pass_at_1"] == 0.5
    assert sft["slices"]["by_authorship"]["human"]["pass_at_1"] == 0.5
    sampled = row(card, "sft", "pool", "sampled")
    assert sampled["metrics"]["mean_pass"] == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)
    assert sampled["metrics"]["pass_at_k"] == {
        "1": pytest.approx(0.4375),
        "4": pytest.approx(0.75),
    }
    base = row(card, "base", "pool", "sampled")
    assert base["metrics"]["mean_pass"] == pytest.approx(0.0625)
    assert base["metrics"]["pass_at_k"]["4"] == pytest.approx(0.25)
    # Every row carries the constant baseline beside the score: each fixture task's
    # hidden tests are add_one(2) == 3 and add_one(-1) == 0, so constant 0 earns half.
    for item in card["rows"]:
        if item["suite"] == "pool":
            assert item["constant_baseline"]["mean_reward"] == 0.5
            assert item["constant_baseline"]["full_pass_share"] == 0.0
            assert item["slices"]["by_source"]["src-a"]["constant_baseline_mean_reward"] == 0.5
    for key in ("checkpoint_sha256", "tokenizer_sha256", "task_file_sha256", "protocol_sha256"):
        assert all(len(item[key]) == 64 for item in card["rows"]), key
    assert all(item["precision"] for item in card["rows"])
    assert sft["sandbox_minutes"] == pytest.approx(4 * 30 / 60)
    assert sft["gpu_minutes"] == pytest.approx(1.5 / 60)


def test_unresolved_attempts_are_counted_not_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = build(tmp_path, monkeypatch)
    rl = row(card, "rl", "pool", "greedy")
    assert rl["unresolved_attempts"] == 1 and rl["unresolved_tasks"] == 1
    assert rl["complete"] is False and rl["scored_tasks"] == 3
    assert rl["metrics"] == {"tasks": 3, "pass_at_1": 1.0}
    assert rl["slices"]["by_source"]["src-b"]["unresolved_tasks"] == 1
    rendered = render_markdown(card)
    assert "| rl | greedy | chat-v1 | pass@1 100.00% | 50.00% / 0.00% | 1 |" in rendered


def test_pairs_use_a_fixed_bootstrap_seed_and_label_cross_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = build(tmp_path, monkeypatch)
    by_key = {(item["pair"], item["suite"], item["metric"]): item for item in card["comparisons"]}
    greedy = by_key[("sft vs base", "pool", "pass_at_1")]
    expected = paired_bootstrap([1.0, 0, 0, 0], [1.0, 1, 1, 0], samples=2000, seed=0)
    assert greedy["difference"] == expected.mean == 0.5
    assert (greedy["lower"], greedy["upper"]) == (expected.lower, expected.upper)
    assert greedy["cross_protocol"] is True and greedy["bootstrap_seed"] == 0
    assert by_key[("sft vs base", "pool", "mean_pass")]["cross_protocol"] is True
    assert ("sft vs base", "pool", "pass_at_4") in by_key
    rl = by_key[("rl vs sft", "pool", "pass_at_1")]
    assert rl["tasks"] == 3 and rl["excluded_tasks"] == 1 and rl["cross_protocol"] is False
    assert rl["difference"] == 0.0 and rl["direction"] == "interval includes 0"
    arc = by_key[("sft vs base", "arc-easy", "accuracy")]
    assert arc["first_score"] == 1.0 and arc["second_score"] == 0.5
    assert arc["cross_protocol"] is False
    again = build(tmp_path / "again", monkeypatch)
    assert [item.get("lower") for item in again["comparisons"]] == [
        item.get("lower") for item in card["comparisons"]
    ]
    assert "cross-protocol" in render_markdown(card)


def test_leak_flag_benchmarks_and_arc_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = build(tmp_path, monkeypatch)
    base = row(card, "base", "humaneval", "greedy")
    assert base["metrics"]["pass_at_1"] == pytest.approx(1 / 3) and base["leak_flag"] is True
    assert row(card, "sft", "humaneval", "greedy")["leak_flag"] is False
    assert card["leak"]["flags"] == {"base": True, "sft": False, "rl": False}
    assert card["leak"]["any"] is True
    assert base["constant_baseline"]["available"] is True
    assert base["constant_baseline"]["full_pass_share"] == pytest.approx(0.0)
    arc = row(card, "base", "arc-easy", "label-likelihood")
    assert arc["metrics"]["accuracy"] == 0.5
    assert arc["constant_baseline"]["best_position_accuracy"] == 0.5
    assert arc["constant_baseline"]["chance"] == 0.25
    assert "LEAK FLAG" in render_markdown(card)


def test_inputs_must_be_named_unique_and_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="not named"):
        build(tmp_path / "a", monkeypatch, checkpoints={"base": BASE, "sft": SFT})
    results, _ = pool_results(tmp_path / "b")
    with pytest.raises(ValueError, match="two greedy inputs"):
        build_scorecard(checkpoints=CHECKPOINTS, results=[results[0], results[0]])
    arc_tasks, reports = arc_inputs(tmp_path / "c")
    other = tmp_path / "c" / "other.jsonl"
    other.write_text(arc_tasks.read_text().replace('"q0"', '"changed"'))
    with pytest.raises(ValueError, match="other tasks"):
        build_scorecard(checkpoints=CHECKPOINTS, arc_reports=reports, arc_tasks=other)


def test_scorecard_cli_writes_json_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results, baseline = pool_results(tmp_path)
    output, markdown = tmp_path / "card.json", tmp_path / "card.md"
    arguments = [
        "scorecard",
        *[f"--checkpoint={name}={sha}" for name, sha in CHECKPOINTS.items()],
        *[f"--results={path}" for path in results],
        f"--pool-baseline={baseline}",
        "--pair",
        "sft:base",
        "--output",
        str(output),
        "--markdown",
        str(markdown),
    ]
    assert main(arguments) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == 5
    card = json.loads(output.read_text())
    assert {item["pair"] for item in card["comparisons"]} == {"sft vs base"}
    assert markdown.read_text().startswith("# Evaluation scorecard")
    with pytest.raises(SystemExit):
        main(arguments)  # The outputs exist now.
