"""The read-only reward-signal report: its statistics by hand on an authored screening record,
its forecast model on synthetic counts, and its refusals. No code is executed."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools.rl_reward_signal import (
    ALL_SCOPE,
    EXTRAPOLATION_LABEL,
    FORECAST_LABEL,
    SOURCE_SCOPE,
    analyze,
    beta_binomial_fit,
    extrapolate,
    load_attempts,
    load_screening,
    main,
    mixed_probability,
    modal_off,
    resample_unequal,
    reward_mix,
    split_half,
    wilson,
)


def row(
    task_id: str, source: str, rewards: list[float], baseline: float | None, reason: Any
) -> dict[str, Any]:
    return {
        "id": task_id,
        "source": source,
        "parseable": baseline is not None,
        "baseline": baseline,
        "baseline_answer": None,
        "rewards": rewards,
        "statuses": ["completed"] * len(rewards),
        "audit_flagged": 0,
        "reason": reason,
    }


def record() -> dict[str, Any]:
    """Six candidates of 4 samples: two sources; one unparseable, one paid by a constant,
    one all-zero group and three eligible."""
    tasks = [
        row("a1", "A", [0.0, 0.0, 0.0, 0.5], 0.0, None),
        row("a2", "A", [0.0, 1.0, 0.0, 1.0], 0.25, None),
        row("a3", "A", [0.0, 0.0, 0.0, 0.0], 0.0, "equal_rewards"),
        row("b1", "B", [0.0, 0.2, 0.0, 0.0], 0.2, None),
        row("b2", "B", [0.0, 0.0, 0.0, 0.0], None, "not_parseable"),
        row("b3", "B", [1.0, 1.0, 1.0, 1.0], 0.6, "constant_baseline"),
    ]
    return {
        "format": "rl-screening-v1",
        "rule": {"samples": 4, "threshold": 0.5, "floor": 6, "cap": 1000},
        "outcome": "floor",
        "kept": None,
        "eligible": ["a1", "a2", "b1"],
        "tasks": tasks,
    }


def write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_the_statistics_by_hand() -> None:
    report = analyze(record())
    every = report["sources"]["all"]
    assert (every["candidates"], every["eligible"], every["parseable_below_threshold"]) == (6, 3, 4)
    assert every["eligible_rate"] == 0.5
    assert every["reasons"] == {
        "constant_baseline": 1,
        "equal_rewards": 1,
        "kept": 3,
        "not_parseable": 1,
    }
    mix = every["reward_mix_eligible"]
    assert (mix["zero"], mix["partial"], mix["full"], mix["attempts"]) == (8, 2, 2, 12)
    assert every["eligible_tasks_with_a_full_reward"] == 1
    # a1 has one sample off its mode, a2 two (a tie), b1 one.
    assert every["samples_off_modal_reward"] == {"1": 2, "2": 1}
    # a1 beats its 0.0 constant; a2's best 1.0 beats 0.25; b1's best equals its 0.2.
    assert every["best_sampled_vs_constant_baseline"] == {"below": 0, "equal": 1, "above": 2}
    assert every["attempts_above_own_baseline"] == 3
    assert report["sources"]["A"]["eligible"] == 2 and report["sources"]["B"]["eligible"] == 1
    assert report["notes"]["executed_code"] is False
    assert report["notes"]["forecast"] == FORECAST_LABEL
    # A source's extrapolation is the whole shortfall as if every candidate came from it.
    assert every["extrapolation"]["scope"] == ALL_SCOPE
    assert report["sources"]["A"]["extrapolation"]["scope"] == SOURCE_SCOPE
    assert "as if every new candidate came from it" in SOURCE_SCOPE
    assert "AceCode exclusion stays in force" in report["notes"]["sources"]
    groups = every["informative_fresh_groups"]
    assert 0 < groups["beta_binomial_mean"] < 1 and groups["forecast"] == FORECAST_LABEL
    assert groups["naive_resample_mean"] == pytest.approx(
        (resample_unequal([0, 0, 0, 0.5], 4) + 0.875 + resample_unequal([0, 0.2, 0, 0], 4)) / 3
    )


def test_the_extrapolation_is_labelled_and_bounded_by_the_wilson_interval() -> None:
    found = extrapolate(161, 1000, floor=320)
    assert found["label"] == EXTRAPOLATION_LABEL
    assert "not a requirement" in found["label"] and "not a guarantee" in found["label"]
    assert found["short_of_floor"] == 159
    assert found["more_candidates_at_observed_rate"] == math.ceil(159 / 0.161) == 988
    low, high = found["more_candidates_at_wilson_95"]
    assert low < 988 < high
    assert extrapolate(0, 69, floor=320)["more_candidates"] is None
    assert extrapolate(400, 1000, floor=320)["more_candidates_at_observed_rate"] == 0


def test_wilson_matches_its_formula() -> None:
    low, high = wilson(161, 1000) or (0.0, 0.0)
    assert low == pytest.approx(0.1396, abs=5e-4) and high == pytest.approx(0.1847, abs=5e-4)
    assert wilson(0, 0) is None
    assert wilson(0, 69) is not None and (wilson(0, 69) or (1.0, 1.0))[0] == pytest.approx(0.0)


def test_small_helpers() -> None:
    assert modal_off([0.0, 0.0, 0.1, 0.2]) == 2 and modal_off([1.0] * 8) == 0
    assert reward_mix([])["zero_share"] is None
    assert resample_unequal([0.0] * 8, 8) == 0.0
    assert resample_unequal([0.0, 1.0], 2) == pytest.approx(0.5)
    rows = [{"rewards": [0, 1, 0, 1]}, {"rewards": [0, 1, 0, 0]}, {"rewards": [0, 0, 1, 1]}]
    found = split_half(rows)
    assert (found["unequal_first_half"], found["also_unequal_second_half"]) == (2, 1)
    assert (found["unequal_second_half"], found["also_unequal_first_half"]) == (1, 1)


def test_the_beta_binomial_fit_recovers_its_parameters() -> None:
    """Counts laid out in proportion to the Beta(2, 5)-binomial distribution over 8 trials."""
    a, b, n = 2.0, 5.0, 8

    def pmf(k: int) -> float:
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        top = math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + a + b)
        return math.comb(n, k) * math.exp(top - log_beta)

    counts = [k for k in range(n + 1) for _ in range(round(200_000 * pmf(k)))]
    fit_a, fit_b = beta_binomial_fit(counts, n)
    assert fit_a == pytest.approx(a, rel=0.03) and fit_b == pytest.approx(b, rel=0.03)


def test_the_predictive_mixing_probability() -> None:
    """With a prior concentrated at p the predictive is the binomial's: 1 - (1-p)^n - p^n."""
    p, scale = 0.3, 1e7
    found = mixed_probability(p * scale, (1 - p) * scale, 0, 0, 8)
    assert found == pytest.approx(1 - 0.7**8 - 0.3**8, rel=1e-4)
    assert mixed_probability(1.0, 1.0, 0, 8, 1) == pytest.approx(0.0, abs=1e-12)  # one draw


def test_the_record_is_checked(tmp_path: Path) -> None:
    assert load_screening(write(tmp_path / "ok.json", record()))["outcome"] == "floor"
    bad = record() | {"format": "other"}
    with pytest.raises(ValueError):
        load_screening(write(tmp_path / "format.json", bad))
    short = record()
    short["tasks"][0]["rewards"] = [0.0]
    with pytest.raises(ValueError):
        load_screening(write(tmp_path / "short.json", short))
    stray = record() | {"eligible": ["zz"]}
    with pytest.raises(ValueError):
        load_screening(write(tmp_path / "stray.json", stray))


def attempts_file(path: Path, rows: list[dict[str, Any]]) -> Path:
    header = {"kind": "header", "max_new_tokens": 8, "attempts": len(rows)}
    lines = [json.dumps(header), *(json.dumps(item) for item in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_attempts_are_read_as_data_only(tmp_path: Path) -> None:
    """The program text is only tested for None: a program that would end the interpreter
    if it ran is counted, not run."""
    hostile = "raise SystemExit('this must never run')"
    rows = [
        {"task_id": "a1", "sample": 0, "code": hostile, "completion_tokens": 8},
        {"task_id": "a1", "sample": 1, "code": None, "completion_tokens": 3},
        {"task_id": "b2", "sample": 0, "code": "x = 1", "completion_tokens": 2},
    ]
    header, read = load_attempts(attempts_file(tmp_path / "a.jsonl", rows), task_ids={"a1", "b2"})
    assert [item["has_program"] for item in read] == [True, False, True]
    assert all("code" not in item for item in read)
    report = analyze(record(), attempts=(header, read))["attempts"]
    assert report["executed"] is False
    assert report["by_source"]["all"]["at_token_cap_share"] == pytest.approx(1 / 3)
    assert report["by_source"]["all eligible"]["no_program_share"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        load_attempts(attempts_file(tmp_path / "b.jsonl", rows), task_ids={"a1"})
    (tmp_path / "c.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_attempts(tmp_path / "c.jsonl", task_ids={"a1"})


def test_the_command_never_writes_beside_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    screening = write(runs / "screening.json", record())
    with pytest.raises(SystemExit):
        main(["--screening", str(screening), "--output", str(runs / "report.json")])
    out = tmp_path / "analysis"
    out.mkdir()
    existing = write(out / "exists.json", {})
    with pytest.raises(SystemExit):
        main(["--screening", str(screening), "--output", str(existing)])
    assert main(["--screening", str(screening), "--output", str(out / "report.json")]) == 0
    saved = json.loads((out / "report.json").read_text())
    assert saved["format"] == "rl-reward-signal-v1"
    assert len(saved["inputs"]["screening_sha256"]) == 64
    printed = capsys.readouterr().out
    assert "extrapolation from the observed eligibility rate" in printed
    assert SOURCE_SCOPE in printed and ALL_SCOPE in printed
    assert "AceCode exclusion stays in force" in printed
