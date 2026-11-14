"""Executed post-training specifications are frozen; only new specifications take new floors.

Each completed run's report records the SHA-256 of the specification it executed. The tracked
file must keep exactly those bytes, and the code that rebuilds and checks it must reproduce
them, so a change to a shared value (such as the standing speed-floor rule) cannot rewrite a
historical specification after the fact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools import posttrain_approved

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = {
    "1": (
        ROOT / "configs" / "rehearsal-a40-s-post-1.json",
        "146653309e430d3518f932e2a8330487b5ec6ce1ebeeade29212fde5082d3850",
    ),
    "1b": (
        ROOT / "configs" / "rehearsal-a40-s-post-1b.json",
        "69ba9dd6dc63d77814fdcdec02a1044030eb831b2a062464fb546fc7773b1ef8",
    ),
}
"""The ``specification_sha256`` each completed run's verified report records, typed here
independently of the module so that editing one place cannot move both."""
BUILDERS = {
    "1": posttrain_approved.run1_specification,
    "1b": posttrain_approved.run1b_specification,
}
FIRST_LAUNCH_PROBE = {"128": 8_714.860743907404, "256": 9_756.688216709286}


@pytest.mark.parametrize("run", sorted(EXECUTED))
def test_each_executed_specification_keeps_its_executed_bytes(run: str) -> None:
    path, digest = EXECUTED[run]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert posttrain_approved.EXECUTED_SPECIFICATION_SHA256[run] == digest
    rebuilt = posttrain_approved.specification_bytes(BUILDERS[run]())
    assert rebuilt == path.read_bytes()


@pytest.mark.parametrize("run", sorted(EXECUTED))
def test_the_receipt_check_accepts_the_executed_file(run: str) -> None:
    raw = json.loads(EXECUTED[run][0].read_text())
    check = {"1": posttrain_approved.check_run1, "1b": posttrain_approved.check_run1b}[run]
    check(raw)


def _set(raw: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_run_one_is_refused_at_the_standing_rules_current_floors() -> None:
    """The floors the rule derives today are not run 1's: a run-1 specification carrying them
    is not the one that ran."""
    floors = dict(posttrain_approved.GENERATION_FLOORS)
    assert floors != posttrain_approved.RUN1_FLOORS
    raw = json.loads(EXECUTED["1"][0].read_text())
    _set(raw, ("probe", "minimum_tokens_per_second"), floors)
    _set(raw, ("planning", "generation_tokens_per_second"), floors)
    with pytest.raises(ValueError, match="differs from the approved run 1"):
        posttrain_approved.check_run1(raw)


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("PROBE", "minimum_tokens_per_second", {"128": 1.0, "256": 1.0}),
        ("PLANNING", "generation_tokens_per_second", {"128": 1.0, "256": 1.0}),
        ("SFT", "reference_slots_per_second", 1.0),
    ],
)
def test_a_changed_shared_value_cannot_rebuild_an_executed_specification(
    monkeypatch: pytest.MonkeyPatch, section: str, key: str, value: Any
) -> None:
    """Run 1b is built from run 1's values; changing any of them must fail loudly rather than
    quietly produce another specification under the old name."""
    monkeypatch.setitem(getattr(posttrain_approved, section), key, value)
    for build in BUILDERS.values():
        with pytest.raises(ValueError, match="executed specification is frozen"):
            build()


def test_the_writers_reproduce_the_executed_bytes(tmp_path: Path) -> None:
    one, one_b = tmp_path / "1.json", tmp_path / "1b.json"
    posttrain_approved.write_specification(one)
    posttrain_approved.write_specification_1b(one_b)
    assert one.read_bytes() == EXECUTED["1"][0].read_bytes()
    assert one_b.read_bytes() == EXECUTED["1b"][0].read_bytes()


def test_each_executed_run_keeps_the_floors_in_force_when_it_ran() -> None:
    """Run 1 relaunched with floors 5% under its first launch's rates; run 1b ran with the
    standing rule over the three hosts measured before it, and with run 1's speed-check rate
    as its SFT reference."""
    assert posttrain_approved.RUN1_FLOORS == {
        batch: float(math.floor(0.95 * rate)) for batch, rate in FIRST_LAUNCH_PROBE.items()
    }
    assert posttrain_approved.RUN1_SFT_REFERENCE == 56_765.0
    first_three = posttrain_approved.MEASURED_HOST_PROBES[:3]
    assert [label for label, _ in first_three] == [
        "run 1 first launch",
        "run 1",
        "run 1b first launch",
    ]
    assert posttrain_approved.derive_generation_floors(first_three) == (
        posttrain_approved.RUN1B_FLOORS
    )
    assert posttrain_approved.RUN1B_FLOORS == {"128": 7_731.0, "256": 8_733.0}
    assert posttrain_approved.RUN1B_SFT_REFERENCE == posttrain_approved.MEASURED_SFT_HOSTS[0][1]
    one = json.loads(EXECUTED["1"][0].read_text())
    one_b = json.loads(EXECUTED["1b"][0].read_text())
    for raw, floors, reference in (
        (one, posttrain_approved.RUN1_FLOORS, posttrain_approved.RUN1_SFT_REFERENCE),
        (one_b, posttrain_approved.RUN1B_FLOORS, posttrain_approved.RUN1B_SFT_REFERENCE),
    ):
        assert raw["probe"]["minimum_tokens_per_second"] == floors
        assert raw["planning"]["generation_tokens_per_second"] == floors
        assert raw["sft"]["reference_slots_per_second"] == reference


def test_the_standing_rule_derives_10_percent_under_the_hosts_average() -> None:
    """Every speed floor is 10% under the running average of the measured hosts for its batch
    size, rounded down; nothing generates at batch 64, so it has no floor."""
    assert posttrain_approved.derive_generation_floors(
        [("only", {"128": 1_000.0, "256": 2_000.0})]
    ) == {"128": 900.0, "256": 1_800.0}
    hosts = posttrain_approved.MEASURED_HOST_PROBES
    expected = {
        batch: float(math.floor(0.9 * sum(rates[batch] for _, rates in hosts) / len(hosts)))
        for batch in ("128", "256")
    }
    assert posttrain_approved.GENERATION_FLOORS == expected
    sft = posttrain_approved.MEASURED_SFT_HOSTS
    assert posttrain_approved.SFT_REFERENCE_SLOTS_PER_SECOND == pytest.approx(
        sum(rate for _, rate in sft) / len(sft)
    )
