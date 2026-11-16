"""Run 1c's approved training rows (``posttrain_approved.RUN1C_TRAIN_ROWS``): the tracked
specification pins them, the bind refuses any other file, and the checker refuses a launch
specification bound to other rows even when every other value is approved.

The rows themselves are private and not in the repository, so these tests build the launch
specification from the pinned values (the estimate and the cost limit derived exactly as the
bind derives them) and bind authored rows only to show that they are refused. Nothing here
touches a network or a checkpoint.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.sft import SFTRow, write_rows
from scratchcoder.data.shards import file_sha256
from tools import posttrain_approved
from tools.posttrain_ops import EstimateLimits, estimate_limit_usd, specification_limits
from tools.posttrain_run import Specification, run1c_placeholders

ROOT = Path(__file__).resolve().parents[1]
SPEC_1C = ROOT / "configs" / "rehearsal-a40-s-post-1c.json"
PINNED = posttrain_approved.RUN1C_TRAIN_ROWS


def launch() -> dict[str, Any]:
    """The launch specification of the pinned rows: the tracked one with the estimate and the
    cost limit filled as ``bind_run1c`` fills them."""
    raw = posttrain_approved.run1c_specification()
    estimate = posttrain_approved.run1c_estimate_usd(raw)
    raw["limits"].update(
        gate_estimate_usd=estimate,
        cost_limit_usd=estimate_limit_usd(posttrain_approved.RUN1C_CEILING_USD, estimate),
    )
    return raw


def authored_rows(path: Path, count: int) -> Path:
    """``count`` authored code rows of 40 token IDs, the second half assistant targets."""
    source = sorted(posttrain_approved.RUN1C_ROW_SOURCES)[0]
    write_rows(
        path,
        [
            SFTRow(
                f"fixture/{index}",
                source,
                tuple(300 + (index + offset) % 500 for offset in range(40)),
                (False,) * 20 + (True,) * 20,
            )
            for index in range(count)
        ],
    )
    return path


def test_the_pin_is_a_well_formed_file_identity_and_one_pass_of_updates() -> None:
    assert set(PINNED) == {"sha256", "size_bytes", "train_rows", "planned_updates"}
    assert re.fullmatch(r"[0-9a-f]{64}", PINNED["sha256"])
    for key in ("size_bytes", "train_rows", "planned_updates"):
        assert type(PINNED[key]) is int and PINNED[key] > 0, key
    sft = posttrain_approved.RUN1C_SFT
    per_update = sft["batch_size"] * sft["accumulation_steps"]
    assert PINNED["planned_updates"] == math.ceil(PINNED["train_rows"] * 1 / per_update)


def test_the_tracked_specification_pins_the_rows_and_leaves_the_estimate_to_the_bind() -> None:
    raw = json.loads(SPEC_1C.read_text())
    assert SPEC_1C.read_bytes() == posttrain_approved.specification_bytes(
        posttrain_approved.run1c_specification()
    )
    assert raw["sft"]["train_rows"] == PINNED["train_rows"]
    assert raw["sft"]["planned_updates"] == PINNED["planned_updates"]
    assert raw["inputs"]["sft-train"] == {
        "kind": "sft-rows",
        "key": None,
        "size_bytes": PINNED["size_bytes"],
        "sha256": PINNED["sha256"],
    }
    assert run1c_placeholders(raw) == ["limits.gate_estimate_usd", "limits.cost_limit_usd"]
    with pytest.raises(ValueError, match="placeholders"):
        posttrain_approved.check_run1c(raw)
    with pytest.raises(ValueError, match="placeholders"):
        Specification.from_dict(raw)


def test_the_launch_specification_of_the_pinned_rows_passes_every_check() -> None:
    raw = launch()
    posttrain_approved.check_run1c(raw)
    spec = Specification.from_dict(raw)
    assert spec.sft_updates == PINNED["planned_updates"]
    derived = specification_limits(raw)
    assert isinstance(derived, EstimateLimits)  # The stop at 120% of the estimate.
    assert derived.limit_usd == raw["limits"]["cost_limit_usd"] <= 1.5
    assert posttrain_approved.gate_launch_fit(raw)["fits"]


def test_the_placeholder_template_is_refused() -> None:
    raw = launch()
    raw["sft"].update(train_rows=None, planned_updates=None)
    raw["inputs"]["sft-train"].update(size_bytes=None, sha256=None)
    raw["limits"].update(gate_estimate_usd=None, cost_limit_usd=None)
    with pytest.raises(ValueError, match="placeholders"):
        posttrain_approved.check_run1c(raw)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("inputs", "sft-train", "sha256"), "ab" * 32, r"inputs\.sft-train\.sha256"),
        (
            ("inputs", "sft-train", "size_bytes"),
            PINNED["size_bytes"] + 1,
            r"inputs\.sft-train\.size_bytes",
        ),
        (("sft", "train_rows"), PINNED["train_rows"] - 1, r"sft\.train_rows"),
    ],
)
def test_a_launch_specification_bound_to_other_rows_is_refused(
    path: tuple[str, ...], value: Any, match: str
) -> None:
    raw = copy.deepcopy(launch())
    target: Any = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        posttrain_approved.check_run1c(raw)


def test_the_bind_refuses_any_file_but_the_pinned_rows(tmp_path: Path) -> None:
    rows = authored_rows(tmp_path / "train.jsonl", 8)
    with pytest.raises(ValueError, match="not run 1c's approved training rows'"):
        posttrain_approved.bind_run1c(rows, expected_sha256=file_sha256(rows))


def test_the_bind_command_refuses_other_rows_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = authored_rows(tmp_path / "train.jsonl", 8)
    output = tmp_path / "launch.json"
    argv = [
        "run1c-bind",
        "--sft-train",
        str(rows),
        "--expected-sha256",
        file_sha256(rows),
        "--output",
        str(output),
    ]
    assert posttrain_approved.main(argv) == 1 and not output.exists()
    assert "approved training rows'" in capsys.readouterr().err
