"""Run 2's wait bounds and liveness rules outside the worker: the derived timeouts, the Mac
watchdog's bounded ``awaiting_rewards`` extension, the policy that carries it, the cost margin
for one full wait, and the in-Pod idle rule with a fresh and a stale Mac lease. Offline; no
Runpod or R2 call is made."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from test_pod_self_destruct import Clock, FakeTransport, timer

from scratchcoder.rl.bridge_mac import LEASE_SECONDS
from scratchcoder.rl.rewards import DEFAULT_HOST_RETRIES, DEFAULT_WORKERS
from scratchcoder.sandbox.budget import io_seconds
from scratchcoder.sandbox.reward import HOST_SLACK_SECONDS, MAX_IO_CASES
from tools import posttrain_approved, posttrain_ops, runpod_watchdog
from tools.posttrain_run import Specification, check_policy
from tools.runpod_watchdog import Policy, check_once, read_heartbeat

TEMPLATE = posttrain_approved.SPECIFICATION_PATH_2
RUNBOOK_1 = Path(__file__).resolve().parents[1] / "POSTTRAIN_RUNBOOK.md"


def gated() -> dict[str, Any]:
    """The run-2 template with gate values filled by placeholders (no real pin)."""
    pin = {"size_bytes": 10, "sha256": "a" * 64}
    return posttrain_approved.run2_specification(
        {
            "ceiling_usd": 5.80,
            "sft-weights": {"key": "posttrain-runs/1/podpost01/sft-weights/checkpoint.pt", **pin},
            "rl-tasks": dict(pin),
            "rl-dev": dict(pin),
            "rl-facts": dict(pin),
            "pool-test": dict(pin),
        }
    )


# The derived timeouts -------------------------------------------------------------------------


def test_every_wait_bound_derives_from_the_budgets_the_workers_and_the_tries() -> None:
    bounds = posttrain_ops.bridge_timeouts(attempts_per_request=32 * 8, dev_programs=200 * 9)
    # One try: the channel's longest budget (a 100-case I/O task) plus its host slack.
    assert io_seconds(MAX_IO_CASES) == 27.0 and HOST_SLACK_SECONDS == 15.0
    assert bounds.attempt_try_seconds == 27.0 + 15.0 == 42.0
    # One attempt: 4 tries (3 retries) and the 1, 2 and 4 s waits between them.
    assert DEFAULT_HOST_RETRIES == 3 and DEFAULT_WORKERS == 8
    assert bounds.reward_attempt_seconds == 4 * 42 + 1 + 2 + 4 == 175.0
    # The Mac's bounds: 32 rounds of an attempt; 225 rounds of a try and one more running.
    assert bounds.mac_reward_timeout_seconds == math.ceil(256 / 8) * 175 == 5_600.0
    assert bounds.mac_dev_timeout_seconds == math.ceil(1_800 / 8) * 42 + 42 == 9_492.0
    # The bridge: one Mac poll, list/read/write each 5 tries to its timeout, the backoff, a
    # Pod poll.
    assert bounds.bridge_allowance_seconds == 2 + 5 * (30 + 120 + 120) + 3 * 15 + 0.5 == 1_397.5
    # The Mac's checks before it scores a request: one window pass, every call to its bound
    # (11 container-engine calls of 60 s, two ssh-add of 30 s, one launchctl of 10 s), and the
    # image check (4 tries of 30 s and the 1, 2 and 4 s waits). test_rl_bridge_window.py ties
    # each term to the bridge's own call bounds.
    assert bounds.window_check_seconds == 11 * 60 + 2 * 30 + 10 == 730
    assert bounds.image_check_seconds == 4 * 30 + 1 + 2 + 4 == 127.0
    assert bounds.mac_check_seconds == 857.0
    assert bounds.reward_timeout_seconds == 5_600 + 1_397.5 + 857 == 7_854.5
    assert bounds.dev_timeout_seconds == 9_492 + 1_397.5 + 857 == 11_746.5
    # The Mac's --reward-timeout and --dev-timeout are the Pod's less the allowance and the
    # checks: the Mac's bound starts when the checks have passed.
    for kind, mac in (("reward", 5_600.0), ("dev", 9_492.0)):
        assert bounds.timeout(kind) - bounds.bridge_allowance_seconds - 857 == mac
        assert bounds.mac_timeout(kind) == mac
    assert bounds.awaiting_reward_extension_seconds == 7_854.5 + 60 == 7_914.5
    assert bounds.awaiting_dev_extension_seconds == 11_746.5 + 60 == 11_806.5
    assert bounds.wait_margin_seconds == 11_746.5
    assert bounds.lease_seconds == LEASE_SECONDS == 60.0
    assert bounds.lost_bridge_seconds == 900 - 300 == 600.0
    with pytest.raises(ValueError):
        bounds.timeout("other")
    with pytest.raises(ValueError):
        bounds.mac_timeout("other")


def test_the_mac_reward_bound_is_run_ones_screening_timeout() -> None:
    """Run 1's runbook scores 256-attempt screening batches under the same worst case."""
    bounds = posttrain_ops.bridge_timeouts(attempts_per_request=256, dev_programs=1_800)
    assert "--timeout 5600" in RUNBOOK_1.read_text()
    assert bounds.mac_reward_timeout_seconds == 5_600.0


def test_the_derivation_refuses_impossible_inputs() -> None:
    for options in (
        {"attempts_per_request": 0, "dev_programs": 1},
        {"attempts_per_request": 1, "dev_programs": 1, "workers": 0},
        {"attempts_per_request": 1, "dev_programs": 1, "ssh_retries": 5},
        {"attempts_per_request": 1, "dev_programs": 1, "pod_idle_seconds": 300},
        {"attempts_per_request": 1, "dev_programs": 1, "mac_poll_seconds": 0},
    ):
        with pytest.raises(ValueError):
            posttrain_ops.bridge_timeouts(**options)


def test_the_specification_writes_the_derivation_down_and_refuses_another_value() -> None:
    raw = json.loads(TEMPLATE.read_text())
    bridge = raw["bridge"]
    assert (bridge["reward_timeout_seconds"], bridge["dev_timeout_seconds"]) == (7_854.5, 11_746.5)
    assert (bridge["mac_reward_timeout_seconds"], bridge["mac_dev_timeout_seconds"]) == (
        5_600.0,
        9_492.0,
    )
    derived = posttrain_ops.specification_bridge_timeouts(raw)
    assert derived is not None and derived.wait_margin_seconds == bridge["wait_margin_seconds"]
    for key in posttrain_ops.DERIVED_BRIDGE_KEYS:
        changed = json.loads(TEMPLATE.read_text())
        changed["bridge"][key] = 420.0  # The proposal's flat value.
        with pytest.raises(ValueError, match="differ from their derivation"):
            posttrain_ops.specification_bridge_timeouts(changed)
    assert posttrain_ops.specification_bridge_timeouts({"run": "1"}) is None


# The watchdog's bounded extension ------------------------------------------------------------


def run2_policy(**changes: Any) -> Policy:
    spec = Specification.from_dict(gated())
    written = posttrain_ops.watchdog_policy(
        spec.ceiling, origin=1_000.0, idle_baseline=1_000.0, bridge=spec.bridge_timeouts
    )
    return Policy(**{**written, **changes})


def beat(path: Path, *, now: float, activity: float, kind: str | None, status: str) -> None:
    value: dict[str, Any] = {
        "schema_version": 1,
        "phase": "rl",
        "step": 4,
        "status": status,
        "last_gpu_activity_unix": activity,
        "updated_at_unix": now,
        "pod_id": "podpost01",
    }
    if kind is not None:
        value["awaiting"] = {"request_id": "x", "kind": kind, "index": 3, "count": 256}
    path.write_text(json.dumps(value))


class Api:
    """The Runpod API: the Pod alive at its captured identity; a DELETE ends it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append(method)
        if method == "DELETE":
            return 204, b""
        if "DELETE" in self.calls:
            return 404, b""
        pod = {
            "id": "podpost01",
            "name": "posttrain-a40-s-2",
            "gpu": {"id": "NVIDIA A40", "count": 1},
            "dataCenterId": "test",
            "createdAt": "2027-01-15T08:00:00+00:00",
            "cost": 0.59,
        }
        return 200, json.dumps(pod).encode()


def outcome(policy: Policy, path: Path, now: float) -> tuple[str, str]:
    identity = runpod_watchdog.PodIdentity(
        "podpost01", "posttrain-a40-s-2", "NVIDIA A40", 1, "test", "2027-01-15T08:00:00+00:00"
    )
    created = runpod_watchdog.parse_timestamp(identity.created_at)
    result = check_once(
        runpod_watchdog.RunpodClient("authored-test-key", Api()),
        identity,
        Policy(**{**asdict(policy), "billing_started_at_unix": created}),
        path,
        now=lambda: created + now,
        sleep=lambda _: None,
        observed=True,
    )
    return result.state, result.reason


@pytest.mark.parametrize("kind", ["reward", "dev"])
def test_an_awaiting_heartbeat_keeps_an_idle_gpu_for_its_bound_and_no_longer(
    kind: str, tmp_path: Path
) -> None:
    policy = run2_policy()
    extension = posttrain_ops.bridge_timeouts(
        attempts_per_request=256, dev_programs=1_800
    ).extension(kind)
    assert runpod_watchdog.awaiting_extension(policy, kind) == extension
    path = tmp_path / "heartbeat.json"
    created = runpod_watchdog.parse_timestamp("2027-01-15T08:00:00+00:00")
    last_gpu = 1_000.0  # Seconds after creation.
    for idle, expected in ((extension - 1, "watching"), (extension, "terminated")):
        now = last_gpu + idle
        beat(
            path,
            now=created + now,
            activity=created + last_gpu,
            kind=kind,
            status="awaiting_rewards",
        )
        assert outcome(policy, path, now)[0] == expected, idle
    assert outcome(policy, path, last_gpu + extension)[1].startswith("idle_gpu_limit")


def test_no_extension_without_a_named_wait_or_under_run_ones_policy(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    created = runpod_watchdog.parse_timestamp("2027-01-15T08:00:00+00:00")
    now = 1_000.0 + 700  # Past the ordinary 600 s idle limit.
    run1 = Policy(18_000, 2.99, 600, 0.6, max_transfer_seconds=900, mode="posttrain")
    cases = (
        (run2_policy(), None, "awaiting_rewards"),  # No awaiting.kind: the ordinary rule.
        (run2_policy(), "reward", "running"),  # Not waiting: the ordinary rule.
        (run2_policy(), "other", "awaiting_rewards"),  # An unknown kind: no extension.
        (run1, "reward", "awaiting_rewards"),  # Run 1's policy grants none.
    )
    for policy, kind, status in cases:
        beat(path, now=created + now, activity=created + 1_000.0, kind=kind, status=status)
        assert outcome(policy, path, now)[1].startswith("idle_gpu_limit"), (kind, status)
    # A heartbeat the worker stopped refreshing is stale whatever it claims to wait on.
    beat(
        path,
        now=created + 1_000.0,
        activity=created + 1_000.0,
        kind="dev",
        status="awaiting_rewards",
    )
    assert outcome(run2_policy(), path, 1_000.0 + 600)[1].startswith("heartbeat_stale")
    found = read_heartbeat(path, now=created + 1_001, max_idle_seconds=600, mode="posttrain")
    assert found.awaiting_kind == "dev"
    assert read_heartbeat(path, now=created + 1_001, max_idle_seconds=600).error is not None


def test_extensions_are_bounded_and_belong_to_post_training_policies() -> None:
    with pytest.raises(ValueError, match="posttrain"):
        Policy(18_000, 2.99, 600, 0.6, mode="rehearsal", awaiting_reward_extension_seconds=60)
    for value in (-1.0, math.inf, 60_001.0):
        with pytest.raises(ValueError):
            Policy(18_000, 2.99, 600, 0.6, mode="posttrain", awaiting_dev_extension_seconds=value)


def test_the_policy_file_carries_the_derived_extensions_and_the_receipt_checks_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = tmp_path / "spec-2.json"
    spec_path.write_text(json.dumps(gated()))
    policy_path = tmp_path / "policy.json"
    arguments = ["limits", "--specification", str(spec_path), "--origin", "1000"]
    assert posttrain_ops.main([*arguments, "--policy-output", str(policy_path)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["wall_seconds"] == 34_800 and record["self_destruct_seconds"] == 34_620
    assert record["watchdog_cost_stop_usd"] == 5.79
    assert record["mac_reward_timeout_seconds"] == 5_600.0
    written = json.loads(policy_path.read_text())
    assert written["awaiting_reward_extension_seconds"] == 7_914.5
    assert written["awaiting_dev_extension_seconds"] == 11_806.5
    spec = Specification.from_dict(gated())
    check_policy(Policy(**written), spec)
    for field in ("awaiting_reward_extension_seconds", "awaiting_dev_extension_seconds"):
        for value in (0.0, written[field] + 1):
            with pytest.raises(ValueError, match="not the derived bound"):
                check_policy(Policy(**{**written, field: value}), spec)
    # Run 1's policy file is unchanged: no extension key, and none may be granted.
    run1 = Specification.load(posttrain_approved.SPECIFICATION_PATH)
    policy1 = posttrain_ops.watchdog_policy(run1.ceiling, origin=1.0, idle_baseline=1.0)
    assert "awaiting_reward_extension_seconds" not in policy1
    with pytest.raises(ValueError, match="not the derived bound"):
        check_policy(Policy(**{**policy1, "awaiting_reward_extension_seconds": 60.0}), run1)
    for field in ("mac_dev_timeout_seconds", "lost_bridge_seconds"):
        assert posttrain_ops.main([*arguments[:3], "--field", field]) == 0
        assert float(capsys.readouterr().out) == record[field]


def test_the_watchdog_refuses_extension_flags_other_than_its_policy_file(tmp_path: Path) -> None:
    written = posttrain_ops.watchdog_policy(
        Specification.from_dict(gated()).ceiling,
        origin=1_000.0,
        idle_baseline=1_000.0,
        bridge=Specification.from_dict(gated()).bridge_timeouts,
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(written))
    flags = Policy(**written)
    runpod_watchdog.check_policy_file(flags, path)
    for change in (
        {"awaiting_reward_extension_seconds": 0.0},
        {"awaiting_dev_extension_seconds": 1.0},
    ):
        with pytest.raises(ValueError, match="differ from the policy file"):
            runpod_watchdog.check_policy_file(Policy(**{**written, **change}), path)
    parser = runpod_watchdog.build_parser()
    argv = [
        "--mode",
        "posttrain",
        "--identity",
        "i.json",
        "--heartbeat",
        "h.json",
        "--receipt",
        "r.json",
        "--max-wall-seconds",
        "34800",
        "--max-cost-usd",
        "5.79",
        "--quoted-total-hourly",
        "0.60",
        "--awaiting-reward-extension-seconds",
        "7914.5",
        "--awaiting-dev-extension-seconds",
        "11806.5",
    ]
    parsed = runpod_watchdog.policy_from_args(parser.parse_args(argv))
    assert (parsed.awaiting_reward_extension_seconds, parsed.awaiting_dev_extension_seconds) == (
        7_914.5,
        11_806.5,
    )


# The cost margin for one full wait --------------------------------------------------------


def test_the_cost_stop_keeps_one_full_wait_at_the_quoted_rate() -> None:
    spec = Specification.from_dict(gated())
    bounds = spec.bridge_timeouts
    assert bounds is not None
    derived = spec.ceiling
    margin_usd = bounds.wait_margin_seconds / 3600 * derived.quoted_total_hourly
    assert margin_usd == pytest.approx(1.95775, abs=1e-5)  # 11,746.5 s at $0.60/h.
    # A wait may start only while it ends before every stop. At launch the worker's lifetime
    # is the in-Pod timer's, 34,620 s, and the finalization reserve takes its last 900 s, so
    # the reserve binds first: no wait starts after 21,973.5 s, $3.662 of quoted spend (the
    # lifetime's end alone would allow $3.812, the $5.80 cost stop $3.842).
    lifetime = derived.self_destruct_seconds
    rate = derived.quoted_total_hourly / 3600
    reserve = lifetime - spec.finalization_reserve_seconds - bounds.wait_margin_seconds
    end = lifetime - bounds.wait_margin_seconds
    cost = derived.worker_cost_stop_usd / rate - bounds.wait_margin_seconds
    assert (lifetime, reserve, end) == (34_620, 21_973.5, 22_873.5)
    assert cost == pytest.approx(23_053.5)
    assert [round(seconds * rate, 3) for seconds in (reserve, end, cost)] == [3.662, 3.812, 3.842]
    # The margin is the longer wait's: a reward wait alone needs less.
    assert bounds.reward_timeout_seconds < bounds.wait_margin_seconds == bounds.dev_timeout_seconds


# The in-Pod idle rule and the Mac lease ----------------------------------------------------


def test_a_fresh_lease_holds_the_pod_and_a_dead_mac_lets_it_go_within_the_bound(
    tmp_path: Path,
) -> None:
    """The worker stamps ``progress_unix`` from each fresh lease it reads during a wait; when the
    Mac dies the stamps stop, and the timer deletes the Pod 900 s after the last renewal."""
    clock = Clock()
    heartbeat = tmp_path / "heartbeat.json"
    unit = timer(
        tmp_path,
        FakeTransport([(204, {}, b""), (404, {}, b"")]),
        clock,
        idle_seconds=900.0,
        activity=(str(heartbeat),),
    )
    unit.deadline = unit.deadline_monotonic = 10**12
    gpu = clock.wall
    renewed = gpu
    for _ in range(30):  # The Mac renews its lease while it scores a long dev check.
        clock.sleep(100.0)
        renewed = clock.wall
        heartbeat.write_text(json.dumps({"last_gpu_activity_unix": gpu, "progress_unix": renewed}))
        assert unit.due() is None
    assert clock.wall - gpu == 3_000  # Far past 900 s of GPU idle, held by the lease.
    clock.sleep(899.0)  # The Mac died just after its last renewal: no further stamp.
    assert unit.due() is None
    clock.sleep(1.0)
    assert unit.due() == "idle_limit"
    assert clock.wall - renewed == 900.0
