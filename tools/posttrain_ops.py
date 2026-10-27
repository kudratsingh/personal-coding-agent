"""Operations for the post-training Pods: the ceiling-derived lifetime, the R2 run layout,
the R2 prefix guard and the Mac's final verification. Nothing here contacts Runpod.

The owner fixes each post-training run by a dollar ceiling (run 1: $3.00). Every time and
cost limit is derived from that ceiling and the quoted rate in one place,
``ceiling_limits``, never from the pretraining rehearsal's fixed 60,000 s:

- the worker's cost stop is the ceiling;
- the Mac watchdog's cost stop is one cent inside it;
- the Pod wall is the ceiling's worth of seconds at the quoted rate, rounded down;
- the in-Pod timer fires ``self_destruct_margin_seconds`` (180 s) before that wall.

For run 1 at $0.60/h: $3.00, $2.99, 18,000 s and 17,820 s. The worker, the readiness
receipt, the runbook's watchdog policy and the bootstrap's timer limit all take them from
here (``limits`` prints them), and the receipt and the worker refuse anything weaker.

Run 2 waits on the Mac's RL rewards and dev verdicts through the reward bridge. Its bounds
on one wait are derived here too, in one place (``bridge_timeouts``): the Pod's reward and
dev timeouts, the Mac's ``--reward-timeout`` and ``--dev-timeout``, the Mac's checks before it
scores a request (one window pass and the image check), the Mac watchdog's idle extension
while the heartbeat reads ``awaiting_rewards``, the margin the worker's cost, wall and
reserve stops keep for one full wait, and the lost-bridge bound that ends a wait whose Mac
lease went stale before the in-Pod idle rule fires.

``limits`` (Mac)
    Print a specification's derived limits; with ``--origin`` also the timer deadline, and
    with ``--policy-output`` write the watchdog policy file.
``round-trip`` (Pod)
    The pretraining rehearsal's R2 round trip under ``posttrain-runs/<run>/<pod id>/``,
    gated on the upload rate one weights-only label needs (``REQUIRED_UPLOAD_BYTES_PER_SECOND``).
``verify-final`` (Mac)
    Check a finished run's ``report.json`` and every artifact it lists in R2 with the Mac's
    own key, the check the supervisor applies before it lets a Pod be deleted.

Credentials come only from the environment and are never printed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.rl.bridge_mac import (
    LEASE_SECONDS,
    image_check_seconds_bound,
    window_check_seconds_bound,
)
from scratchcoder.rl.rewards import DEFAULT_HOST_RETRIES, DEFAULT_WORKERS
from scratchcoder.sandbox.budget import io_seconds
from scratchcoder.sandbox.reward import FUNCTION_BUDGET_SECONDS, HOST_SLACK_SECONDS, MAX_IO_CASES
from tools import rehearsal_ops
from tools.r2_store import SHA256_HEX, Store
from tools.rehearsal_binding import r2_credentials
from tools.rehearsal_ops import ObjectStore, head_sha256

RUN_ROOT = "posttrain-runs"
RUNS = ("1", "2")
"""The post-training runs this code drives: run 1 (probe, ARC, evaluation, SFT, screening)
and run 2 (RL with the Mac in the loop, the more-SFT control and their evaluation)."""
ABSOLUTE_LIFETIME_SECONDS = 60_000
"""The in-Pod timer's absolute cap (``pod_self_destruct.ABSOLUTE_DEADLINE_SECONDS``)."""
SELF_DESTRUCT_MARGIN_SECONDS = 180
WATCHDOG_MARGIN_USD = 0.01
QUOTED_GPU_HOURLY = rehearsal_ops.QUOTED_GPU_HOURLY
QUOTED_TOTAL_HOURLY = rehearsal_ops.QUOTED_TOTAL_HOURLY
MAC_IDLE_SECONDS = rehearsal_ops.IDLE_SECONDS
"""The Mac watchdog's idle limit, as in the pretraining rehearsal (600 s)."""
POD_IDLE_SECONDS = 900
"""The in-Pod idle self-destruct: the Pod deletes itself after this long with no GPU work and
no progress, independent of the Mac (``pod_self_destruct.py --idle-seconds``)."""
MAX_TRANSFER_SECONDS = rehearsal_ops.MAX_TRANSFER_SECONDS
DRAIN_SECONDS = rehearsal_ops.DRAIN_SECONDS
PULL_RESERVE_SECONDS = rehearsal_ops.PULL_RESERVE_SECONDS
S_WEIGHTS_LABEL_BYTES = 550_000_000
"""One weights-only S label: the 541,142,016 bytes of float32 weights, the save's framing and
the label's record, rounded up to 0.55 GB. SFT, RL and more-SFT each start a fresh optimizer,
so no post-training label carries optimizer state (the pretraining rehearsal's labels were
1.65 GB)."""
REQUIRED_UPLOAD_BYTES_PER_SECOND = S_WEIGHTS_LABEL_BYTES / DRAIN_SECONDS * 2
"""About 3.06 MB/s: one weights-only label inside the 360 s final drain with a factor-2
margin, re-derived as the pretraining rehearsal derived 9.17 MB/s for 1.65 GB. One label is
the worst case for run 1: its only weights label (the selected SFT checkpoint) is uploaded an
hour before the final drain, and every other artifact is a small JSON or JSON Lines file."""
ROUND_TRIP_PROBE_BYTES = rehearsal_ops.ROUND_TRIP_PROBE_BYTES
UPLOADING_STATUS = rehearsal_ops.UPLOADING_STATUS
AWAITING_REWARDS_STATUS = "awaiting_rewards"
"""Run 2's heartbeat while the worker waits for the Mac's RL rewards or dev verdicts; run 1
never writes it. The Mac watchdog gives it a bounded idle extension (``bridge_timeouts``)."""
IN_PROGRESS_STATUSES = ("running", UPLOADING_STATUS, AWAITING_REWARDS_STATUS)
SUCCESS_STATUSES = rehearsal_ops.SUCCESS_STATUSES
FAILURE_STATUSES = rehearsal_ops.FAILURE_STATUSES
READINESS_KIND = "posttrain-readiness"
REPORT_KIND = "posttrain-run-report"
OWNER_GATES = rehearsal_ops.OWNER_GATES
EVIDENCE_GATES = (
    "r2_round_trip_tested",
    "r2_inputs_staged_and_hash_checked",
    "mac_parent_copies_verified",
)
"""Set only from receipts: the Pod's R2 round trip, the Pod's fetch and re-verification of
every bound input, and the Mac's own pinned copies of the parent checkpoints."""
READINESS_GATES = (*OWNER_GATES, *EVIDENCE_GATES, "pod_self_destruct_armed")
REPORT_MAX_BYTES = rehearsal_ops.REPORT_MAX_BYTES
R2_FINAL_KEYS = frozenset({"report_key", "report_sha256", "artifact_count"})
LIMIT_KEYS = frozenset(
    {
        "ceiling_usd",
        "quoted_gpu_hourly",
        "quoted_total_hourly",
        "self_destruct_margin_seconds",
        "watchdog_margin_usd",
        "mac_idle_seconds",
        "pod_idle_seconds",
        "max_transfer_seconds",
        "finalization_reserve_seconds",
        "startup_seconds",
    }
)
_SAFE_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class CeilingLimits:
    """Every time and cost limit of one run, derived from its ceiling and quoted rate."""

    ceiling_usd: float
    quoted_total_hourly: float
    wall_seconds: int
    worker_cost_stop_usd: float
    watchdog_cost_stop_usd: float
    self_destruct_seconds: int
    self_destruct_margin_seconds: int

    def deadline_unix(self, origin: float) -> float:
        """The in-Pod timer's deadline for a Pod whose billing began at ``origin``."""
        return float(origin) + self.self_destruct_seconds


def ceiling_limits(
    ceiling_usd: float,
    quoted_total_hourly: float,
    *,
    self_destruct_margin_seconds: float = SELF_DESTRUCT_MARGIN_SECONDS,
    watchdog_margin_usd: float = WATCHDOG_MARGIN_USD,
) -> CeilingLimits:
    """The run's limits from its dollar ceiling: the one place they are derived.

    The wall is the ceiling's worth of whole seconds at the quoted rate, rounded down, so
    the wall's quoted cost never exceeds the ceiling. It must fit the in-Pod timer's
    absolute 60,000 s cap.
    """
    ceiling = _positive(ceiling_usd, "the ceiling")
    hourly = _positive(quoted_total_hourly, "the quoted total hourly rate")
    margin = _positive(self_destruct_margin_seconds, "the self-destruct margin")
    cents = _positive(watchdog_margin_usd, "the watchdog margin")
    if margin != int(margin):
        raise ValueError("the self-destruct margin must be whole seconds")
    wall = math.floor(ceiling * 3600 / hourly + 1e-6)
    if wall > ABSOLUTE_LIFETIME_SECONDS:
        raise ValueError(
            f"a ${ceiling:.2f} ceiling at ${hourly:.2f}/h is {wall:,} s, above the in-Pod "
            f"timer's absolute {ABSOLUTE_LIFETIME_SECONDS:,} s cap"
        )
    if margin >= wall or cents >= ceiling:
        raise ValueError("the margins must be smaller than the wall and the ceiling")
    return CeilingLimits(
        ceiling_usd=ceiling,
        quoted_total_hourly=hourly,
        wall_seconds=wall,
        worker_cost_stop_usd=ceiling,
        watchdog_cost_stop_usd=round(ceiling - cents, 2),
        self_destruct_seconds=int(wall - margin),
        self_destruct_margin_seconds=int(margin),
    )


def read_limits(raw: Mapping[str, Any], *, allow_null: Sequence[str] = ()) -> dict[str, Any]:
    """A specification's ``limits`` object, exactly ``LIMIT_KEYS``, each a positive number;
    ``allow_null`` names keys that may still be ``null`` (run 2's ceiling before the gate)."""
    if not isinstance(raw, Mapping) or set(raw) != LIMIT_KEYS:
        raise ValueError(f"limits must have exactly the keys {sorted(LIMIT_KEYS)}")
    for key in LIMIT_KEYS:
        if raw[key] is None and key in allow_null:
            continue
        if raw[key] is None and key == "ceiling_usd":
            raise ValueError(
                "limits.ceiling_usd is null: the ceiling is fixed at the owner's gate "
                "(python -m tools.posttrain_approved run2-gate)"
            )
        _positive(raw[key], f"limits.{key}")
    return dict(raw)


def specification_limits(specification: Mapping[str, Any]) -> CeilingLimits:
    """``ceiling_limits`` for a specification file's ``limits`` object."""
    limits = read_limits(specification.get("limits", {}))
    return ceiling_limits(
        limits["ceiling_usd"],
        limits["quoted_total_hourly"],
        self_destruct_margin_seconds=limits["self_destruct_margin_seconds"],
        watchdog_margin_usd=limits["watchdog_margin_usd"],
    )


# Run 2's reward bridge: every bound on one wait, derived in one place -----------------------

REWARD_TRIES = 1 + DEFAULT_HOST_RETRIES
"""``LocalRewardSource`` tries an attempt once and retries a host failure 3 times: 4 tries."""
REWARD_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
"""Its waits before those retries: 1 s doubling, capped at 8 s (``LocalRewardSource.backoff``)."""
MAC_POLL_SECONDS = 2.0
"""The Mac server's wait between idle polls (``scratchcoder rl-bridge --poll-seconds``)."""
POD_POLL_SECONDS = 0.5
"""The Pod client's wait between looks at its inbox (``BridgeClient`` default)."""
SSH_TIMEOUT_SECONDS = 120.0
SSH_LIST_TIMEOUT_SECONDS = 30.0
SSH_RETRIES = 4
SSH_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
"""``BridgeServer``'s SSH retries: 4 after the first try, waiting 1, 2, 4 and 8 s."""
AWAITING_SLACK_SECONDS = 60.0
"""Kept on top of a wait's timeout in the Mac watchdog's idle extension: the CPU work between
the last GPU stamp and the request, the supervisor's 5 s mirror and the watchdog's poll."""
LOST_BRIDGE_MARGIN_SECONDS = 300.0
"""How long before the in-Pod idle rule a wait with a stale Mac lease is given up, so the
worker saves and uploads its weights and finalizes before the timer deletes the Pod."""


@dataclass(frozen=True)
class BridgeTimeouts:
    """Run 2's bounds on one bridge wait (seconds), all from ``bridge_timeouts``.

    - ``attempt_try_seconds``: one program's longest try, the reward channel's longest budget
      (27 s for a 100-case I/O task; 8 s for a function task) plus its 15 s host slack. It is
      also the slack for "a program still running" when a Mac-side bound passes.
    - ``reward_attempt_seconds``: one attempt through all its tries and backoff waits.
    - ``mac_reward_timeout_seconds``: a reward request's worst-case scoring on the Mac,
      ``ceil(attempts / workers)`` rounds of that attempt (``rl-bridge --reward-timeout``).
    - ``mac_dev_timeout_seconds``: a dev check's worst case, ``ceil(programs / workers)``
      rounds of one try plus one more try for a program still running
      (``rl-bridge --dev-timeout``).
    - ``bridge_allowance_seconds``: the bridge's own time on top of the Mac's: one Mac poll,
      one listing, one read and one write over SSH, each tried 5 times to its timeout with
      the backoff waits between, and one Pod poll.
    - ``window_check_seconds``: one of the Mac's execution-window passes at its worst, run
      before a request is scored when the last pass is old (every call to its bound,
      ``scratchcoder.rl.bridge_mac.window_check_seconds_bound``).
    - ``image_check_seconds``: the Mac's pinned-image check before every request at its
      worst (every try to its bound and every backoff wait).
    - ``reward_timeout_seconds`` and ``dev_timeout_seconds``: the Pod's, the Mac's plus the
      allowance and both checks, so a Mac-side ``timeout`` reply always reaches a Pod that
      still waits, whatever the checks before it took.
    - ``awaiting_*_extension_seconds``: the Mac watchdog's idle limit while the heartbeat
      reads ``awaiting_rewards`` for that kind of request: the Pod's timeout plus the slack.
    - ``wait_margin_seconds``: one full wait (the longer kind); the worker's cost, wall and
      reserve stops keep it free, since ``should_stop`` is not read during a wait.
    - ``lost_bridge_seconds``: a wait with no Mac lease renewal for this long is given up
      (``bridge_lost``), well before the in-Pod idle rule deletes the Pod.
    """

    attempts_per_request: int
    dev_programs: int
    workers: int
    attempt_try_seconds: float
    reward_attempt_seconds: float
    mac_reward_timeout_seconds: float
    mac_dev_timeout_seconds: float
    bridge_allowance_seconds: float
    window_check_seconds: float
    image_check_seconds: float
    reward_timeout_seconds: float
    dev_timeout_seconds: float
    awaiting_slack_seconds: float
    awaiting_reward_extension_seconds: float
    awaiting_dev_extension_seconds: float
    wait_margin_seconds: float
    lease_seconds: float
    lost_bridge_seconds: float

    @property
    def mac_check_seconds(self) -> float:
        """The Mac's checks before it scores one request, at their worst."""
        return self.window_check_seconds + self.image_check_seconds

    def mac_timeout(self, kind: str) -> float:
        """The Mac's own bound for a ``reward`` or ``dev`` request (``rl-bridge``)."""
        if kind == "reward":
            return self.mac_reward_timeout_seconds
        if kind == "dev":
            return self.mac_dev_timeout_seconds
        raise ValueError(f"unknown bridge request kind {kind!r}")

    def timeout(self, kind: str) -> float:
        """The Pod's timeout for a ``reward`` or ``dev`` request."""
        if kind == "reward":
            return self.reward_timeout_seconds
        if kind == "dev":
            return self.dev_timeout_seconds
        raise ValueError(f"unknown bridge request kind {kind!r}")

    def extension(self, kind: str) -> float:
        """The Mac watchdog's idle extension while the Pod waits on a request of ``kind``."""
        if kind == "reward":
            return self.awaiting_reward_extension_seconds
        if kind == "dev":
            return self.awaiting_dev_extension_seconds
        raise ValueError(f"unknown bridge request kind {kind!r}")


def bridge_timeouts(
    *,
    attempts_per_request: int,
    dev_programs: int,
    workers: int = DEFAULT_WORKERS,
    mac_poll_seconds: float = MAC_POLL_SECONDS,
    pod_poll_seconds: float = POD_POLL_SECONDS,
    ssh_timeout_seconds: float = SSH_TIMEOUT_SECONDS,
    ssh_list_timeout_seconds: float = SSH_LIST_TIMEOUT_SECONDS,
    ssh_retries: int = SSH_RETRIES,
    awaiting_slack_seconds: float = AWAITING_SLACK_SECONDS,
    pod_idle_seconds: float = POD_IDLE_SECONDS,
) -> BridgeTimeouts:
    """Every bound on one run-2 bridge wait, from the channel's budgets and the bridge's rules.

    For the rehearsal (256 attempts, 1,800 dev programs, 8 workers): a try is 27 + 15 = 42 s,
    an attempt 4 x 42 + 1 + 2 + 4 = 175 s, the Mac's reward bound 32 x 175 = 5,600 s and its
    dev bound 225 x 42 + 42 = 9,492 s; the allowance 2 + 5 x (30 + 120 + 120) + 3 x 15 + 0.5 =
    1,397.5 s; the Mac's checks before a request, one window pass of 11 x 60 + 2 x 30 + 10 =
    730 s and the image check of 4 x 30 + 1 + 2 + 4 = 127 s, so the Pod waits 5,600 + 1,397.5 +
    857 = 7,854.5 s for rewards and 11,746.5 s for a dev check.
    """
    for name, count in (
        ("attempts_per_request", attempts_per_request),
        ("dev_programs", dev_programs),
        ("workers", workers),
    ):
        if type(count) is not int or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(ssh_retries) is not int or not 0 <= ssh_retries <= len(SSH_BACKOFF_SECONDS):
        raise ValueError(f"ssh_retries must lie in [0, {len(SSH_BACKOFF_SECONDS)}]")
    for name, value in (
        ("mac_poll_seconds", mac_poll_seconds),
        ("pod_poll_seconds", pod_poll_seconds),
        ("ssh_timeout_seconds", ssh_timeout_seconds),
        ("ssh_list_timeout_seconds", ssh_list_timeout_seconds),
        ("awaiting_slack_seconds", awaiting_slack_seconds),
        ("pod_idle_seconds", pod_idle_seconds),
    ):
        _positive(value, name)
    budget = max(FUNCTION_BUDGET_SECONDS, io_seconds(MAX_IO_CASES))
    try_seconds = budget + HOST_SLACK_SECONDS
    attempt = REWARD_TRIES * try_seconds + math.fsum(REWARD_BACKOFF_SECONDS)
    mac_reward = math.ceil(attempts_per_request / workers) * attempt
    mac_dev = math.ceil(dev_programs / workers) * try_seconds + try_seconds
    tries = 1 + ssh_retries
    backoff = math.fsum(SSH_BACKOFF_SECONDS[:ssh_retries])
    allowance = (
        mac_poll_seconds
        + tries * (ssh_list_timeout_seconds + 2 * ssh_timeout_seconds)
        + 3 * backoff
        + pod_poll_seconds
    )
    window_check = window_check_seconds_bound()
    image_check = image_check_seconds_bound()
    reward = mac_reward + allowance + window_check + image_check
    dev = mac_dev + allowance + window_check + image_check
    lost = pod_idle_seconds - LOST_BRIDGE_MARGIN_SECONDS
    if lost <= LEASE_SECONDS:
        raise ValueError("the in-Pod idle limit leaves no room to detect a lost bridge")
    return BridgeTimeouts(
        attempts_per_request=attempts_per_request,
        dev_programs=dev_programs,
        workers=workers,
        attempt_try_seconds=try_seconds,
        reward_attempt_seconds=attempt,
        mac_reward_timeout_seconds=mac_reward,
        mac_dev_timeout_seconds=mac_dev,
        bridge_allowance_seconds=allowance,
        window_check_seconds=window_check,
        image_check_seconds=image_check,
        reward_timeout_seconds=reward,
        dev_timeout_seconds=dev,
        awaiting_slack_seconds=float(awaiting_slack_seconds),
        awaiting_reward_extension_seconds=reward + awaiting_slack_seconds,
        awaiting_dev_extension_seconds=dev + awaiting_slack_seconds,
        wait_margin_seconds=max(reward, dev),
        lease_seconds=LEASE_SECONDS,
        lost_bridge_seconds=lost,
    )


BRIDGE_KEYS = frozenset(
    {
        "root",
        "workers",
        "mac_poll_seconds",
        "pod_poll_seconds",
        "ssh_timeout_seconds",
        "ssh_list_timeout_seconds",
        "ssh_retries",
        "awaiting_slack_seconds",
        "reward_timeout_seconds",
        "dev_timeout_seconds",
        "mac_reward_timeout_seconds",
        "mac_dev_timeout_seconds",
        "awaiting_reward_extension_seconds",
        "awaiting_dev_extension_seconds",
        "wait_margin_seconds",
        "lost_bridge_seconds",
    }
)
"""A run-2 specification's ``bridge`` object: the inputs of ``bridge_timeouts`` and the bounds
it derives, written out so the runbook and the policy can be read off it."""
DERIVED_BRIDGE_KEYS = (
    "reward_timeout_seconds",
    "dev_timeout_seconds",
    "mac_reward_timeout_seconds",
    "mac_dev_timeout_seconds",
    "awaiting_reward_extension_seconds",
    "awaiting_dev_extension_seconds",
    "wait_margin_seconds",
    "lost_bridge_seconds",
)


def specification_bridge_timeouts(specification: Mapping[str, Any]) -> BridgeTimeouts | None:
    """``bridge_timeouts`` for a run-2 specification (``None`` for a run without a bridge).

    The attempts per request are the RL batch (tasks per update x group size), the dev
    programs the dev subset's tasks x (samples + one greedy). Every derived value the
    specification writes down must equal the derivation."""
    bridge = specification.get("bridge")
    if bridge is None:
        return None
    if not isinstance(bridge, Mapping) or set(bridge) != BRIDGE_KEYS:
        raise ValueError(f"bridge must have exactly the keys {sorted(BRIDGE_KEYS)}")
    rl = specification.get("rl")
    if not isinstance(rl, Mapping):
        raise ValueError("a specification with a bridge needs its rl section")
    limits = read_limits(specification.get("limits", {}), allow_null=("ceiling_usd",))
    derived = bridge_timeouts(
        attempts_per_request=rl["tasks_per_update"] * rl["group_size"],
        dev_programs=rl["dev_tasks"] * (rl["dev_samples"] + 1),
        workers=bridge["workers"],
        mac_poll_seconds=bridge["mac_poll_seconds"],
        pod_poll_seconds=bridge["pod_poll_seconds"],
        ssh_timeout_seconds=bridge["ssh_timeout_seconds"],
        ssh_list_timeout_seconds=bridge["ssh_list_timeout_seconds"],
        ssh_retries=bridge["ssh_retries"],
        awaiting_slack_seconds=bridge["awaiting_slack_seconds"],
        pod_idle_seconds=limits["pod_idle_seconds"],
    )
    differ = [key for key in DERIVED_BRIDGE_KEYS if bridge[key] != getattr(derived, key)]
    if differ:
        raise ValueError(f"the bridge values {differ} differ from their derivation")
    return derived


def watchdog_policy(
    limits: CeilingLimits,
    *,
    origin: float,
    idle_baseline: float,
    idle_seconds: float = MAC_IDLE_SECONDS,
    transfer_seconds: float = MAX_TRANSFER_SECONDS,
    bridge: BridgeTimeouts | None = None,
) -> dict[str, Any]:
    """The Mac watchdog's policy file for a run: wall and cost stop from ``ceiling_limits``,
    and for run 2 the bounded idle extensions of an ``awaiting_rewards`` heartbeat (run 1's
    policy has none, so the idle rule applies to every heartbeat unchanged)."""
    policy: dict[str, Any] = {
        "mode": "posttrain",
        "max_wall_seconds": limits.wall_seconds,
        "max_cost_usd": limits.watchdog_cost_stop_usd,
        "max_idle_seconds": idle_seconds,
        "quoted_total_hourly": limits.quoted_total_hourly,
        "max_transfer_seconds": transfer_seconds,
        "billing_started_at_unix": origin,
        "idle_baseline_unix": idle_baseline,
    }
    if bridge is not None:
        policy.update(
            awaiting_reward_extension_seconds=bridge.awaiting_reward_extension_seconds,
            awaiting_dev_extension_seconds=bridge.awaiting_dev_extension_seconds,
        )
    return policy


def run_prefix(run: str, pod_id: str) -> str:
    """``posttrain-runs/<run>/<pod id>/``, where the run's Pod writes every artifact."""
    if run not in RUNS:
        raise ValueError(f"unknown post-training run {run!r}; the runs are {list(RUNS)}")
    if not isinstance(pod_id, str) or not _SAFE_POD_ID.fullmatch(pod_id):
        raise ValueError("the Pod ID must be a plain identifier")
    return f"{RUN_ROOT}/{run}/{pod_id}/"


class PrefixViolation(PermissionError):
    """A write or delete outside the run's own R2 prefix was refused before any request."""


class GuardedStore(Store):
    """A store that refuses every write and delete outside one prefix: the R2 prefix guard.

    Every request a ``Store`` makes goes through ``_request`` except ``download``'s signed
    GET, so the guard sits there: any method but GET and HEAD whose key does not lie under
    ``prefix`` (plain components, no ``.`` or ``..``) raises ``PrefixViolation`` before a
    socket is opened. Bucket listings (GET with no key) and reads anywhere stay allowed:
    the worker fetches its inputs from other prefixes.
    """

    READ_METHODS = frozenset({"GET", "HEAD"})

    def __init__(self, inner: Store, prefix: str) -> None:
        if not isinstance(prefix, str) or not prefix.endswith("/") or prefix.startswith("/"):
            raise ValueError("the guarded prefix must be relative and end in '/'")
        if any(part in ("", ".", "..") for part in prefix[:-1].split("/")):
            raise ValueError("the guarded prefix must be plain path components")
        super().__init__(
            inner.credentials,
            transport=inner.transport,
            opener=inner.opener,
            clock=inner.clock,
            attempts=inner.attempts,
            sleep=inner.sleep,
        )
        self.prefix = prefix
        self.refused: list[dict[str, str]] = []

    def allowed(self, method: str, key: str) -> bool:
        if method in self.READ_METHODS:
            return True
        parts = key.split("/")
        return key.startswith(self.prefix) and all(part not in ("", ".", "..") for part in parts)

    def _request(
        self,
        method: str,
        key: str = "",
        *,
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO[bytes] | bytes | None = None,
        payload_sha256: str | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if not self.allowed(method, key):
            self.refused.append({"method": method, "key": key})
            raise PrefixViolation(
                f"R2 {method} {key!r} refused: outside the run prefix {self.prefix}"
            )
        if payload_sha256 is None:
            return super()._request(method, key, query=query, headers=headers, body=body)
        return super()._request(
            method, key, query=query, headers=headers, body=body, payload_sha256=payload_sha256
        )


def check_r2_final(value: object, *, run: str, pod_id: str) -> dict[str, Any]:
    """A heartbeat's ``r2_final``: the report key under this run's prefix and its SHA-256."""
    if not isinstance(value, dict) or set(value) != R2_FINAL_KEYS:
        raise ValueError(f"r2_final must have exactly the keys {sorted(R2_FINAL_KEYS)}")
    final = dict(value)
    if final["report_key"] != run_prefix(run, pod_id) + "report.json":
        raise ValueError("the final report key must be this run's report.json")
    if not isinstance(final["report_sha256"], str) or not SHA256_HEX.fullmatch(
        final["report_sha256"]
    ):
        raise ValueError("r2_final report_sha256 must be 64 lower-case hex digits")
    count = final["artifact_count"]
    if type(count) is not int or count < 1:
        raise ValueError("r2_final artifact_count must be a positive integer")
    return final


def report_artifacts(report: Mapping[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    """The artifacts a worker report lists, each under ``prefix`` with a size and SHA-256."""
    rows = report.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("the report lists no artifacts")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"label", "name", "key", "sha256", "bytes"}:
            raise ValueError("an artifact row has exactly label, name, key, sha256 and bytes")
        label, name = row["label"], row["name"]
        if not all(isinstance(item, str) and _LABEL.fullmatch(item) for item in (label, name)):
            raise ValueError("artifact labels and names must be plain components")
        if row["key"] != f"{prefix}{label}/{name}" or row["key"] in seen:
            raise ValueError(f"artifact {label}/{name} has another key or is listed twice")
        if not isinstance(row["sha256"], str) or not SHA256_HEX.fullmatch(row["sha256"]):
            raise ValueError(f"artifact {label}/{name} has no SHA-256")
        if type(row["bytes"]) is not int or row["bytes"] < 1:
            raise ValueError(f"artifact {label}/{name} has no size")
        seen.add(row["key"])
    return [dict(row) for row in rows]


def verify_final(
    store: ObjectStore,
    r2_final: object,
    *,
    run: str,
    pod_id: str,
    destination: Path,
    statuses: Sequence[str] = SUCCESS_STATUSES,
) -> dict[str, Any]:
    """Verify a run's ``report.json`` and every artifact it lists in R2; keep the report.

    HEAD shows the report's SHA-256, and the report is downloaded pinned to it. It must be
    this run's and Pod's post-training report with one of ``statuses``, list exactly
    ``artifact_count`` artifacts under this run's prefix, and each one's HEAD must show its
    recorded SHA-256 and size. Every completed stage's outputs must be among them.
    """
    final = check_r2_final(r2_final, run=run, pod_id=pod_id)
    prefix = run_prefix(run, pod_id)
    report_head = head_sha256(store, final["report_key"], expected=final["report_sha256"])
    if not 0 < report_head["bytes"] <= REPORT_MAX_BYTES:
        raise ValueError(f"report.json in R2 must hold 1 to {REPORT_MAX_BYTES:,} bytes")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    store.download(
        final["report_key"], path, sha256=final["report_sha256"], size=report_head["bytes"]
    )
    report = json.loads(path.read_bytes())
    if not isinstance(report, dict) or report.get("kind") != REPORT_KIND:
        raise ValueError("the R2 report is not a post-training run report")
    if report.get("run") != run or report.get("pod_id") != pod_id:
        raise ValueError("the R2 report belongs to another run or Pod")
    status = report.get("status")
    if status not in statuses:
        raise ValueError(f"the R2 report status {status!r} is not one of {list(statuses)}")
    artifacts = report_artifacts(report, prefix=prefix)
    if len(artifacts) != final["artifact_count"]:
        raise ValueError("the report lists another number of artifacts than the heartbeat")
    listed = {(row["label"], row["name"]) for row in artifacts}
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise ValueError("the R2 report must list its stages")
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("the R2 report's stages must be objects")
        if stage.get("status") != "completed":
            continue
        for output in stage.get("outputs", []):
            if (output.get("label"), output.get("name")) not in listed:
                raise ValueError(f"stage {stage.get('name')} output {output} is not in R2")
    heads = []
    for row in artifacts:
        head = head_sha256(store, row["key"], expected=row["sha256"])
        if head["bytes"] != row["bytes"]:
            raise ValueError(f"{row['key']}: the size in R2 differs from the report's")
        heads.append(head)
    return {
        "schema_version": 1,
        "kind": "posttrain-r2-final-verification",
        "status": "verified",
        "run": run,
        "pod_id": pod_id,
        "prefix": prefix,
        "report": report_head,
        "report_status": status,
        "artifacts": heads,
        "verified_at": utc_now(),
    }


def run_summary(
    report: Mapping[str, Any],
    *,
    run: str,
    pod_id: str,
    verification: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    """A small record of a run for the private archive; the artifacts stay in R2.

    Run 2's adds its results: RL's and the control's records, the selection and the bridge's
    record with its waits counted rather than listed (``report.json`` beside it has them)."""
    stages = []
    for entry in report.get("stages") or []:
        if isinstance(entry, dict):
            stages.append(
                {
                    key: entry.get(key)
                    for key in ("name", "kind", "status", "seconds", "outputs", "stop_reason")
                }
            )
    return {
        "schema_version": 1,
        "kind": "posttrain-run-summary",
        "run": run,
        "pod_id": pod_id,
        "r2_prefix": run_prefix(run, pod_id),
        "source": source,
        "status": report.get("status"),
        "stop_reason": report.get("stop_reason"),
        "stages": stages,
        "probe": report.get("probe"),
        "sft": report.get("sft"),
        "cost": report.get("cost"),
        "timings": report.get("timings"),
        "limits": report.get("limits"),
        "r2_final_verified": verification is not None,
        "r2_verification": None if verification is None else dict(verification),
        "created_at": utc_now(),
        **(run2_results(report) if run == "2" else {}),
    }


def run2_results(report: Mapping[str, Any]) -> dict[str, Any]:
    """Run 2's results for its summary: the arms, the selection and the bridge."""
    bridge = report.get("bridge")
    if isinstance(bridge, Mapping):
        waits = bridge.get("waits")
        bridge = {key: value for key, value in bridge.items() if key != "waits"}
        bridge["wait_count"] = len(waits) if isinstance(waits, list) else None
    return {
        "rl": report.get("rl"),
        "more_sft": report.get("more_sft"),
        "selection": report.get("selection"),
        "bridge": bridge,
    }


def fetch_artifacts(
    store: ObjectStore,
    report_path: Path,
    output: Path,
    *,
    receipt: Path,
    labels: Sequence[str] = (),
) -> dict[str, Any]:
    """Download the artifacts a verified ``report.json`` lists, each pinned to its SHA-256.

    ``labels`` keeps only labels that start with one of these prefixes. Files land at
    ``<output>/<label>/<name>``; one already present with the pinned bytes is kept.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("kind") != REPORT_KIND:
        raise ValueError("not a post-training run report")
    prefix = run_prefix(str(report.get("run")), str(report.get("pod_id")))
    rows = []
    for row in report_artifacts(report, prefix=prefix):
        if labels and not any(row["label"].startswith(label) for label in labels):
            continue
        target = output / row["label"] / row["name"]
        record = rehearsal_ops.fetch_object(
            store,
            key=row["key"],
            sha256=row["sha256"],
            size=row["bytes"],
            output=target,
            receipt=output / ".receipts" / f"{row['label']}--{row['name']}.json",
        )
        rows.append({**row, "path": str(target), "method": record["method"]})
    value = {
        "schema_version": 1,
        "kind": "posttrain-artifact-fetch",
        "created_at": utc_now(),
        "report_sha256": file_sha256(report_path),
        "files": rows,
    }
    atomic_json(receipt, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser(
        "fetch-artifacts", help="download a verified run's artifacts pinned (Mac)"
    )
    artifacts.add_argument("--report", type=Path, required=True, help="the verified report.json")
    artifacts.add_argument("--output", type=Path, required=True)
    artifacts.add_argument("--receipt", type=Path, required=True)
    artifacts.add_argument("--label", action="append", default=[], help="a label prefix")
    limits = commands.add_parser("limits", help="print a specification's derived limits (Mac)")
    limits.add_argument("--specification", type=Path, required=True)
    limits.add_argument("--origin", type=float, help="the billing origin (unix seconds)")
    limits.add_argument("--idle-baseline", type=float, help="the watchdog's idle baseline")
    limits.add_argument("--policy-output", type=Path, help="write the watchdog policy here")
    limits.add_argument("--field", help="print only this value (for shell variables)")
    probe = commands.add_parser("round-trip", help="test PUT/HEAD/GET/DELETE in R2 (Pod)")
    probe.add_argument("--run", choices=RUNS, required=True)
    probe.add_argument("--pod-id", required=True)
    probe.add_argument("--receipt", type=Path, required=True)
    probe.add_argument("--key-prefix", default=rehearsal_ops.POD_KEY_PREFIX)
    final = commands.add_parser("verify-final", help="check a run's final state in R2 (Mac)")
    final.add_argument("--run", choices=RUNS, required=True)
    final.add_argument("--pod-id", required=True)
    final.add_argument("--report-sha256", required=True)
    final.add_argument("--artifact-count", type=int, required=True)
    final.add_argument("--output", type=Path, required=True, help="directory for report.json")
    return parser


def limits_record(args: argparse.Namespace) -> dict[str, Any]:
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise ValueError("the specification must be a JSON object")
    derived = specification_limits(specification)
    bridge = specification_bridge_timeouts(specification)
    record: dict[str, Any] = {"run": specification.get("run"), **asdict(derived)}
    if bridge is not None:
        record.update(asdict(bridge))  # Run 2's wait bounds, by name (``--field``).
    if args.origin is not None:
        record["self_destruct_deadline_unix"] = int(derived.deadline_unix(args.origin))
        baseline = args.origin if args.idle_baseline is None else args.idle_baseline
        record["policy"] = watchdog_policy(
            derived, origin=args.origin, idle_baseline=baseline, bridge=bridge
        )
        if args.policy_output is not None:
            atomic_json(args.policy_output, record["policy"])
    elif args.policy_output is not None or args.idle_baseline is not None:
        raise ValueError("--policy-output and --idle-baseline need --origin")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "limits":
            record = limits_record(args)
            if args.field is not None:
                if args.field not in record or isinstance(record[args.field], dict):
                    raise ValueError(f"no plain field {args.field!r} among {sorted(record)}")
                print(record[args.field])
            else:
                print(json.dumps(record, indent=1, sort_keys=True))
            return 0
        if args.command == "verify-final":
            prefix = run_prefix(args.run, args.pod_id)
            r2_final = {
                "report_key": prefix + "report.json",
                "report_sha256": args.report_sha256,
                "artifact_count": args.artifact_count,
            }
            store = Store(r2_credentials(key_prefix=rehearsal_ops.MAC_KEY_PREFIX))
            result = verify_final(
                store,
                r2_final,
                run=args.run,
                pod_id=args.pod_id,
                destination=args.output,
                statuses=(*SUCCESS_STATUSES, *FAILURE_STATUSES),
            )
            atomic_json(args.output / "r2-verification.json", result)
            print(json.dumps({key: result[key] for key in ("status", "report_status")}))
            return 0
        if args.command == "fetch-artifacts":
            store = Store(r2_credentials(key_prefix=rehearsal_ops.MAC_KEY_PREFIX))
            fetched = fetch_artifacts(
                store, args.report, args.output, receipt=args.receipt, labels=args.label
            )
            print(json.dumps({"files": len(fetched["files"]), "output": str(args.output)}))
            return 0
        prefix = run_prefix(args.run, args.pod_id)
        store = GuardedStore(Store(r2_credentials(key_prefix=args.key_prefix)), prefix)
        record = rehearsal_ops.round_trip(
            store,
            run=args.run,
            pod_id=args.pod_id,
            receipt=args.receipt,
            prefix=prefix,
            required_upload_bytes_per_second=REQUIRED_UPLOAD_BYTES_PER_SECOND,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"posttrain-ops: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    print(json.dumps({key: record[key] for key in sorted(record) if key != "steps"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
