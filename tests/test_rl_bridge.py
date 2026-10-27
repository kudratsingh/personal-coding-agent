"""Unit tests for the RL reward bridge: schemas, the Pod side and the Mac side.

The Pod side runs on a fake clock whose sleeps play the Mac's part, so each interleaving
is exact; the Mac side reads and writes a local directory pair (the fake SSH transport).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from bridge_fixtures import (
    DEV_SHA,
    DEV_TASKS,
    RL_SHA,
    SESSION,
    TRAIN,
    CountingDev,
    CountingRewards,
    FakeClock,
    attempts,
    mac_server,
    pod_client,
)

from scratchcoder.eval.parallel import Execute
from scratchcoder.rl.bridge import (
    AWAITING_REWARDS,
    DEV,
    HOST_FAILURE,
    REFUSED,
    REWARD,
    SCORED,
    TIMEOUT,
    BridgeClient,
    BridgeDevExecutor,
    BridgeError,
    BridgeRefused,
    BridgeReply,
    BridgeReplyError,
    BridgeRequest,
    BridgeRewardSource,
    BridgeSchemaError,
    DevItem,
    RequestName,
    canonical_json,
    decode_reply,
    decode_request,
    failure_reply,
    parse_request_id,
    payload_sha256,
    refusal_reply,
    scored_reply,
    verify_reply,
)
from scratchcoder.rl.bridge_mac import (
    BridgeServer,
    BridgeTransportError,
    DirectoryTransport,
    PodListing,
    SandboxDevExecutor,
)
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.rewards import (
    NO_PROGRAM,
    Attempt,
    AttemptReward,
    RewardHostFailure,
    RewardTimeout,
)
from scratchcoder.sandbox.runner import Result

NAME = RequestName(SESSION, 1, REWARD, 3)


def reward_request(items: list[Attempt] | None = None, name: RequestName = NAME) -> BridgeRequest:
    return BridgeRequest.build(
        name, items or attempts(), task_file_sha256=RL_SHA, created_unix=1_800_000_000.0
    )


def dev_request(index: int = 12, seq: int = 1) -> BridgeRequest:
    items = [DevItem("d0", "ab"), DevItem("d1", "abc"), DevItem("d0", "")]
    return BridgeRequest.build(
        RequestName(SESSION, seq, DEV, index), items, task_file_sha256=DEV_SHA, created_unix=0.0
    )


def fake_rewards(request: BridgeRequest) -> list[AttemptReward]:
    return FakeRewardSource().rewards(
        request.index, [item for item in request.items if isinstance(item, Attempt)]
    )


def scored(request: BridgeRequest) -> BridgeReply:
    return scored_reply(request, rewards=fake_rewards(request), seconds=1.5, answered_unix=2.0)


def mutate(raw: bytes, **changes: Any) -> bytes:
    value = json.loads(raw)
    value.update(changes)
    return json.dumps(value, separators=(",", ":")).encode()


# ---------------------------------------------------------------------------- schemas


def test_request_round_trip_and_canonical_payload() -> None:
    request = reward_request()
    assert request.request_id == f"{SESSION}-000001-reward-000003"
    assert parse_request_id(request.name.file_name) == NAME
    rows = [item.to_json() for item in request.items]
    assert request.payload_sha256 == payload_sha256(rows)
    # The canonical form is sorted, compact ASCII JSON, so both sides hash the same bytes.
    assert canonical_json(rows).startswith(b'[{"code":null,"completion_tokens":1,"sample":0')
    assert decode_request(request.to_bytes(), expected_id=request.request_id) == request
    assert decode_request(dev_request().to_bytes()) == dev_request()
    unicode = reward_request([Attempt("t0", 0, "print('é\\n')\n# ☃", 3)])
    assert decode_request(unicode.to_bytes()) == unicode
    with pytest.raises(ValueError):
        BridgeRequest.build(NAME, [], task_file_sha256=RL_SHA, created_unix=0)
    with pytest.raises(ValueError):
        BridgeRequest.build(NAME, [DevItem("d0", "x")], task_file_sha256=RL_SHA, created_unix=0)
    with pytest.raises(ValueError):
        BridgeRequest.build(NAME, attempts(), task_file_sha256="A" * 64, created_unix=0)
    for bad in ("0123456789ab-1-reward-3", "0123456789AB-000001-reward-000003", "x.json"):
        with pytest.raises(BridgeSchemaError):
            parse_request_id(bad)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"extra": 1}, "schema"),
        ({"format": "rl-bridge-request-v0"}, "schema"),
        ({"count": 7}, "schema"),
        ({"index": 4}, "schema"),
        ({"seq": 2}, "schema"),
        ({"kind": "dev"}, "schema"),
        ({"request_id": f"{SESSION}-000001-reward-000004"}, "schema"),
        ({"payload_sha256": "c" * 64}, "payload_sha256"),
        ({"task_file_sha256": "short"}, "schema"),
        ({"created_unix": float("nan")}, "schema"),
        ({"items": [{"task_id": "t0", "sample": 0, "code": None}] * 8}, "schema"),
    ],
)
def test_decode_request_refuses_a_wrong_schema(change: dict[str, Any], reason: str) -> None:
    raw = mutate(reward_request().to_bytes(), **change)
    with pytest.raises(BridgeSchemaError) as caught:
        decode_request(raw)
    assert caught.value.reason == reason


def test_decode_request_refuses_corrupt_partial_and_oversized_bytes() -> None:
    raw = reward_request().to_bytes()
    for broken in (raw[: len(raw) // 2], b"\xff\xfe", b"[1, 2]", b'{"a": 1, "a": 2}'):
        with pytest.raises(BridgeSchemaError) as caught:
            decode_request(broken)
        assert caught.value.reason in {"corrupt", "schema"}
    with pytest.raises(BridgeSchemaError) as caught:
        decode_request(raw, max_bytes=len(raw) - 1)
    assert caught.value.reason == "oversized"
    with pytest.raises(BridgeSchemaError):
        decode_request(raw, expected_id=f"{SESSION}-000002-reward-000003")
    # A tampered program changes the payload digest.
    value = json.loads(raw)
    value["items"][1]["code"] = "y"
    with pytest.raises(BridgeSchemaError) as caught:
        decode_request(json.dumps(value).encode())
    assert caught.value.reason == "payload_sha256"


def test_reply_round_trip_and_verification() -> None:
    request = reward_request()
    reply = scored(request)
    assert decode_reply(reply.to_bytes()) == reply
    verify_reply(reply, request)
    keys = set(json.loads(reply.to_bytes())["results"][0])
    assert keys == {"reward", "status", "audit_flags", "seconds", "unfinished"}
    dev = dev_request()
    dev_reply = scored_reply(dev, passed=[True, False, True], seconds=0.0, answered_unix=0.0)
    assert decode_reply(dev_reply.to_bytes()) == dev_reply
    verify_reply(dev_reply, dev)
    refusal = refusal_reply(dev.name, "unknown_task", "d9", request=None, answered_unix=0.0)
    verify_reply(decode_reply(refusal.to_bytes()), dev)  # unknown fields may be null
    # A non-finite reward reaches the RL loop, which stops on it rather than crashing.
    nan = scored_reply(
        request,
        rewards=[
            AttemptReward(0.0, NO_PROGRAM),
            *[AttemptReward(math.nan, "completed")] * 7,
        ],
        seconds=0.0,
        answered_unix=0.0,
    )
    assert math.isnan(decode_reply(nan.to_bytes()).rewards[1].reward)


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": f"{SESSION}-000002-reward-000003"},
        {"index": 4, "request_id": f"{SESSION}-000001-reward-000004"},
        {"count": 7},
        {"payload_sha256": "c" * 64},
        {"task_file_sha256": DEV_SHA},
    ],
)
def test_verify_reply_refuses_a_reply_to_another_request(change: dict[str, Any]) -> None:
    request = reward_request()
    raw = mutate(scored(request).to_bytes(), **change)
    with pytest.raises(BridgeSchemaError):
        verify_reply(decode_reply(raw), request)


def test_decode_reply_refuses_malformed_replies() -> None:
    request = reward_request()
    raw = scored(request).to_bytes()
    rows = json.loads(raw)["results"]
    for change in (
        {"extra": True},
        {"format": "rl-bridge-reply-v2"},
        {"results": rows[:-1]},
        {"outcome": "maybe"},
        {"outcome": REFUSED},  # a refusal names a reason
        {"reason": "stale"},  # only a refusal names one
        {"results": [{**rows[0], "status": "host_timeout"}, *rows[1:]]},
        {"results": [{**rows[0], "reward": 1.5}, *rows[1:]]},
        {"detail": "x" * 501},
        {"seconds": -1},
    ):
        with pytest.raises(BridgeSchemaError):
            decode_reply(mutate(raw, **change))
    with pytest.raises(BridgeSchemaError):
        decode_reply(raw, max_bytes=10)
    dev = scored_reply(dev_request(), passed=[True, False, True], seconds=0, answered_unix=0)
    with pytest.raises(BridgeSchemaError):
        decode_reply(mutate(dev.to_bytes(), results=[1, 0, 1]))
    # A scored reply that gives a missing program anything but 0 / no_program is refused.
    wrong = scored_reply(
        request,
        rewards=[AttemptReward(1.0, "completed")] * request.count,
        seconds=0,
        answered_unix=0,
    )
    with pytest.raises(BridgeSchemaError):
        verify_reply(wrong, request)


# --------------------------------------------------------------------------- Pod side


def answer_with(pod: Path, replies: list[bytes | None], request_file: list[Path]) -> Any:
    """A sleep hook that writes the next reply (or nothing) into the inbox each sleep."""

    def hook() -> None:
        outbox = sorted((pod / "outbox").glob("*.json"))
        if not outbox or not replies:
            return
        request_file.append(outbox[0])
        raw = replies.pop(0)
        if raw is not None:
            (pod / "inbox" / outbox[0].name).write_bytes(raw)

    return hook


def test_rewards_travel_to_the_mac_and_back(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    seen: list[dict[str, Any] | None] = []
    with pod_client(pod, clock) as client:
        server = mac_server(pod, tmp_path / "mac")

        def mac() -> None:
            seen.append(client.heartbeat())
            server.poll_once()

        clock.hooks.append(mac)
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60)
        batch = attempts()
        rewards = source.rewards(3, batch)
        assert rewards == FakeRewardSource().rewards(3, batch)
        assert rewards[0] == AttemptReward(0.0, NO_PROGRAM)  # travelled, never executed
        assert client.heartbeat() is None and client.waiting() is None
        status = seen[0]
        assert status is not None and status["status"] == AWAITING_REWARDS
        assert (status["kind"], status["index"], status["count"]) == (REWARD, 3, 8)
        assert status["request_id"] == f"{SESSION}-000001-reward-000003"
        assert status["age_seconds"] == pytest.approx(clock.slept[0])
        # Answered requests and replies are archived; the boxes are empty again.
        assert not list((pod / "outbox").iterdir()) and not list((pod / "inbox").iterdir())
        done = sorted(path.name for path in (pod / "done").iterdir())
        assert done == [
            f"{SESSION}-000001-reward-000003.{part}.json" for part in ("reply", "request")
        ]
        assert source.rewards(4, []) == []  # nothing to send, no request
        assert client.requests_sent == 1
        server.close()


def test_a_timeout_withdraws_the_request_and_raises_reward_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    with pod_client(pod, clock, poll_seconds=2.0) as client:
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=7)
        with pytest.raises(RewardTimeout, match="within 7 s"):
            source.rewards(5, attempts())
        assert clock.now == pytest.approx(7.0) and clock.slept == [2, 2, 2, 1]
        assert not list((pod / "outbox").iterdir())
        assert [path.name for path in (pod / "withdrawn").iterdir()] == [
            f"{SESSION}-000001-reward-000005.json"
        ]
        assert client.waiting() is None
    with pytest.raises(ValueError):
        BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=0)
    with pytest.raises(ValueError):
        BridgeRewardSource(client, task_file_sha256="nope", timeout=1)


def test_the_pod_rejects_wrong_replies_and_accepts_the_right_one(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    batch = attempts()
    request = reward_request(batch)
    good = scored(request).to_bytes()
    bad = [
        mutate(good, index=4, request_id=f"{SESSION}-000001-reward-000004"),  # wrong index
        mutate(good, count=7),
        mutate(good, payload_sha256="c" * 64),
        mutate(good, task_file_sha256=DEV_SHA),
        mutate(good, format="other"),
        good[: len(good) // 2],  # a partial file
        b"\x00" * 64,  # corrupt
        good + b" " * 2048,  # over the reply bound below
    ]
    replies: list[bytes | None] = [None, *bad, good]
    files: list[Path] = []
    clock.hooks.append(answer_with(pod, replies, files))
    with pod_client(pod, clock, max_rejected=len(bad), max_reply_bytes=len(good) + 100) as client:
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=600)
        assert source.rewards(3, batch) == FakeRewardSource().rewards(3, batch)
    rejected = sorted((pod / "rejected").iterdir())
    assert len(rejected) == len(bad)


def test_too_many_rejected_replies_raise_a_bridge_reply_error(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    clock.hooks.append(answer_with(pod, [b"{}"] * 5, []))
    with pod_client(pod, clock, max_rejected=2) as client:
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=600)
        with pytest.raises(BridgeReplyError, match="3 replies"):
            source.rewards(3, attempts())
        assert not list((pod / "outbox").iterdir())  # withdrawn
        assert client.waiting() is None


@pytest.mark.parametrize(
    ("outcome", "error"),
    [(HOST_FAILURE, RewardHostFailure), (TIMEOUT, RewardTimeout), (REFUSED, BridgeRefused)],
)
def test_failure_replies_map_to_the_rl_loop_errors(
    tmp_path: Path, outcome: str, error: type[Exception]
) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    request = reward_request()
    if outcome == REFUSED:
        reply = refusal_reply(request.name, "task_file", "wrong", request=request, answered_unix=0)
    else:
        reply = failure_reply(request, outcome, "the Mac said so", seconds=1, answered_unix=0)
    clock.hooks.append(answer_with(pod, [reply.to_bytes()], []))
    with pod_client(pod, clock) as client:
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60)
        with pytest.raises(error):
            source.rewards(3, attempts())


def test_leftovers_and_stray_replies_never_answer_a_request(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    for box in ("outbox", "inbox"):
        (pod / box).mkdir(parents=True)
    old = "fedcba987654-000009-reward-000001.json"
    (pod / "outbox" / old).write_text("{}")
    (pod / "inbox" / old).write_text("{}")
    clock = FakeClock()
    with pod_client(pod, clock) as client:
        assert client.withdrawn_at_start == 2
        assert not list((pod / "outbox").iterdir()) and not list((pod / "inbox").iterdir())
        # A reply to an earlier request (out of order) sits in the inbox: it is never read
        # as the answer to this one, and it is moved aside before the request is written.
        first = reward_request()
        (pod / "inbox" / first.name.file_name).write_bytes(scored(first).to_bytes())
        stray = RequestName(SESSION, 1, REWARD, 2)
        (pod / "inbox" / stray.file_name).write_bytes(b"{}")
        server = mac_server(pod, tmp_path / "mac")
        clock.hooks.append(lambda: server.poll_once())
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60)
        assert source.rewards(3, attempts()) == FakeRewardSource().rewards(3, attempts())
        withdrawn = {path.name for path in (pod / "withdrawn").iterdir()}
        assert f"stray-{first.name.file_name}" in withdrawn
        assert f"stray-{stray.file_name}" in withdrawn
        server.close()


def test_one_client_per_bridge_root(tmp_path: Path) -> None:
    with pod_client(tmp_path / "pod"):
        with pytest.raises(BridgeError, match="another bridge client"):
            BridgeClient(tmp_path / "pod")
    with pod_client(tmp_path / "pod") as again:
        assert again.session == SESSION
    with pytest.raises(ValueError):
        BridgeClient(tmp_path / "other", session="not-hex")
    closed = BridgeClient(tmp_path / "closed")
    closed.close()
    with pytest.raises(BridgeError, match="closed"):
        closed.exchange(REWARD, 1, attempts(), task_file_sha256=RL_SHA, timeout=1)


def test_the_dev_executor_sends_only_ids_and_programs(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    sent: list[dict[str, Any]] = []
    with pod_client(pod, clock) as client:
        dev = CountingDev()
        server = mac_server(pod, tmp_path / "mac", dev=dev)

        def mac() -> None:
            for path in (pod / "outbox").glob("*.json"):
                sent.append(json.loads(path.read_bytes()))
            server.poll_once()

        clock.hooks.append(mac)
        executor = BridgeDevExecutor(client, task_file_sha256=DEV_SHA, timeout=60)
        tasks = {task.id: task for task in DEV_TASKS}
        pairs = [(tasks["d0"], "ab"), (tasks["d1"], "abc"), (tasks["d2"], "")]
        assert executor(pairs) == [True, False, True]
        with executor.labelled(25):
            assert executor(pairs[:1]) == [True]
        assert executor([]) == []
        assert [row["index"] for row in sent] == [0, 25]  # call count, then the label
        assert sent[0]["items"] == [
            {"task_id": "d0", "code": "ab"},
            {"task_id": "d1", "code": "abc"},
            {"task_id": "d2", "code": ""},
        ]
        assert "hidden" not in json.dumps(sent)  # only IDs and programs travel
        assert dev.calls[0] == [("d0", "ab"), ("d1", "abc"), ("d2", "")]
        server.close()


# --------------------------------------------------------------------------- Mac side


def write_request(pod: Path, request: BridgeRequest) -> None:
    (pod / "outbox").mkdir(parents=True, exist_ok=True)
    (pod / "inbox").mkdir(parents=True, exist_ok=True)
    (pod / "outbox" / request.name.file_name).write_bytes(request.to_bytes())


def reply_for(pod: Path, name: RequestName) -> BridgeReply:
    return decode_reply((pod / "inbox" / name.file_name).read_bytes())


def test_the_mac_refuses_unknown_tasks_and_a_wrong_task_file(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    rewards, dev = CountingRewards(), CountingDev()
    unknown = reward_request(
        [Attempt("t0", 0, "x", 1), Attempt("train-secret", 1, "y", 1)],
        name=RequestName("111111111111", 1, REWARD, 1),
    )
    wrong_file = BridgeRequest.build(
        RequestName("222222222222", 1, REWARD, 1),
        attempts(),
        task_file_sha256=DEV_SHA,
        created_unix=0,
    )
    dev_unknown = BridgeRequest.build(
        RequestName("333333333333", 1, DEV, 0),
        [DevItem("t0", "x")],  # a train task is not in the dev subset
        task_file_sha256=DEV_SHA,
        created_unix=0,
    )
    for request in (unknown, wrong_file, dev_unknown):
        write_request(pod, request)
    with mac_server(pod, tmp_path / "mac", rewards=rewards, dev=dev) as server:
        assert len(server.poll_once()) == 3
    refused = reply_for(pod, unknown.name)
    assert refused.outcome == REFUSED and refused.reason == "unknown_task"
    assert "kept screening list" in refused.detail and "train-secret" in refused.detail
    assert reply_for(pod, wrong_file.name).reason == "task_file"
    refused = reply_for(pod, dev_unknown.name)
    assert refused.reason == "unknown_task" and "dev subset" in refused.detail
    assert rewards.calls == [] and dev.calls == []  # nothing was scored
    # The Pod sees a refusal as an error that stops the run, not as a reward.
    clock = FakeClock()
    pod2 = tmp_path / "pod2"
    with pod_client(pod2, clock) as client:
        server = mac_server(pod2, tmp_path / "mac2")
        clock.hooks.append(lambda: server.poll_once())
        source = BridgeRewardSource(client, task_file_sha256=DEV_SHA, timeout=60)
        with pytest.raises(BridgeRefused) as caught:
            source.rewards(1, attempts())
        assert caught.value.reason == "task_file"
        server.close()


def test_the_mac_refuses_oversized_and_wrong_schema_and_defers_a_partial_file(
    tmp_path: Path,
) -> None:
    pod = tmp_path / "pod"
    big = reward_request(attempts(("t0", "t1", "t2")), name=RequestName(SESSION, 1, REWARD, 1))
    write_request(pod, big)
    other = "fedcba987654"
    schema = RequestName(other, 1, REWARD, 1)
    (pod / "outbox" / schema.file_name).write_bytes(
        mutate(reward_request(name=schema).to_bytes(), extra=1)
    )
    partial = RequestName("aaaaaaaaaaaa", 1, REWARD, 1)
    whole = reward_request(name=partial).to_bytes()
    (pod / "outbox" / partial.file_name).write_bytes(whole[:40])
    corrupt = RequestName("bbbbbbbbbbbb", 1, REWARD, 1)
    (pod / "outbox" / corrupt.file_name).write_bytes(b"not json at all")
    rewards = CountingRewards()
    with mac_server(
        pod, tmp_path / "mac", rewards=rewards, max_request_bytes=len(whole) + 64
    ) as server:
        server.poll_once()
        assert reply_for(pod, big.name).reason == "oversized"
        assert reply_for(pod, schema).reason == "schema"
        # A file that does not parse is looked at again before it is refused.
        assert not (pod / "inbox" / partial.file_name).exists()
        assert not (pod / "inbox" / corrupt.file_name).exists()
        (pod / "outbox" / partial.file_name).write_bytes(whole[:80])  # a slow writer
        server.poll_once()
        assert reply_for(pod, corrupt).reason == "corrupt"  # unchanged: refused
        assert not (pod / "inbox" / partial.file_name).exists()  # it grew: deferred again
        (pod / "outbox" / partial.file_name).write_bytes(whole)  # the writer finished
        server.poll_once()
        progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
        assert progress["deferred"] == 3 and progress["refused"] == 3
    assert reply_for(pod, partial).outcome == SCORED
    assert rewards.calls == [(1, 8)]


def test_a_repeated_request_is_answered_from_the_store_never_rescored(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = reward_request()
    write_request(pod, request)
    rewards = CountingRewards()
    reads: list[str] = []

    class Counting(DirectoryTransport):
        def read_request(self, name: str, *, max_bytes: int) -> bytes:
            reads.append(name)
            return super().read_request(name, max_bytes=max_bytes)

    with mac_server(pod, tmp_path / "mac", rewards=rewards, transport=Counting(pod)) as server:
        assert server.poll_once() == [request.request_id]
        first = (pod / "inbox" / request.name.file_name).read_bytes()
        assert server.poll_once() == []  # reply still there: nothing pulled, nothing sent
        assert reads == [request.name.file_name]
        (pod / "inbox" / request.name.file_name).unlink()  # the Pod quarantined it
        assert server.poll_once() == [request.request_id]
        assert (pod / "inbox" / request.name.file_name).read_bytes() == first
        assert rewards.calls == [(3, 8)]
        progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
        assert progress["answered"] == 1 and progress["replayed"] == 1
        assert progress["last_answered"]["replayed"] is True
        # The same ID with another payload is a conflict, never the stored answer.
        (pod / "inbox" / request.name.file_name).unlink()
        changed = reward_request([Attempt("t0", 0, "zz", 1)])
        write_request(pod, changed)
        server.poll_once()
        assert reply_for(pod, request.name).reason == "conflict"
        assert rewards.calls == [(3, 8)]


def test_out_of_order_requests_are_refused_as_stale(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    older = reward_request(name=RequestName(SESSION, 1, REWARD, 1))
    newer = reward_request(name=RequestName(SESSION, 2, REWARD, 2))
    write_request(pod, newer)
    write_request(pod, older)
    rewards = CountingRewards()
    with mac_server(pod, tmp_path / "mac", rewards=rewards) as server:
        server.poll_once()
        assert reply_for(pod, older.name).reason == "stale"
        assert reply_for(pod, newer.name).outcome == SCORED
        assert rewards.calls == [(2, 8)]
    late = reward_request(name=RequestName(SESSION, 1, REWARD, 9))
    write_request(pod, late)
    with mac_server(pod, tmp_path / "mac", rewards=rewards) as server:  # after a restart
        server.poll_once()
    assert reply_for(pod, late.name).reason == "stale"
    assert rewards.calls == [(2, 8)]


class Crash(BaseException):
    """Stands in for the Mac process dying (not an ordinary exception)."""


def test_a_mac_restart_mid_request_never_scores_twice(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = reward_request()
    write_request(pod, request)
    rewards = CountingRewards()

    class DiesOnPush(DirectoryTransport):
        def write_reply(self, name: str, data: bytes) -> None:
            raise Crash

    server = mac_server(pod, tmp_path / "mac", rewards=rewards, transport=DiesOnPush(pod))
    with pytest.raises(Crash):
        server.poll_once()
    server.close()
    assert rewards.calls == [(3, 8)] and not (pod / "inbox" / request.name.file_name).exists()
    with mac_server(pod, tmp_path / "mac", rewards=rewards) as restarted:
        assert restarted.poll_once() == [request.request_id]
    assert rewards.calls == [(3, 8)]  # answered from the store
    assert reply_for(pod, request.name) == decode_reply(
        (tmp_path / "mac" / "replies" / request.name.file_name).read_bytes()
    )


def test_a_mac_crash_while_scoring_scores_again_after_the_restart(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = reward_request()
    write_request(pod, request)
    broken = CountingRewards(error=RuntimeError("sandbox bug"))
    server = mac_server(pod, tmp_path / "mac", rewards=broken)
    with pytest.raises(RuntimeError):
        server.serve(poll_seconds=0, max_polls=1, progress_seconds=None)
    progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
    assert progress["state"] == "failed" and "sandbox bug" in progress["last_error"]
    assert progress["current"] is None and progress["last_seen"]["request_id"] == request.request_id
    server.close()
    rewards = CountingRewards()
    with mac_server(pod, tmp_path / "mac", rewards=rewards) as restarted:
        restarted.poll_once()
    assert rewards.calls == [(3, 8)] and reply_for(pod, request.name).outcome == SCORED


class Flaky(DirectoryTransport):
    def __init__(self, root: Path, failures: dict[str, int]) -> None:
        super().__init__(root)
        self.failures = failures

    def _maybe_fail(self, what: str) -> None:
        if self.failures.get(what, 0) > 0:
            self.failures[what] -= 1
            raise BridgeTransportError(f"ssh {what} dropped")

    def listing(self) -> PodListing:
        self._maybe_fail("list")
        return super().listing()

    def read_request(self, name: str, *, max_bytes: int) -> bytes:
        self._maybe_fail("read")
        return super().read_request(name, max_bytes=max_bytes)

    def write_reply(self, name: str, data: bytes) -> None:
        self._maybe_fail("write")
        super().write_reply(name, data)


def test_ssh_failures_are_retried_with_bounded_backoff(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = reward_request()
    write_request(pod, request)
    rewards = CountingRewards()
    sleeps: list[float] = []
    transport = Flaky(pod, {"list": 2, "read": 1, "write": 3})
    with mac_server(
        pod, tmp_path / "mac", rewards=rewards, transport=transport, sleeps=sleeps
    ) as server:
        assert server.poll_once() == [request.request_id]
        assert sleeps == [1.0, 2.0, 1.0, 1.0, 2.0, 4.0]
        assert rewards.calls == [(3, 8)] and reply_for(pod, request.name).outcome == SCORED
        progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
        assert progress["transport_failures"] == 6
    sleeps.clear()
    transport = Flaky(pod, {"list": 100})
    with mac_server(
        pod, tmp_path / "mac", transport=transport, sleeps=sleeps, transport_retries=2
    ) as server:
        with pytest.raises(BridgeTransportError):
            server.serve(poll_seconds=5, max_consecutive_failures=2, progress_seconds=None)
        # Two cycles of three tries with 1 s and 2 s backoff, and a poll wait between them.
        assert sleeps == [1.0, 2.0, 5, 1.0, 2.0]
        progress = json.loads((tmp_path / "mac" / "progress.json").read_bytes())
        assert progress["state"] == "failed" and progress["consecutive_failures"] == 2


@pytest.mark.parametrize(
    ("error", "outcome"),
    [(RewardHostFailure("docker unavailable"), HOST_FAILURE), (RewardTimeout("slow"), TIMEOUT)],
)
def test_mac_host_failures_and_timeouts_become_stored_replies(
    tmp_path: Path, error: Exception, outcome: str
) -> None:
    pod = tmp_path / "pod"
    request = reward_request()
    write_request(pod, request)
    rewards = CountingRewards(error=error)
    with mac_server(pod, tmp_path / "mac", rewards=rewards) as server:
        server.poll_once()
    reply = reply_for(pod, request.name)
    assert reply.outcome == outcome and reply.rewards == ()
    assert "docker" not in reply.detail and "slow" not in reply.detail  # stays on the Mac
    assert (tmp_path / "mac" / "replies" / request.name.file_name).exists()
    events = (tmp_path / "mac" / "events.jsonl").read_text()
    assert "scoring_failed" in events


def test_a_failed_pre_scoring_check_runs_nothing(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = dev_request()
    write_request(pod, request)
    dev = CountingDev()

    def check(_: BridgeRequest) -> None:
        raise RuntimeError("image changed")

    with mac_server(pod, tmp_path / "mac", dev=dev, before_scoring=check) as server:
        server.poll_once()
    assert reply_for(pod, request.name).outcome == HOST_FAILURE and dev.calls == []


def test_progress_records_what_was_seen_scored_and_answered(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    request = dev_request(index=12)
    write_request(pod, request)
    states: list[dict[str, Any]] = []
    progress_path = tmp_path / "mac" / "progress.json"

    def dev(pairs: Any) -> list[bool]:
        states.append(json.loads(progress_path.read_bytes()))
        return [True] * len(pairs)

    with mac_server(pod, tmp_path / "mac", dev=dev) as server:
        server.serve(poll_seconds=0, max_polls=2, progress_seconds=None)
    during = states[0]
    assert during["state"] == "scoring"
    assert during["current"]["request_id"] == request.request_id
    assert during["current"]["count"] == 3
    final = json.loads(progress_path.read_bytes())
    assert final["state"] == "stopped" and final["current"] is None
    assert final["last_seen"]["request_id"] == request.request_id
    answered = final["last_answered"]
    assert answered["request_id"] == request.request_id and answered["kind"] == DEV
    assert answered["index"] == 12 and answered["outcome"] == SCORED
    assert final["pod_task_sha256"] == {"reward": RL_SHA, "dev": DEV_SHA}
    assert final["allowed_tasks"] == {"reward": len(TRAIN), "dev": 4}
    assert final["polls"] == 2 and final["updated_unix"] >= final["started_unix"]


def test_the_server_checks_its_configuration(tmp_path: Path) -> None:
    pod = tmp_path / "pod"
    for kwargs in (
        {"pod_task_sha256": {"reward": RL_SHA}},
        {"pod_task_sha256": {"reward": RL_SHA, "dev": "x"}},
        {"reward_tasks": [DEV_TASKS[0]]},
        {"dev_tasks": TRAIN},
        {"reward_tasks": [TRAIN[0], TRAIN[0]]},
    ):
        options: dict[str, Any] = {
            "reward_tasks": TRAIN,
            "dev_tasks": [DEV_TASKS[0]],
            "pod_task_sha256": {"reward": RL_SHA, "dev": DEV_SHA},
            "state": tmp_path / "mac",
            "reward_source": CountingRewards(),
            "dev_execute": CountingDev(),
            **kwargs,
        }
        with pytest.raises(ValueError):
            BridgeServer(DirectoryTransport(pod), **options)
    with mac_server(pod, tmp_path / "mac"):
        with pytest.raises(BridgeError, match="another bridge server"):
            mac_server(pod, tmp_path / "mac")


def fake_execute(statuses: dict[str, list[str]]) -> Execute:
    """An ``isolated_execution`` stand-in: each program's code names its status sequence."""

    def execute(code: str, tests: str, mode: str) -> Result:
        status = statuses[code].pop(0)
        if status == "host":  # the host watchdog ended the call: never a verdict
            return Result("timeout", host_stalled=True)
        return Result(status)

    return execute


def test_the_sandbox_dev_executor_never_guesses_a_verdict() -> None:
    task = DEV_TASKS[0]
    statuses = {
        "ok": ["passed"],
        "bad": ["failed"],
        "slow": ["timeout"],
        "flaky": ["host", "passed"],
    }
    executor = SandboxDevExecutor(workers=2, execute=fake_execute(statuses), sleep=lambda s: None)
    pairs = [(task, code) for code in ("ok", "bad", "slow", "flaky")]
    assert executor(pairs) == [True, False, False, True]
    stuck = SandboxDevExecutor(
        workers=1, retries=2, execute=fake_execute({"x": ["host"] * 3}), sleep=lambda s: None
    )
    with pytest.raises(RewardHostFailure, match="1 of 1 dev programs stayed unresolved"):
        stuck([(task, "x")])
    big = "x" * 200_000  # over the sandbox input limit: the candidate's failure, not run
    assert SandboxDevExecutor(execute=fake_execute({}))([(task, big)]) == [False]


def test_a_dead_sandbox_fails_a_dev_check_without_running_every_program() -> None:
    task = DEV_TASKS[0]
    calls: list[str] = []

    def down(code: str, tests: str, mode: str) -> Result:
        calls.append(code)
        return Result("unavailable")

    executor = SandboxDevExecutor(
        workers=1, retries=1, execute=down, sleep=lambda s: None, max_consecutive_unresolved=2
    )
    with pytest.raises(RewardHostFailure, match="looks down"):
        executor([(task, f"p{index}") for index in range(50)])
    assert len(calls) < 10  # stopped after two unresolved calls, not 50 x 2 tries


def test_the_client_carries_one_request_at_a_time(tmp_path: Path) -> None:
    clock = FakeClock()
    pod = tmp_path / "pod"
    errors: list[Exception] = []
    with pod_client(pod, clock) as client:
        server = mac_server(pod, tmp_path / "mac")

        def second_caller() -> None:
            try:
                client.exchange(REWARD, 9, attempts(), task_file_sha256=RL_SHA, timeout=1)
            except BridgeError as error:
                errors.append(error)
            server.poll_once()

        clock.hooks.append(second_caller)
        source = BridgeRewardSource(client, task_file_sha256=RL_SHA, timeout=60)
        assert source.rewards(3, attempts()) == FakeRewardSource().rewards(3, attempts())
        assert client.requests_sent == 1
        server.close()
    assert errors and "one request at a time" in str(errors[0])
