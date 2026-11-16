"""Family ``jobs``: job queues and the order and times they run in."""

from __future__ import annotations

from typing import Any

from scratchcoder.agent.synth.template import Bug, Family, Func, Params, Template


def _pick(p: Params, queue: list[Any], bug: Any = None) -> int:
    best = 0 if bug == "empty_zero" else -1
    for i, (_name, prio) in enumerate(queue):
        if bug == "latest_tie":
            better = prio >= queue[best][1]
        elif bug == "lowest":
            better = prio < queue[best][1]
        else:
            better = prio > queue[best][1]
        if best == -1 or better:
            best = i
    return best


def _order(p: Params, queue: list[Any], bug: Any = None) -> list[str]:
    pending = list(queue)
    names = []
    while pending:
        names.append(pending.pop(_pick(p, pending, bug))[0])
    return names


def _waits(p: Params, durations: list[int], bug: Any = None) -> list[int]:
    out = []
    clock = 1 if bug == "start_one" else 0
    for d in durations:
        if bug == "after":
            clock += d
            out.append(clock)
            continue
        out.append(clock)
        clock = d if bug == "no_sum" else clock + d
    return out


def _finish(p: Params, durations: list[int], bug: Any = None) -> list[int]:
    return [w + d for w, d in zip(_waits(p, durations, bug), durations, strict=False)]


def _rr(p: Params, durations: list[int], bug: Any = None) -> list[int]:
    left = list(durations)
    slices = []
    while sum(left) > 0:
        for i in range(len(left)):
            if left[i] > 0:
                slices.append(i)
                left[i] = max(left[i] - p["quantum"], 0)
    return slices


def _cool(p: Params, names: list[str], bug: Any = None) -> int:
    last: dict[str, int] = {}
    slot = 0
    for name in names:
        if name in last and slot - last[name] <= p["gap"]:
            slot = last[name] + p["gap"] + 1
        last[name] = slot
        slot += 1
    return slot


Q1 = [("build", 2), ("test", 5), ("lint", 5), ("docs", 1)]
Q2 = [("a", 1), ("b", 1), ("c", 1)]
Q3 = [("x", 3), ("y", 7), ("z", 2)]

FAMILY = Family(
    id="jobs",
    domain="job queues: priority dispatch, waiting and finish times, round robin, cooldowns",
    modules={"main": ("jobs", "tasks", "work"), "dispatch": ("dispatch", "runner", "plan")},
    docs={"main": "Job queue helpers.", "dispatch": "Dispatch order and timing."},
    names={
        "pick": ("next_job", "pick_job", "choose_job"),
        "order": ("run_order", "dispatch_order", "schedule"),
        "waits": ("wait_times", "waiting", "start_times"),
        "finish": ("finish_times", "completion", "end_times"),
        "rr": ("round_robin", "take_turns", "time_slices"),
        "cool": ("cooldown_slots", "spaced_slots", "slots_needed"),
    },
    consts={"quantum": (2, 3), "gap": (1, 2)},
    funcs=(
        Func(
            "pick",
            "main",
            "def $pick(queue):\n"
            '    """Index of the job to run next: the highest priority, the earliest on ties;\n'
            '    -1 for an empty queue. Each job is a (name, priority) pair."""\n'
            "    best = -1\n"
            "    for i, (name, prio) in enumerate(queue):\n"
            "        if best == -1 or prio > queue[best][1]:\n"
            "            best = i\n"
            "    return best",
            _pick,
            public=[("ties", (Q1,)), ("first", (Q3,)), ("empty", ([],))],
            hidden=[
                (Q2,),
                ([("solo", 4)],),
                ([("a", 1), ("b", 9)],),
                ([("a", 9), ("b", 1)],),
                ([("a", 3), ("b", 3), ("c", 4), ("d", 4)],),
                ([("a", 0), ("b", 0)],),
                ([],),
            ],
        ),
        Func(
            "order",
            "dispatch",
            "def $order(queue):\n"
            '    """Job names in the order they run: the next job each time, removed once it\n'
            '    has run."""\n'
            "    pending = list(queue)\n"
            "    names = []\n"
            "    while pending:\n"
            "        names.append(pending.pop($pick(pending))[0])\n"
            "    return names",
            _order,
            public=[("mixed", (Q1,))],
            hidden=[([],), (Q2,), (Q3,), ([("a", 1), ("b", 2), ("c", 1)],)],
            uses=("pick",),
        ),
        Func(
            "waits",
            "main",
            "def $waits(durations):\n"
            '    """How long each job waits when the jobs run one after another in order."""\n'
            "    out = []\n"
            "    clock = 0\n"
            "    for d in durations:\n"
            "        out.append(clock)\n"
            "        clock += d\n"
            "    return out",
            _waits,
            hidden=[
                ([],),
                ([5],),
                ([1, 2, 3],),
                ([0, 0, 4],),
                ([7, 1, 1, 7],),
                ([2, 2],),
            ],
        ),
        Func(
            "finish",
            "dispatch",
            "def $finish(durations):\n"
            '    """When each job ends: its wait plus its own duration."""\n'
            "    return [w + d for w, d in zip($waits(durations), durations)]",
            _finish,
            public=[("three", ([3, 1, 4],)), ("one", ([6],))],
            hidden=[([],), ([2],), ([1, 1, 1, 1],), ([5, 0, 2],), ([10, 20],)],
            uses=("waits",),
        ),
        Func(
            "rr",
            "main",
            "def $rr(durations):\n"
            '    """Job index of each time slice when the jobs take turns, $quantum units at a\n'
            '    time, until every job is done."""\n'
            "    left = list(durations)\n"
            "    slices = []\n"
            "    while sum(left) > 0:\n"
            "        for i in range(len(left)):\n"
            "            if left[i] > 0:\n"
            "                slices.append(i)\n"
            "                left[i] = max(left[i] - $quantum, 0)\n"
            "    return slices",
            _rr,
            public=[("two", ([4, 1],))],
            hidden=[([],), ([0, 0],), ([7],), ([1, 5, 2],)],
        ),
        Func(
            "cool",
            "main",
            "def $cool(names):\n"
            '    """Time slots used to run the jobs in order, one per slot, when a job must wait\n'
            "    at least $gap empty or other slots after its previous run (it is delayed to the\n"
            '    first slot that allows it)."""\n'
            "    last = {}\n"
            "    slot = 0\n"
            "    for name in names:\n"
            "        if name in last and slot - last[name] <= $gap:\n"
            "            slot = last[name] + $gap + 1\n"
            "        last[name] = slot\n"
            "        slot += 1\n"
            "    return slot",
            _cool,
            public=[
                ("repeat", (["a", "a"],)),
                ("spaced", (["a", "b", "c", "a"],)),
                ("mixed", (["a", "b", "a", "b"],)),
            ],
            hidden=[
                ([],),
                (["x"],),
                (["x", "y"],),
                (["x", "x", "x"],),
                (["x", "y", "x"],),
                (["x", "y", "z", "w", "x"],),
                (["p", "p", "q", "q"],),
                (["a", "b", "b", "a"],),
            ],
        ),
    ),
)

TEMPLATES = (
    Template(
        id="jobs1",
        family=FAMILY,
        workflow="W1",
        target="pick",
        concept="highest-priority job with an earliest-first tie-break",
        bugs={
            "latest_tie": Bug(edits=(("prio > queue[best][1]", "prio >= queue[best][1]"),)),
            "lowest": Bug(edits=(("prio > queue[best][1]", "prio < queue[best][1]"),)),
            "empty_zero": Bug(edits=(("    best = -1\n", "    best = 0\n"),)),
        },
        spec=(
            "pick(queue) returns the index of the job to run next. queue is a list of (name, "
            "priority) pairs. Return -1 for an empty queue; otherwise the smallest index whose "
            "priority equals the largest priority in the queue."
        ),
        domain="queue: list of 0..6 (str, int) tuples, priorities in 0..9.",
        returns="int",
        include=("order", "rr"),
    ),
    Template(
        id="jobs2",
        family=FAMILY,
        workflow="W4",
        target="cool",
        concept="slots needed with a cooldown between repeats",
        bugs={"stub_raise": Bug(stub="raise"), "stub_none": Bug(stub="none")},
        spec=(
            "cool(names) simulates running jobs in list order, one per time slot, slots "
            "numbered from 0. A job may run in slot s only if it has not run before, or s - t > "
            "params['gap'] where t is the slot of the most recent earlier run of the same name. "
            "Each job runs in the first slot that satisfies the rule and is at least the lower "
            "bound: 0 for the first job, otherwise the slot of the previous list entry + 1. "
            "Return the number of slots used, i.e. the last job's slot + 1 (0 for no "
            "jobs)."
        ),
        domain="names: list of 0..8 short str job names (repeats allowed).",
        returns="int",
        include=("rr", "finish"),
    ),
    Template(
        id="jobs3",
        family=FAMILY,
        workflow="W3",
        target="waits",
        concept="waiting times of jobs run back to back",
        bugs={
            "after": Bug(
                edits=(
                    (
                        "        out.append(clock)\n        clock += d",
                        "        clock += d\n        out.append(clock)",
                    ),
                )
            ),
            "start_one": Bug(edits=(("    clock = 0\n", "    clock = 1\n"),)),
            "no_sum": Bug(edits=(("clock += d", "clock = d"),)),
        },
        spec=(
            "waits(durations) returns a list the same length as durations: element i is the "
            "sum of durations[0:i] (so the first job waits 0)."
        ),
        domain="durations: list of 0..8 ints in 0..20.",
        returns="list of int",
        include=("pick",),
        caller="finish",
    ),
)
