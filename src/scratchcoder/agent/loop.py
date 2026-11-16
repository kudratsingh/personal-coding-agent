"""The mode-H loop for one instance (case x policy x condition x decoding).

1. **Feedback 0.** Run the public tests on the original repository (sandbox).
2. **Locate.** From the failing tests, follow calls statically to the reachable functions in
   the allowed files (:func:`scratchcoder.agent.workspace.locate`).
3. **Prompt.** Task, repository map, the located functions as snippets, the changes so far
   (removed and added lines of every applied edit, against the original repository), the
   condensed failure; budgeted and counted exactly.
4. **Edit.** The policy writes one function. The host applies it as an AST span replacement
   of the same-named shown function, or refuses it (wrong name, no block, unparsable,
   extra code, a file that may not be edited).
5. **Feedback.** After an applied edit, run the public tests again. Stop when they pass;
   otherwise condense the failure and repeat from 2, up to the condition's edit turns and the
   case's budgets.
6. **Grade.** Run the hidden correctness and preservation checks on the final workspace. The
   instance is solved only if the public tests, the correctness checks and the preservation
   checks all have the supervisor status ``passed``. Hidden check sources never enter a
   prompt or the trace; only their statuses are recorded.
7. **Deliver.** A review packet: unified diff, touched files, feedback before and after,
   the per-turn summary and the static audit flags of every applied function.

A host failure that outlasts the retries makes the instance **unscored** (failure class
``grading``), never a pass or a fail.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from scratchcoder.agent import execution, workspace
from scratchcoder.agent.audit import audit_refusal, name_failure, reply_facts
from scratchcoder.agent.cases import Case
from scratchcoder.agent.prompt import (
    LAYOUTS,
    Blocks,
    ContextLimit,
    Prompt,
    Snippet,
    build_layout,
    name_prefill,
)
from scratchcoder.data.chat import TextEncoder
from scratchcoder.sandbox.reward import audit_attempt

FAILURE_CLASSES = (
    "grading",
    "tool_interface",
    "context",
    "formatting",
    "action_selection",
    "model_capability",
)
GRADE_CHECKS = ("correctness", "preservation")
"""The hidden check scripts of a case, graded in this order by default."""
MAX_TRACE_TEXT = 65536
"""Generations are at most 324 tokens, so this keeps every completion whole in the trace."""


@dataclass(frozen=True)
class Condition:
    name: str
    edit_turns: int


SINGLE = Condition("single", 1)
REPAIR_3 = Condition("repair-3", 3)
CONDITIONS = {SINGLE.name: SINGLE, REPAIR_3.name: REPAIR_3}


@dataclass(frozen=True)
class Decoding:
    label: str
    temperature: float
    top_p: float = 1.0
    index: int = 0
    """The sample index that, with the case and turn, sets the seed (ignored when greedy)."""


GREEDY = Decoding("greedy", 0.0)


@dataclass(frozen=True)
class Proposal:
    code: str | None
    """The first closed python block of the reply, or None."""
    text: str = ""
    """The decoded reply (bounded in the trace)."""
    generated_tokens: int = 0
    finish_reason: str = ""
    seconds: float = 0.0
    """Generation time spent for this proposal (0 for a cache hit)."""
    cached: bool = False
    cached_seconds: float = 0.0
    """For a cache hit, the time the reused generation took when it was made."""
    tokens: tuple[int, ...] = ()
    """The generated token IDs (for the refusal audit); empty for non-model policies."""
    closed_block: bool | None = None
    """Whether the reply had a closed python block; None means ``code is not None``."""
    recovery: dict[str, Any] | None = None
    """When ``code`` was recovered from an unclosed block: the rule and span kept."""
    prefill_tokens: int = 0
    """Reply tokens the harness wrote before generation (the name prefill); not counted in
    ``generated_tokens``, but part of ``tokens`` and ``text``."""
    prefill_text: str = ""


class Policy(Protocol):
    name: str

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: Prompt,
        max_new_tokens: int,
        decoding: Decoding,
        files: Mapping[str, str],
    ) -> Proposal: ...


class Trace:
    """One JSON object per line; to a stream, or kept in memory when ``stream`` is None."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, **fields}
        self.events.append(row)
        if self.stream is not None:
            self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.stream.flush()


@dataclass
class Turn:
    turn: int
    prompt_tokens: int
    block_tokens: dict[str, int]
    shown: list[str]
    truncations: list[str]
    generation_cap: int
    generated_tokens: int = 0
    finish_reason: str = ""
    generation_seconds: float = 0.0
    cached: bool = False
    """The generation was reused from an identical earlier request (same prompt and seed)."""
    cached_seconds: float = 0.0
    """For a reused generation, the time it took when it was first made."""
    closed_block: bool = False
    recovered: bool = False
    recovery: dict[str, Any] | None = None
    parseable: bool = False
    defined: list[str] = field(default_factory=list)
    """Top-level function names the reply's code defines."""
    shown_defined: list[str] = field(default_factory=list)
    """Those of them that were shown in the prompt."""
    target_defined: bool = False
    """The reply defines exactly one shown function, and it is a target of the case."""
    name_failure: str = ""
    """For an ``unknown_name`` refusal, one of :data:`scratchcoder.agent.audit.NAME_FAILURES`."""
    audit: list[str] = field(default_factory=list)
    """Harness-defect flags: contradictions between the refusal and the reply rules."""
    edit: str = ""
    """``applied`` or the refusal reason."""
    function: str = ""
    feedback: str = ""
    """The feedback run's status after an applied edit."""
    prefill_function: str = ""
    """With ``--prefill-name``: the shown function (``shown[0]``) whose name the harness wrote
    before generation; empty when off, when the prompt showed nothing, or when the prefill
    was skipped (``prefill_skipped``)."""
    prefill_text: str = ""
    prefill_tokens: int = 0
    """Tokens the harness wrote (not in ``generated_tokens``; they count against the context)."""
    prefill_skipped: str = ""
    """With ``--prefill-name``: why this turn got no prefill (:data:`prompt.PREFILL_SKIPS`)."""


@dataclass
class InstanceResult:
    case_id: str
    policy: str
    condition: str
    decoding: str
    scored: bool = True
    solved: bool = False
    failure_class: str | None = None
    ended: str = ""
    turns: list[Turn] = field(default_factory=list)
    grade: dict[str, str] = field(default_factory=dict)
    target_located: bool = False
    targets_touched: bool = False
    audit_flags: list[str] = field(default_factory=list)
    harness_defects: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)
    diff: str = ""
    feedback_before: str = ""
    feedback_after: str = ""
    generated_tokens: int = 0
    generation_seconds: float = 0.0
    """Generation time actually spent by this instance (cache hits add nothing)."""
    cached_generation_seconds: float = 0.0
    """Generation time of reused generations, attributed to this instance as well, so that
    ``generation_seconds + cached_generation_seconds`` is comparable across conditions."""
    sandbox_seconds: float = 0.0
    sandbox_calls: int = 0
    host_failures: int = 0
    wall_seconds: float = 0.0

    @property
    def edits_applied(self) -> int:
        return sum(turn.edit == "applied" for turn in self.turns)

    def summary(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("diff")
        row["edits_applied"] = self.edits_applied
        row["turns_used"] = len(self.turns)
        return row


class _Unresolved(Exception):
    pass


def classify(result: InstanceResult) -> str | None:
    """The failure class of a finished instance, or None when it is solved."""
    if not result.scored:
        return "grading"
    if result.solved:
        return None
    if result.ended == "context" and not result.edits_applied:
        return "context"
    if not result.target_located:
        return "context"
    if not result.edits_applied:
        refusals = {turn.edit for turn in result.turns}
        if refusals & set(workspace.INTERFACE_REFUSALS):
            return "tool_interface"
        return "formatting"
    if not result.targets_touched:
        return "action_selection"
    return "model_capability"


def _previous(original: Mapping[str, str], files: Mapping[str, str]) -> list[str]:
    """The cumulative change: the current workspace against the original repository."""
    lines: list[str] = []
    for path in workspace.touched(original, files):
        lines.append(f"{path}:")
        lines.extend(workspace.changed_lines(original[path], files[path]))
    return lines


def public_test_sources(case: Case, files: Mapping[str, str], failing: list[str]) -> list[str]:
    """The source of each failing public test function (all of them when none is known), in
    file order; public material only."""
    sources: list[str] = []
    for path in case.public_tests:
        found = workspace.functions(files[path], path)
        names = [name for name in found if name.startswith("test_")]
        chosen = [name for name in names if name in failing] or ([] if failing else names)
        sources.extend(workspace.function_source(files[path], found[name]) for name in chosen)
    if failing and not sources:
        return public_test_sources(case, files, [])
    return sources


def blocks_for(
    case: Case,
    files: Mapping[str, str],
    candidates: list[workspace.Candidate],
    observation: str,
    original: Mapping[str, str],
    tests: list[str] | None = None,
) -> Blocks:
    """The prompt blocks of one turn (no hidden material: task, map, shown code, feedback)."""
    snippets = [
        Snippet(
            item.function.key,
            workspace.function_source(
                files[item.function.path],
                workspace.functions(files[item.function.path], item.function.path)[
                    item.function.name
                ],
            ),
        )
        for item in candidates
    ]
    return Blocks(
        task=case.task,
        map=workspace.repository_map(files),
        snippets=snippets,
        observation=observation,
        previous=_previous(original, files),
        tests=tests or (),
    )


def run_instance(
    case: Case,
    policy: Policy,
    condition: Condition,
    decoding: Decoding,
    encoder: TextEncoder,
    *,
    execute: execution.Execute,
    trace: Trace,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] | None = None,
    layout: str = "harness",
    prefill_name: bool = False,
    grade_checks: tuple[str, ...] = GRADE_CHECKS,
) -> InstanceResult:
    """Run one instance. ``layout`` is the prompt layout (:data:`prompt.LAYOUTS`); the
    training layout shows the failing public tests and, on repair turns only, the feedback.

    ``prefill_name`` (a harness aid, ``--prefill-name``) starts every reply with a python block
    and ``def <name>(`` for the nearest shown function (:func:`prompt.name_prefill`, computed
    from the prompt alone; not when that function is decorated, which the turn records). Its
    tokens count against the context, not the generated-token budget, and are recorded apart
    from generated tokens. Off, nothing changes.

    ``grade_checks`` names the hidden check scripts graded, one sandbox call each, in order
    (default: correctness, then preservation). The synthetic-data builder passes only
    ``("correctness",)`` with a case whose correctness script already holds every check it
    grades, so one call decides the verdict; the feedback calls are unchanged."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown prompt layout {layout!r}")
    if not grade_checks or len(set(grade_checks)) != len(grade_checks):
        raise ValueError("grade_checks must name distinct check scripts")
    if any(kind not in GRADE_CHECKS for kind in grade_checks):
        raise ValueError(f"grade_checks must be among {GRADE_CHECKS}")
    started = clock()
    budgets = case.budgets
    result = InstanceResult(case.id, policy.name, condition.name, decoding.label)
    original = dict(case.files)
    files = dict(original)
    feedback_runs = 0
    trace.emit(
        "start",
        case=case.id,
        policy=policy.name,
        condition=condition.name,
        decoding=asdict(decoding),
        spec_sha256=case.spec_sha256,
        budgets=asdict(budgets),
    )

    def sandbox(kind: str, program: str, tests: str) -> execution.SandboxRun:
        run = execution.run_call(program, tests, execute=execute, sleep=sleep)
        result.sandbox_calls += 1
        result.sandbox_seconds += run.seconds
        result.host_failures += run.host_failures
        trace.emit(
            "sandbox",
            kind=kind,
            status=run.status,
            resolved=run.resolved,
            host_failures=run.host_failures,
            seconds=round(run.seconds, 4),
        )
        if not run.resolved:
            raise _Unresolved(kind)
        return run

    try:
        last = sandbox("feedback", *execution.public_call(files, case.public_tests))
        feedback_runs += 1
        observation = execution.condense(last, files)
        result.feedback_before = observation
        trace.emit("feedback", turn=0, status=last.status, observation=observation)
        refusal = ""
        turns = min(condition.edit_turns, budgets.max_edit_turns)
        for number in range(1, turns + 1):
            if last.passed:
                result.ended = "public_passed"
                break
            if clock() - started > budgets.max_wall_seconds:
                result.ended = "budget_wall"
                break
            if feedback_runs >= budgets.max_feedback_runs:
                result.ended = "budget_feedback"
                break
            remaining = budgets.max_generated_tokens - result.generated_tokens
            if remaining <= 0:
                result.ended = "budget_tokens"
                break
            failing = execution.failing_tests(execution.parse_report(last.stdout))
            candidates = workspace.locate(files, case.public_tests, failing, case.allowed_files)
            shown_text = observation if not refusal else f"EDIT refused: {refusal}\n{observation}"
            tests = None
            if layout == "training":
                tests = public_test_sources(case, files, failing)
                if number == 1:
                    shown_text = ""
            try:
                blocks = blocks_for(case, files, candidates, shown_text, original, tests)
                prompt = build_layout(encoder, blocks, layout)
            except ContextLimit as error:
                trace.emit("context_limit", turn=number, reason=str(error))
                result.ended = "context"
                break
            if all(target in prompt.shown for target in case.targets):
                result.target_located = True
            if prefill_name:
                prompt = name_prefill(prompt, encoder)
            cap = min(prompt.generation_cap(), remaining)
            turn = Turn(
                number,
                prompt.tokens,
                prompt.block_tokens,
                list(prompt.shown),
                list(prompt.truncations),
                cap,
            )
            result.turns.append(turn)
            prefill_fields: dict[str, Any] = {}
            if prefill_name:
                turn.prefill_function = prompt.shown[0] if prompt.prefill else ""
                turn.prefill_text = prompt.prefill_text
                turn.prefill_tokens = len(prompt.prefill)
                turn.prefill_skipped = prompt.prefill_skipped
                prefill_fields = {
                    "prefill_function": turn.prefill_function,
                    "prefill_text": turn.prefill_text,
                    "prefill_tokens": turn.prefill_tokens,
                    "prefill_skipped": turn.prefill_skipped,
                }
            trace.emit(
                "prompt",
                turn=number,
                prompt_sha256=prompt.sha256,
                tokens=prompt.tokens,
                block_tokens=prompt.block_tokens,
                shown=list(prompt.shown),
                truncations=list(prompt.truncations),
                generation_cap=cap,
                text=prompt.text,
                **prefill_fields,
            )
            proposal = policy.propose(case, number, prompt, cap, decoding, files)
            if proposal.prefill_tokens != turn.prefill_tokens:
                raise ValueError("the policy did not start its reply with the prompt's prefill")
            turn.generated_tokens = proposal.generated_tokens
            turn.finish_reason = proposal.finish_reason
            turn.generation_seconds = proposal.seconds
            turn.cached = proposal.cached
            turn.cached_seconds = proposal.cached_seconds
            turn.recovered = proposal.recovery is not None
            turn.recovery = proposal.recovery
            turn.closed_block = (
                proposal.code is not None and not turn.recovered
                if proposal.closed_block is None
                else proposal.closed_block
            )
            result.cached_generation_seconds += proposal.cached_seconds
            result.generated_tokens += proposal.generated_tokens
            result.generation_seconds += proposal.seconds
            trace.emit(
                "generation",
                turn=number,
                tokens=proposal.generated_tokens,
                finish_reason=proposal.finish_reason,
                seconds=round(proposal.seconds, 4),
                cached=proposal.cached,
                cached_seconds=round(proposal.cached_seconds, 4),
                closed_block=turn.closed_block,
                recovered=turn.recovered,
                recovery=proposal.recovery,
                text=proposal.text[:MAX_TRACE_TEXT],
                **({"prefill_tokens": proposal.prefill_tokens} if prefill_name else {}),
            )
            shown = [item.function for item in candidates if item.function.key in prompt.shown]
            facts = reply_facts(proposal.code, shown)
            turn.parseable = facts["parseable"]
            turn.defined = facts["defined"]
            turn.shown_defined = facts["shown_defined"]
            if len(turn.shown_defined) == 1:
                keys = [item.key for item in shown if item.name == turn.shown_defined[0]]
                turn.target_defined = any(key in case.targets for key in keys)
            try:
                edit = workspace.parse_reply(proposal.code, shown, files, case.allowed_files)
                if turn.recovered and edit.source not in (proposal.code or ""):
                    raise workspace.EditRefused("unparsable", "recovered function not verbatim")
                files = workspace.apply_function(files, edit, case.allowed_files)
            except workspace.EditRefused as error:
                refusal = workspace.REFUSAL_TEXT[error.reason]
                turn.edit = error.reason
                if error.reason == "unknown_name":
                    turn.name_failure = name_failure(turn.defined, shown, files, case.targets)
                turn.audit = audit_refusal(
                    error.reason,
                    tokens=proposal.tokens,
                    code=proposal.code,
                    shown=shown,
                    files=files,
                    targets=case.targets,
                )
                result.harness_defects = sorted(set(result.harness_defects) | set(turn.audit))
                trace.emit(
                    "edit",
                    turn=number,
                    verdict="refused",
                    reason=error.reason,
                    name_failure=turn.name_failure,
                    audit=turn.audit,
                    parseable=turn.parseable,
                    defined=turn.defined,
                )
                continue
            refusal = ""
            key = f"{edit.path}::{edit.name}"
            turn.edit, turn.function = "applied", key
            flags = audit_attempt(edit.source)
            result.audit_flags = sorted(set(result.audit_flags) | set(flags))
            if key in case.targets:
                result.targets_touched = True
            trace.emit(
                "edit",
                turn=number,
                verdict="applied",
                function=key,
                dropped=list(edit.dropped),
                audit_flags=list(flags),
            )
            last = sandbox("feedback", *execution.public_call(files, case.public_tests))
            feedback_runs += 1
            observation = execution.condense(last, files)
            turn.feedback = last.status
            trace.emit("feedback", turn=number, status=last.status, observation=observation)
        else:
            result.ended = "public_passed" if last.passed else "turns_exhausted"
        result.feedback_after = observation
        if files == original:
            # The null patch is validated to fail the correctness checks before any run.
            result.grade = {"public": last.status, "correctness": "not_run_null_patch"}
            result.solved = False
        else:
            scripts = {"correctness": case.correctness, "preservation": case.preservation}
            graded = {
                kind: sandbox(kind, *execution.check_call(files, scripts[kind]))
                for kind in grade_checks
            }
            result.grade = {"public": last.status}
            result.grade.update({kind: run.status for kind, run in graded.items()})
            result.solved = last.passed and all(run.passed for run in graded.values())
    except _Unresolved as error:
        result.scored = False
        result.ended = f"unresolved_{error.args[0]}"
    result.touched = workspace.touched(original, files)
    result.diff = workspace.unified_diff(original, files)
    if result.turns and not result.target_located:
        result.harness_defects = sorted(set(result.harness_defects) | {"target_not_shown"})
    result.failure_class = classify(result)
    result.wall_seconds = clock() - started
    trace.emit(
        "end",
        scored=result.scored,
        solved=result.solved,
        failure_class=result.failure_class,
        ended=result.ended,
        grade=result.grade,
        touched=result.touched,
        generated_tokens=result.generated_tokens,
        generation_seconds=round(result.generation_seconds, 4),
        cached_generation_seconds=round(result.cached_generation_seconds, 4),
        sandbox_seconds=round(result.sandbox_seconds, 4),
        sandbox_calls=result.sandbox_calls,
        wall_seconds=round(result.wall_seconds, 4),
    )
    return result


def packet(result: InstanceResult, case: Case, trace_path: str) -> dict[str, Any]:
    """The review packet of one instance (no hidden check material)."""
    return {
        "schema": "agent-review-packet-v1",
        "case": case.id,
        "title": case.title,
        "task": case.task,
        "policy": result.policy,
        "condition": result.condition,
        "decoding": result.decoding,
        "scored": result.scored,
        "solved": result.solved,
        "failure_class": result.failure_class,
        "grade": result.grade,
        "touched_files": result.touched,
        "diff": result.diff,
        "feedback_before": result.feedback_before,
        "feedback_after": result.feedback_after,
        "turns": [asdict(turn) for turn in result.turns],
        "audit_flags": result.audit_flags,
        "trace": trace_path,
    }


def write_instance(directory: Path, result: InstanceResult, case: Case, trace_path: str) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "packet.json").write_text(
        json.dumps(packet(result, case, trace_path), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "diff.patch").write_text(result.diff, encoding="utf-8")
