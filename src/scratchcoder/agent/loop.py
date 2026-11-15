"""The mode-H loop for one instance (case x policy x condition x decoding).

1. **Feedback 0.** Run the public tests on the original repository (sandbox).
2. **Locate.** From the failing tests, follow calls statically to the reachable functions in
   the allowed files (:func:`scratchcoder.agent.workspace.locate`).
3. **Prompt.** Task, repository map, the located functions as snippets, the last edit's
   changed lines, the condensed failure; budgeted and counted exactly.
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
from scratchcoder.agent.cases import Case
from scratchcoder.agent.prompt import Blocks, ContextLimit, Prompt, Snippet, build
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
MAX_TRACE_TEXT = 4000


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
    cached: bool = False


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
    edit: str = ""
    """``applied`` or the refusal reason."""
    function: str = ""
    feedback: str = ""
    """The feedback run's status after an applied edit."""


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
    touched: list[str] = field(default_factory=list)
    diff: str = ""
    feedback_before: str = ""
    feedback_after: str = ""
    generated_tokens: int = 0
    generation_seconds: float = 0.0
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
    lines: list[str] = []
    for path in workspace.touched(original, files):
        lines.append(f"{path}:")
        lines.extend(workspace.changed_lines(original[path], files[path]))
    return lines


def blocks_for(
    case: Case,
    files: Mapping[str, str],
    candidates: list[workspace.Candidate],
    observation: str,
    original: Mapping[str, str],
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
) -> InstanceResult:
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
            try:
                prompt = build(encoder, blocks_for(case, files, candidates, shown_text, original))
            except ContextLimit as error:
                trace.emit("context_limit", turn=number, reason=str(error))
                result.ended = "context"
                break
            if all(target in prompt.shown for target in case.targets):
                result.target_located = True
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
            )
            proposal = policy.propose(case, number, prompt, cap, decoding, files)
            turn.generated_tokens = proposal.generated_tokens
            turn.finish_reason = proposal.finish_reason
            turn.generation_seconds = proposal.seconds
            turn.cached = proposal.cached
            result.generated_tokens += proposal.generated_tokens
            result.generation_seconds += proposal.seconds
            trace.emit(
                "generation",
                turn=number,
                tokens=proposal.generated_tokens,
                finish_reason=proposal.finish_reason,
                seconds=round(proposal.seconds, 4),
                cached=proposal.cached,
                closed_block=proposal.code is not None,
                text=proposal.text[:MAX_TRACE_TEXT],
            )
            shown = [item.function for item in candidates if item.function.key in prompt.shown]
            try:
                edit = workspace.parse_reply(proposal.code, shown, files, case.allowed_files)
                files = workspace.apply_function(files, edit, case.allowed_files)
            except workspace.EditRefused as error:
                refusal = workspace.REFUSAL_TEXT[error.reason]
                turn.edit = error.reason
                trace.emit("edit", turn=number, verdict="refused", reason=error.reason)
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
            correctness = sandbox("correctness", *execution.check_call(files, case.correctness))
            preservation = sandbox("preservation", *execution.check_call(files, case.preservation))
            result.grade = {
                "public": last.status,
                "correctness": correctness.status,
                "preservation": preservation.status,
            }
            result.solved = last.passed and correctness.passed and preservation.passed
    except _Unresolved as error:
        result.scored = False
        result.ended = f"unresolved_{error.args[0]}"
    result.touched = workspace.touched(original, files)
    result.diff = workspace.unified_diff(original, files)
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
