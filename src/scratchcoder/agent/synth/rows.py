"""Edit and repair rows from the real harness loop, driven by a scripted policy.

For one instance with ``k`` mutants (wrong attempts), :func:`build_instance` runs
:func:`scratchcoder.agent.loop.run_instance` with a policy that replies, turn by turn, with
mutant 1, ..., mutant k and then the reference function. The loop does what it always does:

1. **F0**: the original repository's public tests (sandbox); the condensed failure is the
   turn-1 observation;
2. for each mutant: apply it, run the public tests (**F1..Fk**, sandbox); a mutant that
   fails them gives the next turn's observation and changes-so-far block;
3. apply the reference and run the public tests (**F(k+1)**, sandbox);
4. **G**: one grading call whose test string holds the hidden correctness checks, the hidden
   preservation checks and the independent oracle's differential and property checks
   (``grade_checks=("correctness",)`` with that combined script; the verdict is the
   supervisor's typed status).

Then **O**: the original repository against the hidden correctness checks (sandbox), which must
fail. That is ``k + 4`` calls per instance (6 for the default ``k = 2``), each counted by the
ledger. O runs last, only when everything before it was accepted, so a rejected instance
does not spend it.

The policy records every prompt the loop built. The rows are:

- **edit row**: the turn-1 prompt -> the reference function;
- **repair row j**: the turn-(j+1) prompt (the workspace holds mutant j, the changes-so-far
  block shows its diff and the observation is its real failure) -> the reference function.

A prompt does not depend on the turn number, so the repair row of mutant j is the prompt a
one-mistake run with mutant j would have built.

Each row is rendered with chat-v1 as ``tools/sft_data.py`` renders code rows: the user turn is
the prompt text (``generation_prompt``), and the assistant turn is ``<|python_start|>`` + the
function + ``<|python_end|>`` + ``<|assistant_end|>``, all assistant tokens under the loss
mask. The row's user-turn token IDs must equal the harness prompt IDs exactly, and the reply
must be what the harness parser accepts: the first closed python block, whose one function
the editor applies as the reference text.

Acceptance (all must hold, else the instance is rejected with its reasons and gives no
rows): every loop call resolved; every mutant was applied and its public run failed; the
reference was applied and its public run passed; the grading call passed; the original
failed the correctness checks; every prompt showed the target; every repair prompt holds a
failing observation and the changes-so-far block; for a traceback task, the original's
observation shows the exception line and the raising function the task names.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from scratchcoder.agent import execution, loop, workspace
from scratchcoder.agent.cases import Case
from scratchcoder.agent.prompt import PREVIOUS_HEADER, Prompt
from scratchcoder.agent.synth import GENERATOR_VERSION, independent, mutation
from scratchcoder.data.chat import (
    ASSISTANT_END,
    PYTHON_END,
    PYTHON_START,
    Message,
    Rendered,
    TextDecoder,
    TextEncoder,
    extract_program,
    generation_prompt,
    is_special,
    program_text,
)
from scratchcoder.data.sft import SFTRow, loss_tokens, row_json

SOURCES = {"edit": "agent-sft1-edit", "repair": "agent-sft1-repair"}
POLICY_NAME = "scripted-rows"
FAILED_PREFIX = "RESULT run_tests: failed"


class RowError(ValueError):
    pass


def calls_per_instance(mutants: int) -> int:
    """Sandbox calls one instance spends when nothing is retried: F0, one feedback run per
    mutant, the reference feedback run, the grading call and the original's check."""
    return mutants + 4


class ScriptedPolicy:
    """Replies with fixed code, one reply per turn, and keeps every prompt it is shown."""

    name = POLICY_NAME

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[Prompt] = []
        self.files: list[dict[str, str]] = []

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: Prompt,
        max_new_tokens: int,
        decoding: loop.Decoding,
        files: Mapping[str, str],
    ) -> loop.Proposal:
        self.prompts.append(prompt)
        self.files.append(dict(files))
        code = self.replies[turn - 1] if turn <= len(self.replies) else None
        return loop.Proposal(code, code or "", 0, "scripted")


def render_row(
    encoder: TextEncoder,
    decoder: TextDecoder,
    prompt: Prompt,
    code: str,
    *,
    case: Case,
    files: Mapping[str, str],
) -> Rendered:
    """The chat-v1 row ``prompt -> code``, with the checks that make it a harness row."""
    ids = generation_prompt(encoder, [Message("user", prompt.text)])
    if tuple(ids) != prompt.ids:
        raise RowError("the row's user turn does not encode to the harness prompt IDs")
    text = program_text(code)
    content = encoder.encode(text)
    if not content or any(is_special(token) for token in content):
        raise RowError("the reply must encode to ordinary tokens")
    reply = [PYTHON_START, *content, PYTHON_END, ASSISTANT_END]
    if extract_program(reply, decoder) != text:
        raise RowError("the harness would not extract the reply's block as written")
    path, _, name = case.targets[0].partition("::")
    shown = [
        workspace.functions(files[key.split("::")[0]], key.split("::")[0])[key.split("::")[1]]
        for key in prompt.shown
    ]
    edit = workspace.parse_reply(text, shown, files, case.allowed_files)
    if (edit.path, edit.name, edit.source) != (path, name, text):
        raise RowError("the harness parser would not apply the reply as the target")
    mask = [False] * len(ids) + [True] * len(reply)
    return Rendered(tuple([*ids, *reply]), tuple(mask))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _traceback_seen(observation: str, traceback: Mapping[str, str]) -> bool:
    lines = observation.split("\n")
    if traceback["error_line"] not in lines:
        return False
    frame = f"  in {traceback['raise_file']}:"
    name = f" {traceback['raise_function']}: "
    return any(line.startswith(frame) and name in line for line in lines)


@dataclasses.dataclass
class InstanceBuild:
    instance: str
    accepted: bool
    reasons: list[str]
    calls: int
    steps: list[dict[str, Any]]
    rows: list[SFTRow]
    meta: list[dict[str, Any]]

    def record(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "calls": self.calls,
            "steps": self.steps,
            "rows": [row.id for row in self.rows],
        }


def build_instance(
    case: Case,
    synth: Mapping[str, Any],
    oracle: independent.Oracle,
    encoder: TextEncoder,
    decoder: TextDecoder,
    *,
    execute: execution.Execute,
    trace_stream: TextIO | None = None,
    sleep: Callable[[float], None] | None = None,
    diagnose: bool = False,
) -> InstanceBuild:
    """Run one instance through the loop and, when every rule holds, render its rows."""
    mutants = [item["source"] for item in synth["mutants"]]
    k = len(mutants)
    if not 1 <= k or k + 1 > case.budgets.max_edit_turns or k + 2 > case.budgets.max_feedback_runs:
        raise RowError("the mutant count does not fit the case budgets")
    target_path, _, target_name = case.targets[0].partition("::")
    reference = case.reference_function(case.targets[0])
    module = target_path.removesuffix(".py")
    block = independent.block(
        oracle, module, target_name, synth["oracle_params"], int(synth["seed"]) % 2**31
    )
    grading = independent.grading_script(case.correctness, case.preservation, block)
    graded = dataclasses.replace(case, correctness=grading)
    policy = ScriptedPolicy([*mutants, reference])
    trace = loop.Trace(trace_stream)
    steps: list[dict[str, Any]] = []
    counted = [0]

    def counting(program: str, tests: str, mode: str) -> Any:
        counted[0] += 1
        return execute(program, tests, mode)

    result = loop.run_instance(
        graded,
        policy,
        loop.Condition(f"rows-{k}", k + 1),
        loop.GREEDY,
        encoder,
        execute=counting,
        trace=trace,
        sleep=sleep,
        grade_checks=("correctness",),
    )
    reasons: list[str] = []
    if not result.scored:
        reasons.append(f"unscored:{result.ended}")
    turns = result.turns
    for index, turn in enumerate(turns[:k], 1):
        steps.append({"step": f"mutant-{index}", "edit": turn.edit, "feedback": turn.feedback})
        if turn.edit != "applied":
            reasons.append(f"mutant_{index}_refused:{turn.edit}")
        elif turn.feedback == "passed":
            reasons.append(f"mutant_{index}_passed_public")
        elif turn.feedback != "failed":
            reasons.append(f"mutant_{index}_feedback:{turn.feedback}")
    if len(turns) == k + 1:
        last = turns[k]
        steps.append({"step": "reference", "edit": last.edit, "feedback": last.feedback})
        if last.edit != "applied" or last.feedback != "passed":
            reasons.append("reference_public_not_passed")
    elif not any(reason.endswith("_passed_public") for reason in reasons):
        reasons.append(f"turns:{len(turns)}:{result.ended}")
    steps.append({"step": "grade", "grade": dict(result.grade)})
    if result.grade.get("correctness") != "passed":
        reasons.append(f"grading:{result.grade.get('correctness', 'not_run')}")
    prompts = policy.prompts
    for index, prompt in enumerate(prompts):
        if case.targets[0] not in prompt.shown:
            reasons.append(f"turn_{index + 1}_target_not_shown")
        if index and (FAILED_PREFIX not in prompt.text or PREVIOUS_HEADER not in prompt.text):
            reasons.append(f"turn_{index + 1}_repair_context_missing")
    if synth.get("traceback") and not _traceback_seen(result.feedback_before, synth["traceback"]):
        reasons.append("traceback_not_observed")
    if diagnose and any(reason.startswith("grading:") for reason in reasons):
        files = case.reference_files()
        for label, checks in (
            ("diagnose-correctness", case.correctness),
            ("diagnose-preservation", case.preservation),
            ("diagnose-independent", block),
        ):
            run = execution.run_call(
                *execution.check_call(files, checks), execute=counting, sleep=sleep
            )
            steps.append({"step": label, "status": run.status, "resolved": run.resolved})
    if not reasons:
        run = execution.run_call(
            *execution.check_call(case.files, case.correctness), execute=counting, sleep=sleep
        )
        steps.append(
            {"step": "original-correctness", "status": run.status, "resolved": run.resolved}
        )
        if not run.resolved:
            reasons.append("unscored:original_correctness")
        elif run.passed:
            reasons.append("original_passes_correctness")
    rows: list[SFTRow] = []
    meta: list[dict[str, Any]] = []
    if not reasons:
        try:
            for index, prompt in enumerate(prompts):
                rendered = render_row(
                    encoder, decoder, prompt, reference, case=case, files=policy.files[index]
                )
                kind = "edit" if index == 0 else "repair"
                row_id = f"agent-sft1/{case.id}/{kind if index == 0 else f'repair-{index}'}"
                row = SFTRow.of(row_id, SOURCES[kind], rendered)
                rows.append(row)
                meta.append(
                    {
                        "id": row_id,
                        "kind": kind,
                        "instance": case.id,
                        "template_id": synth["template_id"],
                        "family": synth["family"],
                        "workflow_type": synth["workflow_type"],
                        "bug_kind": synth["bug_kind"],
                        "mutation_kind": synth["mutants"][index - 1]["kind"] if index else None,
                        "seed": synth["seed"],
                        "generator_version": GENERATOR_VERSION,
                        "turn": index + 1,
                        "prompt_sha256": prompt.sha256,
                        "prompt_tokens": prompt.tokens,
                        "truncations": list(prompt.truncations),
                        "loss_tokens": loss_tokens(row),
                        "row_sha256": row.sha256(),
                        "user_text": prompt.text,
                        "reply": program_text(reference),
                        "spec_sha256": case.spec_sha256,
                        "oracle_sha256": oracle.sha256,
                    }
                )
        except (RowError, workspace.EditRefused) as error:
            reasons.append(f"row:{error}")
            rows, meta = [], []
    return InstanceBuild(case.id, not reasons, reasons, counted[0], steps, rows, meta)


def run_controls(
    case: Case,
    synth: Mapping[str, Any],
    oracle: independent.Oracle,
    *,
    execute: execution.Execute,
    sleep: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    """The negative controls of one instance's template: each replaces the reference target
    with a trivial body and runs the full grading script once; each must not pass."""
    path, _, name = case.targets[0].partition("::")
    reference = case.reference_function(case.targets[0])
    block = independent.block(
        oracle, path.removesuffix(".py"), name, synth["oracle_params"], int(synth["seed"]) % 2**31
    )
    grading = independent.grading_script(case.correctness, case.preservation, block)
    rows = []
    for kind in mutation.CONTROL_KINDS:
        source = mutation.control(reference, kind, synth.get("control_constant"))
        files = workspace.apply_function(
            case.reference_files(), workspace.Edit(path, name, source), case.allowed_files
        )
        run = execution.run_call(
            *execution.check_call(files, grading), execute=execute, sleep=sleep
        )
        rows.append(
            {
                "instance": case.id,
                "template_id": synth["template_id"],
                "control": kind,
                "status": run.status,
                "resolved": run.resolved,
                "ok": run.resolved and not run.passed,
            }
        )
    return rows


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_rows(path: Path, rows: Sequence[SFTRow]) -> None:
    """Rows in the ``tools/sft_data.py`` row format (``id``, ``ids``, ``source``, ``spans``)."""
    append_jsonl(path, [row_json(row) for row in rows])
