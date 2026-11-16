"""The first development scorecard of the agent loop (``scratchcoder.agent``), on dev cases only.

Subcommands, all with explicit paths:

- ``register --cases DIR``: write ``DIR/registry.json`` (every case file with its SHA-256).
  Refuses to replace an existing registry. Runs nothing.
- ``plan --cases DIR --tokenizer PATH``: load and statically validate the cases, and print
  each case's turn-1 prompt size without its observation, with the three conditions of
  ``scratchcoder.agent.prompt.fit_bound``: the worst-case prompt of any turn (the snippet
  block at its full budget with a separator for each further snippet a turn could show, both
  feedback blocks at theirs, plus their separators) is within
  the 700-token prompt budget; that prompt plus 1.2 x the whole reference fix fits the 1,024
  context; and 1.2 x the largest reference function plus its two block tokens fits the
  324-token generation cap. Runs nothing.
- ``run``: everything that executes. It refuses to start without ``--receipt``, a passing
  execution-window receipt written by ``tools.mac_window_check exec`` for this very command
  in the last 30 minutes (``tools.admission_check``'s receipt rule), and without a new
  ``--out`` directory. Then, in order:

  1. validates every case in the sandbox (``scratchcoder.agent.validation``); an invalid case
     is recorded and not scored;
  2. runs the reference-fix oracle through the full loop (repair-3) on every valid case, a
     harness check reported apart from model rows;
  3. loads the checkpoint (its SHA-256 must equal ``--checkpoint-sha256``) on the CPU and runs
     every valid case under each condition (``single``, ``repair-3``) with greedy decoding and
     ``--seeds`` sampled seeds, at batch 1, in canonical order (case, decoding, condition);
  4. writes one JSONL trace per instance (``traces/``), one review packet and diff per
     instance (``packets/``), ``validation.json``, ``scorecard.json`` and ``scorecard.md``.

  ``--prefill-name`` (a harness aid, off by default) starts every model reply with a python
  block and ``def <name>(`` for the prompt's nearest shown function; the oracle rows never
  use it. After the run, the traces are read back to report how often the prefilled name was
  a target's (reporting only).

- ``compare --runs DIR ... --out-md PATH --out-json PATH``: one table across runs that share
  everything but their configuration (``scratchcoder.agent.scorecard.comparison``).

Case IDs are the identity of every output: they are checked at registry load
(``scratchcoder.agent.cases.check_case_ids``), and a scorecard or comparison refuses
results that would merge two cases.

  Model-written code runs only inside the sandbox, through
  ``scratchcoder.eval.harness.isolated_execution``; the host only parses it.

Held-out locations and sealed split names are refused before anything is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import execution, loop, prompt, scorecard, validation
from scratchcoder.agent.fit import plan_case
from scratchcoder.agent.policy import AGENT_SEED, Generate, ModelPolicy, OraclePolicy
from scratchcoder.agent.recovery import MODES as RECOVERY_MODES
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import PROTOCOL, TextDecoder, TextEncoder
from tools.admission_check import _receipt_image
from tools.verifier_scan import SealedPathError, guard

ModelLoader = Callable[[Path, Path, int], tuple[Generate, TextEncoder, TextDecoder, int]]
"""(checkpoint, tokenizer, threads) -> (generate, encoder, decoder, threads in use)."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_heldout(paths: Sequence[Path]) -> None:
    for path in paths:
        if any(agent_cases.forbidden_part(part) for part in path.resolve().parts):
            raise SealedPathError(f"refusing {path}: held-out agent cases are sealed")


def _tokenizer(path: Path) -> tuple[TextEncoder, TextDecoder]:
    from scratchcoder.tokenizer.bpe import ByteBPE
    from scratchcoder.tokenizer.bulk import FastEncoder

    tokenizer = ByteBPE.load(path)
    return FastEncoder(tokenizer), tokenizer


def load_model(
    checkpoint: Path, tokenizer: Path, threads: int
) -> tuple[Generate, TextEncoder, TextDecoder, int]:
    import torch

    from scratchcoder.agent.policy import model_generate
    from scratchcoder.eval.checkpoint import load_checkpoint
    from scratchcoder.tokenizer.bulk import FastEncoder

    model, bpe = load_checkpoint(checkpoint, tokenizer, device="cpu")
    torch.set_num_threads(threads)
    return model_generate(model), FastEncoder(bpe), bpe, torch.get_num_threads()


def _under(directory: Path, name: str) -> Path:
    """``directory / name``, refused unless it is a direct child of ``directory``."""
    path = directory / name
    if (
        "/" in name
        or "\\" in name
        or name in {"", ".", ".."}
        or (path.resolve().parent != directory.resolve())
    ):
        raise ValueError(f"output name escapes {directory}: {name!r}")
    return path


def config_of(args: argparse.Namespace) -> dict[str, Any]:
    """The named agent configuration of a run (recorded in the scorecard identity). The
    defaults are the first development run's configuration."""
    return {
        "prompt": args.prompt,
        "sampled_temperature": args.sampled_temperature,
        "sampled_top_p": args.sampled_top_p,
        "recovery": args.recovery,
        "prefill_name": args.prefill_name,
    }


def compare(args: argparse.Namespace) -> int:
    for path in (args.out_md, args.out_json):
        if path.exists():
            print(f"refused: {path} exists", file=sys.stderr)
            return 2
    cards = []
    for run_dir in args.runs:
        card = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
        cards.append((run_dir.name, card))
    try:
        table = scorecard.comparison(cards, vary_checkpoint=args.vary_checkpoint)
    except agent_cases.CaseError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if not table["comparable"] and not args.allow_noncomparable:
        for row in table["rows"]:
            if not row["comparable"]:
                print(
                    f"refused: {row['run']} is not comparable with the baseline: "
                    f"{', '.join(row['noncomparable_fields'])}",
                    file=sys.stderr,
                )
        return 2
    atomic_json(args.out_json, table)
    args.out_md.write_text(scorecard.comparison_markdown(table), encoding="utf-8")
    print(json.dumps({"rows": len(table["rows"]), "markdown": str(args.out_md)}))
    return 0


def register(args: argparse.Namespace) -> int:
    target = args.cases / agent_cases.REGISTRY_NAME
    if target.exists():
        print(f"refused: {target} exists", file=sys.stderr)
        return 2
    registry = agent_cases.build_registry(args.cases)
    target.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": [case["id"] for case in registry["cases"]]}))
    return 0


def plan(args: argparse.Namespace) -> int:
    encoder, _ = _tokenizer(args.tokenizer)
    rows = [plan_case(case, encoder) for case in agent_cases.load_cases(args.cases)]
    print(json.dumps({"registry_sha256": agent_cases.registry_sha256(args.cases), "cases": rows}))
    return 0 if all(row["fits"] and row["targets_shown"] for row in rows) else 1


def _decodings(args: argparse.Namespace) -> list[loop.Decoding]:
    found = [loop.GREEDY] if args.greedy else []
    found += [
        loop.Decoding(f"sample-{index}", args.sampled_temperature, args.sampled_top_p, index)
        for index in range(args.seeds)
    ]
    return found


def run(
    args: argparse.Namespace,
    *,
    execute: execution.Execute,
    model_loader: ModelLoader,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    try:
        image_id = _receipt_image(args.receipt, datetime.now(UTC), args.argv)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out: Path = args.out
    if out.exists():
        print(f"refused: {out} exists; every run writes a new directory", file=sys.stderr)
        return 2
    try:
        cases = agent_cases.load_cases(args.cases, only=set(args.case) if args.case else None)
    except (agent_cases.CaseError, OSError, ValueError, KeyError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    checkpoint_sha = _sha256_file(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        print("refused: the checkpoint's SHA-256 differs from --checkpoint-sha256", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out / "traces").mkdir()
    (out / "packets").mkdir()
    started = utc_now()
    reports = [validation.validate_case(case, execute=execute) for case in cases]
    atomic_json(out / "validation.json", {"cases": reports})
    valid = [case for case, report in zip(cases, reports, strict=True) if report["valid"]]
    results: list[loop.InstanceResult] = []
    names: set[str] = set()
    traced: list[tuple[str, Path]] = []

    def instance(
        case: agent_cases.Case,
        policy: loop.Policy,
        condition: loop.Condition,
        decoding: loop.Decoding,
        encoder: TextEncoder,
        prefill_name: bool = False,
    ) -> None:
        name = f"{case.id}--{policy.name}--{condition.name}--{decoding.label}"
        if name.casefold() in names:  # Registry load already refuses colliding case IDs.
            raise agent_cases.CaseError(f"two instances would share the output name {name}")
        names.add(name.casefold())
        trace_path = _under(out / "traces", f"{name}.jsonl")
        _under(out / "packets", name)
        with trace_path.open("x", encoding="utf-8") as stream:
            result = loop.run_instance(
                case,
                policy,
                condition,
                decoding,
                encoder,
                execute=execute,
                trace=loop.Trace(stream),
                clock=clock,
                layout=args.prompt,
                prefill_name=prefill_name,
            )
        loop.write_instance(out / "packets" / name, result, case, str(trace_path.relative_to(out)))
        results.append(result)
        traced.append((f"{result.policy}/{result.condition}", trace_path))
        print(
            json.dumps(
                {
                    "instance": name,
                    "solved": result.solved,
                    "scored": result.scored,
                    "failure_class": result.failure_class,
                    "turns": len(result.turns),
                    "wall_seconds": round(result.wall_seconds, 2),
                }
            ),
            flush=True,
        )

    generate, encoder, decoder, threads = model_loader(
        args.checkpoint, args.tokenizer, args.threads
    )
    conditions = [loop.CONDITIONS[name] for name in args.conditions]
    if not args.skip_oracle:  # A harness check, through the same encoder as the model rows.
        for case in valid:
            instance(case, OraclePolicy(), loop.REPAIR_3, loop.GREEDY, encoder)
    model = ModelPolicy(generate, decoder, recovery=args.recovery)
    decodings = _decodings(args)
    for case in valid:
        for decoding in decodings:
            for condition in conditions:
                instance(case, model, condition, decoding, encoder, args.prefill_name)
    commit, dirty = git_state()
    identity = {
        "started_at": started,
        "finished_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer_sha256": _sha256_file(args.tokenizer),
        "cases_registry_sha256": agent_cases.registry_sha256(args.cases),
        "cases": [case.id for case in cases],
        "valid_cases": [case.id for case in valid],
        "runtime_image_id": image_id,
        "supervisor_sha256": _sha256_file(
            Path(execution.__file__).parents[1] / "sandbox" / "program.py"
        ),
        "protocol": PROTOCOL,
        "config": config_of(args),
        "conditions": [asdict(condition) for condition in conditions],
        "decodings": [asdict(decoding) for decoding in decodings],
        "seed_base": AGENT_SEED,
        "batch_size": 1,
        "precision": "float32",
        "device": "cpu",
        "threads": threads,
        "budgets": {
            "prompt": prompt.PROMPT_BUDGET,
            "blocks": prompt.BUDGETS,
            "function": prompt.FUNCTION_BUDGET,
            "generation_cap": prompt.GENERATION_CAP,
            "context": prompt.CONTEXT,
        },
        "generation_cache_hits": sum(turn.cached for r in results for turn in r.turns),
    }
    report = None
    if args.prefill_name:  # Reporting only: read back from the traces once every instance ran.
        report = scorecard.prefill_report(
            (
                (label, [json.loads(line) for line in path.read_text("utf-8").splitlines()])
                for label, path in traced
                if label.startswith("model/")
            ),
            {case.id: case.targets for case in valid},
        )
    card = scorecard.build(results, identity, prefill=report)
    atomic_json(out / "scorecard.json", card)
    (out / "scorecard.md").write_text(scorecard.markdown(card), encoding="utf-8")
    print(json.dumps({"scorecard": str(out / "scorecard.json")}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--cases", type=Path, required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--cases", type=Path, required=True)
    plan_parser.add_argument("--tokenizer", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--cases", type=Path, required=True)
    run_parser.add_argument("--tokenizer", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--checkpoint-sha256", required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--receipt", type=Path, required=True)
    run_parser.add_argument("--seeds", type=int, default=5, help="sampled seeds per case")
    run_parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    run_parser.add_argument(
        "--sampled-temperature",
        "--temperature",
        dest="sampled_temperature",
        type=float,
        default=1.0,
        help="temperature of the sampled rows (greedy rows are unchanged)",
    )
    run_parser.add_argument(
        "--sampled-top-p", "--top-p", dest="sampled_top_p", type=float, default=1.0
    )
    run_parser.add_argument("--prompt", choices=prompt.LAYOUTS, default="harness")
    run_parser.add_argument("--recovery", choices=RECOVERY_MODES, default="off")
    run_parser.add_argument(
        "--prefill-name",
        action="store_true",
        help="harness aid: start each model reply with `def <name>(` of the nearest shown "
        "function (recorded apart from generated tokens; never credited as task-following)",
    )
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--out-md", type=Path, required=True)
    compare_parser.add_argument("--out-json", type=Path, required=True)
    compare_parser.add_argument(
        "--allow-noncomparable",
        action="store_true",
        help="write the table even when runs differ outside their configuration (flagged)",
    )
    compare_parser.add_argument(
        "--vary-checkpoint",
        action="store_true",
        help="compare checkpoints: the checkpoint may differ, as the one differing factor",
    )
    run_parser.add_argument(
        "--conditions", nargs="+", default=["single", "repair-3"], choices=sorted(loop.CONDITIONS)
    )
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--case", action="append", help="only these case IDs (repeatable)")
    run_parser.add_argument("--skip-oracle", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: execution.Execute | None = None,
    model_loader: ModelLoader | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    paths = [getattr(args, name, None) for name in ("cases", "tokenizer", "checkpoint")]
    if args.command == "compare":
        paths += [*args.runs, args.out_md, args.out_json]
    if args.command == "run":
        paths += [args.out, args.receipt]
        if args.seeds < 0 or not args.greedy and args.seeds == 0:
            parser.error("run needs greedy decoding or at least one sampled seed")
        if (
            not 0 < args.sampled_temperature <= 2
            or not 0 < args.sampled_top_p <= 1
            or args.threads < 1
        ):
            parser.error("invalid temperature, top-p or thread count")
    present = [path for path in paths if path is not None]
    try:
        _refuse_heldout(present)
        guard(present)
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    if args.command == "compare":
        return compare(args)
    if args.command == "register":
        return register(args)
    if args.command == "plan":
        try:
            return plan(args)
        except agent_cases.CaseError as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
    from scratchcoder.eval.harness import isolated_execution

    return run(
        args,
        execute=isolated_execution if execute is None else execute,
        model_loader=load_model if model_loader is None else model_loader,
    )


if __name__ == "__main__":
    raise SystemExit(main())
