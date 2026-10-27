"""Developer entry point; pipeline stages are added in later milestones."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scratchcoder.common.smoke import run_smoke


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch only implemented commands and return a process exit status."""
    parser = argparse.ArgumentParser(prog="scratchcoder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    smoke.add_argument("--output", type=Path)
    build = subparsers.add_parser("data-build")
    build.add_argument("--synthetic", action="store_true", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--vocab-size", type=int, default=512)
    tokenizer = subparsers.add_parser("tok-train")
    tokenizer.add_argument("--corpus", type=Path, nargs="+", required=True)
    tokenizer.add_argument("--output", type=Path, required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=512)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--sources", type=Path, required=True)
    train.add_argument("--tokenizer", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("runs"))
    train.add_argument("--run-id", required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--validation", type=Path)
    train.add_argument("--device", default="cpu", help="cpu (default), cuda or cuda:N")
    evaluation = subparsers.add_parser("eval")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--tokenizer", type=Path, required=True)
    evaluation.add_argument("--tasks", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--scorecard", type=Path, default=Path("runs/scorecard.jsonl"))
    evaluation.add_argument("--samples", type=int, default=1)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--max-new-tokens", type=int, default=128)
    evaluation.add_argument("--temperature", type=float, default=0.0)
    evaluation.add_argument("--prompt-mode", choices=("base", "instruction"), default="base")
    choices = subparsers.add_parser("eval-choice")
    choices.add_argument("--checkpoint", type=Path, required=True)
    choices.add_argument("--tokenizer", type=Path, required=True)
    choices.add_argument("--tasks", type=Path, required=True)
    choices.add_argument("--output", type=Path, required=True)
    choices.add_argument("--scorecard", type=Path, default=Path("runs/scorecard.jsonl"))
    choices.add_argument("--seed", type=int, default=0)
    validation = subparsers.add_parser("eval-validation")
    validation.add_argument("--checkpoint", type=Path, required=True)
    validation.add_argument("--tokenizer", type=Path, required=True)
    validation.add_argument("--sources", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--scorecard", type=Path, default=Path("runs/scorecard.jsonl"))
    validation.add_argument("--documents", type=int, required=True)
    validation.add_argument("--sequence-length", type=int)
    benchmark = subparsers.add_parser(
        "generation-benchmark",
        help="time batched generation on synthetic prompts and print JSON",
    )
    model_source = benchmark.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint", type=Path)
    model_source.add_argument("--config", type=Path, help="YAML or JSON with a model mapping")
    benchmark.add_argument("--tokenizer", type=Path, help="required with --checkpoint")
    benchmark.add_argument("--device", default="cpu", help="cpu (default), cuda or cuda:N")
    benchmark.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64, 128, 256])
    benchmark.add_argument("--prompt-lengths", type=int, nargs="+", default=[190])
    benchmark.add_argument("--max-new-tokens", type=int, default=256)
    benchmark.add_argument("--temperature", type=float, default=1.0)
    benchmark.add_argument("--top-k", type=int)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--repeats", type=int, default=1)
    rl_prescreen = subparsers.add_parser(
        "rl-prescreen",
        help="count RL candidates by parseability and constant baseline; runs no code",
    )
    rl_prescreen.add_argument("--tasks", type=Path, required=True, help="the pool's train.jsonl")
    rl_prescreen.add_argument("--candidates", type=Path, required=True)
    rl_prescreen.add_argument("--threshold", type=float, default=0.5)
    rl_screen = subparsers.add_parser(
        "rl-screen",
        help="score screening attempts in the Docker sandbox and write the keep-rule record",
    )
    rl_screen.add_argument("--tasks", type=Path, required=True, help="the pool's train.jsonl")
    rl_screen.add_argument("--candidates", type=Path, required=True)
    rl_screen.add_argument("--attempts", type=Path, required=True, help="Attempt JSON lines")
    rl_screen.add_argument("--output", type=Path, required=True)
    rl_screen.add_argument(
        "--timeout", type=float, required=True, help="seconds per batch of 256 attempts"
    )
    rl_screen.add_argument("--workers", type=int, default=8)
    rl_screen.add_argument("--threshold", type=float, default=0.5)
    rl_screen.add_argument("--cap", type=int, default=1000)
    rl_screen.add_argument("--floor", type=int, default=320)
    rl_screen.add_argument(
        "--require-header",
        action="store_true",
        help="refuse an attempts file without the Pod worker's header binding it to the tasks",
    )
    export = subparsers.add_parser(
        "pod-task-export",
        help="write a task file for a Pod with the hidden tests removed, and its record",
    )
    export.add_argument("--kind", choices=("pool", "candidates", "arc"), required=True)
    export.add_argument("--tasks", type=Path, help="pool: a split file; candidates: train.jsonl")
    export.add_argument("--candidates", type=Path, help="candidates: the RL candidate list")
    export.add_argument("--arc-directory", type=Path, help="arc: the pinned ARC-Easy Parquet files")
    export.add_argument("--output", type=Path, required=True, help="the new export file")
    export.add_argument("--record", type=Path, required=True, help="the new record file")
    rl_export = subparsers.add_parser(
        "rl-pod-export",
        help="write run 2's withheld RL and dev-subset task files and the facts file (Mac)",
    )
    rl_export.add_argument("--train-tasks", type=Path, required=True, help="pool train.jsonl")
    rl_export.add_argument("--kept", type=Path, required=True, help="the RL screening record")
    rl_export.add_argument("--dev-tasks", type=Path, required=True, help="pool dev.jsonl")
    rl_export.add_argument("--output", type=Path, required=True, help="a new directory")
    rl_bridge = subparsers.add_parser(
        "rl-bridge",
        help="serve one Pod's RL rewards and dev verdicts from the Mac over SSH",
    )
    rl_bridge.add_argument("--host", required=True, help="user@public-ip of the Pod")
    rl_bridge.add_argument("--port", type=int, required=True)
    rl_bridge.add_argument("--identity-file", type=Path, required=True)
    rl_bridge.add_argument("--known-hosts", type=Path, required=True)
    rl_bridge.add_argument("--remote-root", required=True, help="the Pod's bridge directory")
    rl_bridge.add_argument("--train-tasks", type=Path, required=True, help="pool train.jsonl")
    rl_bridge.add_argument("--kept", type=Path, required=True, help="the RL screening record")
    rl_bridge.add_argument("--dev-tasks", type=Path, required=True, help="pool dev.jsonl")
    rl_bridge.add_argument("--pod-reward-sha256", required=True, help="the Pod's RL task file")
    rl_bridge.add_argument("--pod-dev-sha256", required=True, help="the Pod's dev task file")
    rl_bridge.add_argument("--state", type=Path, required=True, help="the Mac's bridge state")
    rl_bridge.add_argument("--workers", type=int, default=8)
    rl_bridge.add_argument(
        "--reward-timeout",
        type=float,
        required=True,
        help="Mac-side bound on one reward request: the Pod's reward timeout less the allowance",
    )
    rl_bridge.add_argument(
        "--dev-timeout",
        type=float,
        required=True,
        help="Mac-side bound on one dev check: the Pod's dev timeout less the allowance",
    )
    rl_bridge.add_argument("--poll-seconds", type=float, default=2.0)
    rl_bridge.add_argument("--ssh-timeout", type=float, default=120.0)
    rl_bridge.add_argument("--ssh-retries", type=int, default=4)
    rl_bridge.add_argument(
        "--dotenv",
        type=Path,
        help="the .env file to check (default: .env in the current directory and the checkout)",
    )
    rl_bridge.add_argument(
        "--no-dotenv", action="store_true", help="skip the .env check: there is no .env file"
    )
    rl_bridge.add_argument("--image-id", help="the pinned sandbox image's local ID, checked")
    rl_bridge.add_argument("--max-polls", type=int)
    from scratchcoder.eval import pipeline_cli

    pipeline_cli.add_parsers(subparsers)
    args = parser.parse_args(argv)
    if args.command in pipeline_cli.COMMANDS:
        return pipeline_cli.run(args, parser)
    if args.command in {"rl-prescreen", "rl-screen"}:
        import json

        from scratchcoder.rl.command import prescreen_candidates, screen_attempts

        if args.command == "rl-prescreen":
            summary = prescreen_candidates(args.tasks, args.candidates, threshold=args.threshold)
        else:
            summary = screen_attempts(
                tasks_path=args.tasks,
                candidates_path=args.candidates,
                attempts_path=args.attempts,
                output=args.output,
                timeout=args.timeout,
                workers=args.workers,
                threshold=args.threshold,
                cap=args.cap,
                floor=args.floor,
                require_header=args.require_header,
            )
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return 0
    if args.command == "pod-task-export":
        import json

        from scratchcoder.eval.export import write_export

        record = write_export(
            args.kind,
            output=args.output,
            record=args.record,
            tasks_path=args.tasks,
            candidates_path=args.candidates,
            arc_directory=args.arc_directory,
        )
        printed = {key: record[key] for key in ("export", "output") if key in record}
        sys.stdout.write(json.dumps(printed, indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "rl-pod-export":
        import json

        from scratchcoder.rl.pod_export import write_rl_pod_export

        record = write_rl_pod_export(
            train_tasks=args.train_tasks,
            kept=args.kept,
            dev_tasks=args.dev_tasks,
            output=args.output,
        )
        printed = {
            name: {key: row[key] for key in ("name", "sha256", "bytes")}
            for name, row in record["files"].items()
        }
        sys.stdout.write(json.dumps(printed, indent=2, sort_keys=True) + "\n")
        return 0
    if args.command == "rl-bridge":
        from scratchcoder.rl.bridge_command import (
            BridgeEnvironmentError,
            resolve_dotenv,
            serve_command,
        )
        from scratchcoder.rl.bridge_mac import BridgeLeaseHeld

        try:
            dotenv = resolve_dotenv(args.dotenv, disabled=args.no_dotenv)
            return serve_command(
                host=args.host,
                port=args.port,
                identity_file=args.identity_file,
                known_hosts=args.known_hosts,
                remote_root=args.remote_root,
                train_tasks=args.train_tasks,
                kept=args.kept,
                dev_tasks=args.dev_tasks,
                pod_reward_sha256=args.pod_reward_sha256,
                pod_dev_sha256=args.pod_dev_sha256,
                state=args.state,
                workers=args.workers,
                reward_timeout=args.reward_timeout,
                dev_timeout=args.dev_timeout,
                poll_seconds=args.poll_seconds,
                ssh_timeout=args.ssh_timeout,
                ssh_retries=args.ssh_retries,
                dotenv=dotenv,
                image_id=args.image_id,
                max_polls=args.max_polls,
            )
        except BridgeEnvironmentError as error:
            sys.stderr.write(f"rl-bridge: {error}\n")
            return 2
        except BridgeLeaseHeld as error:
            sys.stderr.write(f"rl-bridge: {error}\n")
            return 3
    if args.command == "generation-benchmark":
        import json

        from scratchcoder.engine.benchmark import run_generation_benchmark

        result = run_generation_benchmark(
            checkpoint=args.checkpoint,
            tokenizer=args.tokenizer,
            config=args.config,
            device=args.device,
            precision=args.precision,
            batch_sizes=args.batch_sizes,
            prompt_lengths=args.prompt_lengths,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            repeats=args.repeats,
        )
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return 0
    if args.command == "smoke":
        return run_smoke(sys.stdout, config_path=args.config, output_root=args.output)
    if args.command == "data-build":
        from scratchcoder.data.fixtures import prepare_fixture

        if args.output.exists():
            parser.error("output directory must not already exist")
        prepare_fixture(args.output, vocab_size=args.vocab_size)
        return 0
    if args.command == "tok-train":
        from scratchcoder.tokenizer.bpe import ByteBPE

        ByteBPE.train(
            (path.read_text(encoding="utf-8") for path in args.corpus), args.vocab_size
        ).save(args.output)
        return 0
    if args.command == "eval-validation":
        from scratchcoder.eval.validation_command import evaluate_validation_checkpoint

        evaluate_validation_checkpoint(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            sources_path=args.sources,
            output=args.output,
            scorecard=args.scorecard,
            documents=args.documents,
            sequence_length=args.sequence_length,
        )
        return 0
    if args.command == "eval-choice":
        from scratchcoder.eval.choice_command import evaluate_choice_checkpoint

        evaluate_choice_checkpoint(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            tasks_path=args.tasks,
            output=args.output,
            scorecard=args.scorecard,
            seed=args.seed,
        )
        return 0
    if args.command == "eval":
        from scratchcoder.eval.command import evaluate_checkpoint
        from scratchcoder.eval.harness import EvaluationConfig

        evaluate_checkpoint(
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            tasks_path=args.tasks,
            output=args.output,
            scorecard=args.scorecard,
            config=EvaluationConfig(
                samples=args.samples,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                prompt_mode=args.prompt_mode,
            ),
        )
        return 0
    from scratchcoder.train.command import train_from_files

    train_from_files(
        config_path=args.config,
        sources_path=args.sources,
        tokenizer_path=args.tokenizer,
        output_root=args.output,
        run_id=args.run_id,
        resume=args.resume,
        validation_path=args.validation,
        device=args.device,
    )
    return 0
