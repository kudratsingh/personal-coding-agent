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
    args = parser.parse_args(argv)
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
    from scratchcoder.train.command import train_from_files

    train_from_files(
        config_path=args.config,
        sources_path=args.sources,
        tokenizer_path=args.tokenizer,
        output_root=args.output,
        run_id=args.run_id,
        resume=args.resume,
    )
    return 0
