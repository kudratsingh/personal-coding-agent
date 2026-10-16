"""Evaluate the complete pinned ARC-Easy test split with an own-project checkpoint."""

import argparse
import json
from pathlib import Path

from scratchcoder.data.arc import arc_splits, read_arc_easy
from scratchcoder.data.manifest import download
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.choice_command import checkpoint_choice_scorer
from scratchcoder.eval.choices import evaluate_choices
from scratchcoder.eval.harness import save_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    if args.download:
        for specification in arc_splits().values():
            download(specification.manifest, args.data_directory / specification.filename)
    tasks, metadata = read_arc_easy(args.data_directory)
    model, tokenizer = load_checkpoint(args.checkpoint, args.tokenizer)
    report = evaluate_choices(
        tasks,
        checkpoint_choice_scorer(model, tokenizer),
        checkpoint_hash=file_sha256(args.checkpoint),
        tokenizer_hash=tokenizer.sha256,
    )
    report["benchmark"] = dict(metadata, model_context=model.config.context)
    save_report(report, args.output, scorecard=args.output.parent / "scorecard.jsonl")
    # Bounded lines survive log clients that omit very long single-line records.
    print(
        "ARC_REPORT="
        + json.dumps(
            {key: value for key, value in report.items() if key != "per_task"}, sort_keys=True
        ),
        flush=True,
    )
    for row in report["per_task"]:
        print("ARC_TASK=" + json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
