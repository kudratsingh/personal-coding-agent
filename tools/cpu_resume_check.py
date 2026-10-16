"""Bounded CPU checkpoint equivalence on authored token IDs, never a benchmark."""

import argparse
import gc
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import file_sha256
from scratchcoder.model.gpt import expected_parameters
from scratchcoder.train.trainer import Trainer


def configuration(size: str, precision: str, seed: int, steps: int) -> RunConfig:
    if size not in {"tiny", "nano"}:
        raise ValueError("size must be tiny or nano")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if type(steps) is not int or not 2 <= steps <= 16:
        raise ValueError("CPU check requires between2 and16 steps")
    model = (
        ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8)
        if size == "tiny"
        else ModelConfig(vocab_size=32768, width=256, layers=4, heads=4, context=256)
    )
    return RunConfig(
        model=model,
        train=TrainConfig(
            seed=seed,
            steps=steps,
            sequence_length=model.context,
            batch_size=2,
            accumulation_steps=2,
            precision=precision,
        ),
        data_mix={"web": 0.5, "python": 0.5},
    )


def assert_identical(actual: Any, reference: Any) -> None:
    """Recursive exact checkpoint comparison; no numeric tolerances are relaxed."""
    if isinstance(reference, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError("checkpoint tensor type differs")
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    elif isinstance(reference, dict):
        if type(actual) is not type(reference) or actual.keys() != reference.keys():
            raise AssertionError("checkpoint mapping structure differs")
        for name in reference:
            assert_identical(actual[name], reference[name])
    elif isinstance(reference, (tuple, list)):
        if type(actual) is not type(reference) or len(actual) != len(reference):
            raise AssertionError("checkpoint sequence structure differs")
        for left, right in zip(actual, reference, strict=True):
            assert_identical(left, right)
    else:
        if type(actual) is not type(reference) or actual != reference:
            raise AssertionError("checkpoint scalar differs")


def run_case(root: Path, config: RunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(config.train.seed)
    sources = {
        name: [
            [256, *(rng.randrange(1, 256) for _ in range(config.model.context + 17))]
            for _ in range(8)
        ]
        for name in sorted(config.data_mix)
    }
    data_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()

    def trainer(name: str) -> Trainer:
        return Trainer(
            config,
            MixedLoader(
                sources,
                config.data_mix,
                sequence_length=config.train.sequence_length,
                batch_size=config.train.batch_size,
                seed=config.train.seed,
            ),
            run_id=name,
            run_root=root,
            data_hash=data_hash,
            tokenizer_hash="authored-token-IDs-not-a-production-tokenizer",
            device="cpu",
        )

    full = trainer("full")
    expected_losses = full.train()
    full.checkpoint(root / "full.pt")
    expected_random = torch.rand(8)
    del full
    gc.collect()

    partial = trainer("partial")
    split = config.train.steps // 2
    first_losses = partial.train(stop_after=split)
    assert_identical(first_losses, expected_losses[:split])
    partial.checkpoint(root / "partial.pt")
    del partial
    gc.collect()

    resumed = trainer("resumed")
    torch.manual_seed(999)  # Deliberately unrelated fresh-trainer RNG before restore.
    resumed.resume(root / "partial.pt")
    remaining_losses = resumed.train()
    assert_identical(first_losses + remaining_losses, expected_losses)
    resumed.checkpoint(root / "resumed.pt")
    torch.testing.assert_close(torch.rand(8), expected_random, atol=0, rtol=0)
    for parameter in resumed.model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise AssertionError("nonfinite final parameter")
    del resumed
    gc.collect()

    reference = torch.load(root / "full.pt", map_location="cpu", weights_only=True)
    actual = torch.load(root / "resumed.pt", map_location="cpu", weights_only=True)
    fields = (
        "version",
        "config",
        "data_hash",
        "tokenizer_hash",
        "validation_hash",
        "model",
        "optimizer",
        "step",
        "rank_states",
    )
    for field in fields:
        assert_identical(actual[field], reference[field])
    record = json.loads((root / "resumed/record.json").read_text())
    assert_identical(record["status"], "completed")
    assert_identical(
        record["resume_checkpoint"],
        {
            "sha256": file_sha256(root / "partial.pt"),
            "source_run_id": "partial",
            "step": split,
        },
    )
    return {
        "configuration": asdict(config),
        "parameters": expected_parameters(config.model),
        "source_hash": data_hash,
        "data_kind": "authored-synthetic-token-IDs",
        "exact_resume": True,
        "compared_checkpoint_fields": list(fields),
        "next_random_probe_exact": True,
        "all_final_parameters_finite": True,
        "losses": expected_losses,
        "split_step": split,
        "checkpoint_hashes": {
            name: file_sha256(root / f"{name}.pt") for name in ("full", "partial", "resumed")
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_checks(
    root: Path,
    *,
    size: str,
    seeds: Sequence[int],
    precisions: Sequence[str],
    steps: int,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds):
        raise ValueError("provide between1 and8 distinct seeds")
    if not precisions or len(set(precisions)) != len(precisions):
        raise ValueError("provide distinct supported precisions")
    cases = [
        configuration(size, precision, seed, steps) for seed in seeds for precision in precisions
    ]
    if root.exists():
        raise FileExistsError("CPU check output must be a new directory")
    root.mkdir(parents=True)
    commit, dirty = git_state()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "device": "cpu",
        "threads": 1,
        "size": size,
        "planned_cases": len(cases),
        "cases": [],
        "limitations": [
            "authored-token-IDs-only-not-real-corpus-or-coding-capability",
            "no-production-tokenizer-compression-or-source-approval",
            "not-GPU-throughput-precision-or-durability-acceptance",
            "same-process-trainer-recreation-not-distributed-fault-recovery",
        ],
    }
    path = root / "report.json"
    atomic_json(path, report)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for config in cases:
            name = f"seed-{config.train.seed}-{config.train.precision}"
            progress(f"Starting {name}: {expected_parameters(config.model):,} parameters")
            result = run_case(root / name, config)
            report["cases"].append(result)
            atomic_json(path, report)
            progress(f"Passed {name}: exact resume ({result['elapsed_seconds']:.1f}s)")
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["ended_at"] = utc_now()
        try:
            atomic_json(path, report)
        finally:
            torch.set_num_threads(previous_threads)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=("tiny", "nano"), default="tiny")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 23])
    parser.add_argument(
        "--precisions", choices=("float32", "bfloat16"), nargs="+", default=["float32", "bfloat16"]
    )
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    report = run_checks(
        args.output, size=args.size, seeds=args.seeds, precisions=args.precisions, steps=args.steps
    )
    print(
        f"CPU resume check {report['status']}: {len(report['cases'])} cases; "
        f"{args.output / 'report.json'}"
    )


if __name__ == "__main__":
    main()
