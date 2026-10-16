"""Prepare a synthetic-only project nano checkpoint for integration, not capability."""

import argparse
from dataclasses import replace
from pathlib import Path

from scratchcoder.common.config import RunConfig
from scratchcoder.common.records import atomic_json
from scratchcoder.data.fixtures import prepare_fixture
from scratchcoder.train.command import train_from_files


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    config = RunConfig.load(Path("configs/nano.yaml"))
    tokenizer, _, _ = prepare_fixture(output / "data", vocab_size=config.model.vocab_size)
    config = replace(config, model=replace(config.model, vocab_size=tokenizer.vocab_size))
    config.save(output / "config.yaml")
    train_from_files(
        config_path=output / "config.yaml",
        sources_path=output / "data/stage1-sources.json",
        tokenizer_path=output / "data/tokenizer.json",
        output_root=output,
        run_id="train",
    )
    atomic_json(
        output / "purpose.json",
        {
            "synthetic_training_only": True,
            "benchmark_training": False,
            "purpose": "pipeline integration, not model capability",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
