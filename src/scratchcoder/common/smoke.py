"""Offline synthetic data -> two-stage training -> sampling -> validation check."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import torch

from scratchcoder.common.config import RunConfig
from scratchcoder.common.records import atomic_json
from scratchcoder.data.fixtures import prepare_fixture
from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import TokenShard
from scratchcoder.engine.generation import generate
from scratchcoder.train.trainer import Trainer


def run_smoke(
    output: TextIO,
    *,
    config_path: Path = Path("configs/smoke.yaml"),
    output_root: Path | None = None,
) -> int:
    config = RunConfig.load(config_path)
    directory = output_root or Path("runs") / f"smoke-{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tokenizer, paths, metadata = prepare_fixture(
        directory / "data", vocab_size=config.model.vocab_size
    )
    data_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    config = replace(config, model=replace(config.model, vocab_size=tokenizer.vocab_size))
    stage_metrics: dict[str, dict[str, float]] = {}
    stage_checkpoint: Path | None = None
    for stage in (1, 2):
        shares = {"web": 1.0} if stage == 1 else {"web": 0.1, "python": 0.9}
        sources = {"web": TokenShard(paths[f"web-stage{stage}"], tokenizer_hash=tokenizer.sha256)}
        if stage == 2:
            sources["python"] = TokenShard(paths["python-train"], tokenizer_hash=tokenizer.sha256)
        options = replace(
            config,
            data_mix=shares,
            parent_checkpoint=None if stage_checkpoint is None else str(stage_checkpoint),
        )
        loader = MixedLoader(
            sources,
            shares,
            sequence_length=options.train.sequence_length,
            batch_size=options.train.batch_size,
            seed=options.train.seed,
        )
        trainer = Trainer(
            options,
            loader,
            run_id=f"stage{stage}",
            run_root=directory,
            data_hash=data_hash,
            tokenizer_hash=tokenizer.sha256,
        )
        losses = trainer.train()
        stage_checkpoint = directory / f"stage{stage}.pt"
        trainer.checkpoint(stage_checkpoint)
        metrics = {"final_training_loss": losses[-1]}
        for domain in ("web", "python"):
            shard = TokenShard(paths[f"{domain}-validation"], tokenizer_hash=tokenizer.sha256)
            ids = shard[0][: options.train.sequence_length + 1]
            loss = trainer.validation_loss(torch.tensor([ids[:-1]]), torch.tensor([ids[1:]]))
            byte_tokens = tokenizer.byte_tokens()
            byte_count = sum(len(byte_tokens.get(token, b"")) for token in ids[1:])
            metrics[f"{domain}_validation_loss"] = loss
            metrics[f"{domain}_bits_per_byte"] = loss * (len(ids) - 1) / math.log(2) / byte_count
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("smoke validation is not finite")
        stage_metrics[f"stage{stage}"] = metrics
        output.write(f"stage{stage}: trained, checkpointed, and validated\n")
    prompt = tokenizer.encode("def")
    generated = generate(trainer.model, torch.tensor([prompt]), max_new_tokens=8)
    sample_text = tokenizer.decode(generated[0].tolist())
    atomic_json(
        directory / "summary.json",
        {
            "synthetic": True,
            "benchmark": False,
            "stages": stage_metrics,
            "sample": sample_text,
            "tokenizer_hash": tokenizer.sha256,
        },
    )
    output.write(f"Synthetic smoke passed; artifacts: {directory}\n")
    return 0
