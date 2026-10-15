"""Local training entry point for explicitly supplied token shards."""

import hashlib
import json
from pathlib import Path

import torch

from scratchcoder.common.config import RunConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import TokenShard
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.trainer import Trainer


def train_from_files(
    *,
    config_path: Path,
    sources_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    run_id: str,
    resume: Path | None = None,
) -> None:
    config = RunConfig.load(config_path)
    tokenizer = ByteBPE.load(tokenizer_path)
    if config.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary must match the frozen tokenizer")
    if resume is None and (output_root / run_id).exists():
        raise ValueError("run already exists; use a new run ID or explicit resume")
    raw = json.loads(sources_path.read_text())
    sources = {
        name: TokenShard(sources_path.parent / path, tokenizer_hash=tokenizer.sha256)
        for name, path in raw.items()
    }
    identity = hashlib.sha256(
        json.dumps(
            {name: source.identity for name, source in sources.items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    torch.set_num_threads(1)
    loader = MixedLoader(
        sources,
        config.data_mix,
        sequence_length=config.train.sequence_length,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    trainer = Trainer(
        config,
        loader,
        run_id=run_id,
        run_root=output_root,
        data_hash=identity,
        tokenizer_hash=tokenizer.sha256,
    )
    if resume is not None:
        trainer.resume(resume)
    trainer.train()
    trainer.checkpoint(output_root / run_id / "checkpoint.pt")
