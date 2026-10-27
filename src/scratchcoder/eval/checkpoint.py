"""Restricted tensor-only loading for the project's own local checkpoints."""

from pathlib import Path

import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.trainer import training_device


def load_checkpoint(
    checkpoint: Path, tokenizer_path: Path, *, device: str = "cpu"
) -> tuple[GPT, ByteBPE]:
    """Load weights on CPU, check identities, then move the model to ``device``.

    ``device`` is ``cpu`` (default), ``cuda`` or ``cuda:N``; an unavailable device is
    rejected rather than replaced.
    """
    resolved = training_device(device)
    tokenizer = ByteBPE.load(tokenizer_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["tokenizer_hash"] != tokenizer.sha256:
        raise ValueError("checkpoint and tokenizer identities differ")
    config = ModelConfig(**saved["config"]["model"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary differs from tokenizer")
    model = GPT(config)
    model.load_state_dict(saved["model"])
    model.to(resolved)
    model.eval()
    torch.set_num_threads(1)
    return model, tokenizer
