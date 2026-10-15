"""Explicitly fail until the complete tiny training pipeline is implemented."""

from typing import TextIO

PENDING_STAGES = (
    "data preparation",
    "tokenizer training",
    "pretraining stage 1",
    "pretraining stage 2",
    "sampling",
    "evaluation",
)


def run_smoke(output: TextIO) -> int:
    """Report missing stages without spending money or creating artifacts."""
    output.write("Smoke pipeline is incomplete. Missing stages:\n")
    for stage in PENDING_STAGES:
        output.write(f"  - {stage}\n")
    output.write("No training or evaluation was run.\n")
    return 2
