"""Generation throughput probe on synthetic prompts; it measures speed, not quality."""

import json
import platform
import resource
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from scratchcoder.common.config import ConfigLoader, ModelConfig
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import PRECISIONS, SAMPLER_VERSION, generate_batch
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.model.gpt import GPT
from scratchcoder.train.trainer import synchronized_time, training_device


def model_config(path: Path) -> ModelConfig:
    """Read the ``model`` mapping of a YAML run configuration or a JSON run specification."""
    text = path.read_text(encoding="utf-8")
    try:
        raw: Any = (
            json.loads(text) if path.suffix == ".json" else yaml.load(text, Loader=ConfigLoader)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError("invalid model configuration file") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("configuration must contain a model mapping")
    try:
        return ModelConfig(**raw["model"])
    except TypeError as error:
        raise ValueError(f"invalid model fields: {error}") from error


def peak_memory(device: torch.device) -> tuple[int, str]:
    """CUDA allocator peak since the last reset, or the process's peak resident set on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device), "cuda-max-allocated"
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, "process-max-rss"


def _positive(values: Sequence[int], name: str) -> list[int]:
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"{name} must be positive integers")
    return list(values)


def benchmark_generation(
    model: GPT,
    *,
    batch_sizes: Sequence[int],
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seed: int,
    precision: str,
    repeats: int,
) -> list[dict[str, Any]]:
    """Time ``generate_batch`` per batch size after an untimed two-token warm-up.

    Row ``r`` of a batch has prompt length ``prompt_lengths[r % len(prompt_lengths)]``,
    uniformly random token IDs and seed ``seed + r``. No stop tokens are used, so
    every row spends its whole budget. Each repeat first times a one-token run
    (prefill and one draw), then the full run; ``tokens_per_second`` is generated
    tokens over the median full-run time, prefill included.

    Each batch is one ``generate_batch`` call, left-padded to its longest prompt.
    With mixed lengths it is one unsorted batch, padded more than the
    length-sorted batches of ``generate_many``; ``prompt_slots`` (batch size times
    the longest prompt) and ``prompt_padding_fraction`` record that padding. To
    time a sorted batch, give lengths from one batch of the sorted pool.
    """
    batch_sizes = _positive(batch_sizes, "batch sizes")
    prompt_lengths = _positive(prompt_lengths, "prompt lengths")
    if any(length >= model.config.context for length in prompt_lengths):
        raise ValueError("prompt lengths must leave room for generation within the context")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    results = []
    for batch_size in batch_sizes:
        lengths = [prompt_lengths[row % len(prompt_lengths)] for row in range(batch_size)]
        prompts = [
            torch.randint(0, model.config.vocab_size, (length,), generator=generator).tolist()
            for length in lengths
        ]
        options: dict[str, Any] = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seeds": [seed + row for row in range(batch_size)],
            "precision": precision,
        }
        generate_batch(model, prompts, max_new_tokens=min(2, max_new_tokens), **options)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        seconds, prefill = [], []
        for _ in range(repeats):
            start = synchronized_time(device)
            generate_batch(model, prompts, max_new_tokens=1, **options)
            middle = synchronized_time(device)
            completions = generate_batch(model, prompts, max_new_tokens=max_new_tokens, **options)
            prefill.append(middle - start)
            seconds.append(synchronized_time(device) - middle)
        generated = sum(len(completion.tokens) for completion in completions)
        slots = batch_size * max(lengths)
        memory, memory_kind = peak_memory(device)
        results.append(
            {
                "batch_size": batch_size,
                "prompt_tokens": sum(lengths),
                "prompt_slots": slots,
                "prompt_padding_fraction": 1 - sum(lengths) / slots,
                "generated_tokens": generated,
                "seconds": seconds,
                "tokens_per_second": generated / statistics.median(seconds),
                # Prefill plus one sampled token: the fixed cost inside ``seconds``.
                "prefill_seconds": statistics.median(prefill),
                "peak_memory_bytes": memory,
                "peak_memory_kind": memory_kind,
            }
        )
    return results


def run_generation_benchmark(
    *,
    checkpoint: Path | None,
    tokenizer: Path | None,
    config: Path | None,
    device: str,
    precision: str,
    batch_sizes: Sequence[int],
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    """Build the model from a checkpoint or a configuration with random weights and probe it."""
    if (checkpoint is None) == (config is None):
        raise ValueError("give exactly one of a checkpoint or a model configuration")
    if precision not in PRECISIONS:
        raise ValueError("precision must be float32 or bfloat16")
    resolved = training_device(device)
    source: dict[str, Any]
    if config is not None:
        specification = model_config(config)
        torch.manual_seed(seed)
        model = GPT(specification).to(resolved).eval()
        source = {"kind": "random-initialization", "config_path": str(config), "init_seed": seed}
    elif checkpoint is not None and tokenizer is not None:
        model, _ = load_checkpoint(checkpoint, tokenizer, device=str(resolved))
        source = {"kind": "checkpoint", "checkpoint_sha256": file_sha256(checkpoint)}
    else:
        raise ValueError("a checkpoint needs its tokenizer")
    results = benchmark_generation(
        model,
        batch_sizes=batch_sizes,
        prompt_lengths=prompt_lengths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        precision=precision,
        repeats=repeats,
    )
    return {
        "kind": "generation-benchmark",
        "sampler_version": SAMPLER_VERSION,
        "device": str(resolved),
        "device_name": (
            torch.cuda.get_device_name(resolved)
            if resolved.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "precision": precision,
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "model": {
            **source,
            "config": asdict(model.config),
            "parameters": model.parameter_count(),
        },
        "settings": {
            "batch_sizes": list(batch_sizes),
            "prompt_lengths": list(prompt_lengths),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "repeats": repeats,
            "stops": [],
        },
        "results": results,
    }
