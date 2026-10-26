"""Tiny authored-token precision and exact-resume preflight for CPU or one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig, RunConfig
from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.loader import MixedLoader
from scratchcoder.model.gpt import GPT, expected_parameters
from scratchcoder.optim.adamw import cross_entropy
from scratchcoder.train.trainer import Trainer
from tools.cpu_resume_check import assert_identical, configuration

LOGITS_ATOL = 0.005
LOGITS_RTOL = 0.03
LOSS_ATOL = 0.001
LOSS_RTOL = 0.001
GRADIENT_RELATIVE_NORM_MAX = 0.03
# Compiled against eager training forward (opt-in check; the rehearsal requires it). Float32
# differs only by reordered reductions; bfloat16 autocast also by where casts are fused.
COMPILE_TOLERANCES = {
    "float32": {"loss_atol": 1e-4, "loss_rtol": 1e-4, "gradient_relative_norm_max": 1e-3},
    "bfloat16": {
        "loss_atol": LOSS_ATOL * 10,
        "loss_rtol": LOSS_RTOL * 10,
        "gradient_relative_norm_max": GRADIENT_RELATIVE_NORM_MAX,
    },
}
RESUME_STEPS = 4
A40_MIN_MEMORY_BYTES = 44 * 1024**3
# The A40 Pod (driver 595.91.07, torch 2.9.1) reports 45,490 MiB to PyTorch and 46,068 MiB to
# nvidia-smi: the driver reserves about 578 MiB. 768 MiB still rejects a 1 GiB mismatch.
MEMORY_REPORT_TOLERANCE_BYTES = 768 * 1024**2


def _bare_gpu_uuid(value: str) -> str:
    """Compare UUIDs without nvidia-smi's "GPU-" prefix, which PyTorch omits."""
    text = value.strip().casefold()
    return text[4:] if text.startswith("gpu-") else text


def hardware_description(device: torch.device) -> dict[str, Any]:
    """Describe the selected device and reject CUDA that is not a single A40."""
    if device.type == "cpu":
        return {"device_type": "cpu", "device": "cpu", "gpu_checked": False}
    if device.type != "cuda" or device.index is None:
        raise ValueError("hardware description requires cpu or an explicit CUDA ordinal")
    count = torch.cuda.device_count()
    if count != 1 or device.index != 0:
        raise ValueError(
            f"CUDA preflight requires exactly one visible GPU at cuda:0; found {count}"
        )
    properties = torch.cuda.get_device_properties(device)
    name = str(properties.name)
    capability = list(torch.cuda.get_device_capability(device))
    if capability[0] < 8:
        raise ValueError("CUDA preflight requires Ampere-or-newer BF16-capable hardware")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("selected CUDA device does not support BF16")
    if name.strip().casefold() not in {"nvidia a40", "a40"}:
        raise ValueError(f"CUDA preflight requires an NVIDIA A40; found {name!r}")
    torch_uuid = str(getattr(properties, "uuid", ""))
    if not torch_uuid:
        raise ValueError("PyTorch did not report the selected CUDA device UUID")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"could not verify CUDA device with nvidia-smi: {error}") from error
    rows = [line.split(",", 3) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("nvidia-smi did not report exactly one GPU with the expected fields")
    smi_name, smi_uuid, driver_version, memory_mib_text = (field.strip() for field in rows[0])
    if smi_name.casefold() not in {"nvidia a40", "a40"}:
        raise ValueError(f"nvidia-smi reports a non-A40 device: {smi_name!r}")
    if _bare_gpu_uuid(smi_uuid) != _bare_gpu_uuid(torch_uuid):
        raise ValueError("PyTorch and nvidia-smi report different CUDA device UUIDs")
    try:
        memory_mib = int(memory_mib_text)
    except ValueError as error:
        raise ValueError("nvidia-smi reported an invalid GPU memory size") from error
    total_memory = int(properties.total_memory)
    smi_memory_bytes = memory_mib * 1024**2
    if total_memory <= 0 or smi_memory_bytes <= 0:
        raise ValueError("CUDA device memory reports must be positive")
    if total_memory < A40_MIN_MEMORY_BYTES or smi_memory_bytes < A40_MIN_MEMORY_BYTES:
        raise ValueError("CUDA device memory is below the minimum expected for an NVIDIA A40")
    if abs(total_memory - smi_memory_bytes) > MEMORY_REPORT_TOLERANCE_BYTES:
        raise ValueError("PyTorch and nvidia-smi report materially different CUDA memory sizes")
    return {
        "device_type": "cuda",
        "device": str(device),
        "gpu_checked": True,
        "gpu_count": count,
        "name": name,
        "uuid": torch_uuid,
        "capability": capability,
        "total_memory_bytes": total_memory,
        "nvidia_smi_total_memory_mib": memory_mib,
        "nvidia_smi_total_memory_bytes": smi_memory_bytes,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "driver_version": driver_version,
        "nvidia_smi_name": smi_name,
        "nvidia_smi_memory_mib": memory_mib,
        "bf16_supported": True,
    }


def resolve_device(value: str) -> torch.device:
    """Validate CPU or an explicit CUDA ordinal before creating output files."""
    if value == "cpu":
        return torch.device("cpu")
    if not isinstance(value, str) or re.fullmatch(r"cuda:[0-9]+", value) is None:
        raise ValueError("device must be cpu or cuda:<nonnegative index>")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose cpu explicitly")
    index = int(value.partition(":")[2])
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"CUDA device cuda:{index} is unavailable (device count {count})")
    return torch.device("cuda", index)


def _precision_check(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(1947)
    model_config = ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8)
    reference = GPT(model_config).to(device)
    inputs = torch.arange(16, dtype=torch.long, device=device).reshape(2, 8)
    targets = (inputs + 1) % model_config.vocab_size
    reference_logits = reference(inputs)
    reference_loss = cross_entropy(reference_logits, targets)
    reference_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    reference_gradients = [
        parameter.grad.detach().clone()
        for parameter in reference.parameters()
        if parameter.grad is not None
    ]
    if len(reference_gradients) != len(list(reference.parameters())):
        raise AssertionError("float32 reference has a missing gradient")
    initial_state = {
        name: tensor.detach().clone() for name, tensor in reference.state_dict().items()
    }

    actual = GPT(model_config).to(device)
    actual.load_state_dict(initial_state)
    actual.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        actual_logits = actual(inputs)
        actual_loss = cross_entropy(actual_logits, targets)
    if actual_logits.dtype != torch.bfloat16 or actual_loss.dtype != torch.float32:
        raise AssertionError("autocast did not preserve the expected bfloat16/float32 dtypes")
    torch.testing.assert_close(
        actual_logits.float(), reference_logits, atol=LOGITS_ATOL, rtol=LOGITS_RTOL
    )
    torch.testing.assert_close(actual_loss, reference_loss, atol=LOSS_ATOL, rtol=LOSS_RTOL)
    actual_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.

    relative_errors: list[float] = []
    for parameter, reference_gradient in zip(actual.parameters(), reference_gradients, strict=True):
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError("autocast produced a missing or nonfinite gradient")
        reference_norm = float(reference_gradient.norm())
        error_norm = float((gradient - reference_gradient).norm())
        relative_error = error_norm / reference_norm if reference_norm else error_norm
        if relative_error > GRADIENT_RELATIVE_NORM_MAX:
            raise AssertionError("autocast gradient relative norm error exceeds 0.03")
        relative_errors.append(relative_error)
    if any(parameter.dtype != torch.float32 for parameter in actual.parameters()):
        raise AssertionError("autocast changed master parameter dtype")
    return {
        "status": "passed",
        "same_initial_weights": True,
        "reference_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "logits_atol": LOGITS_ATOL,
        "logits_rtol": LOGITS_RTOL,
        "loss_atol": LOSS_ATOL,
        "loss_rtol": LOSS_RTOL,
        "gradient_relative_norm_max": GRADIENT_RELATIVE_NORM_MAX,
        "max_gradient_relative_norm_error": max(relative_errors),
        "loss_absolute_error": abs(float(actual_loss.detach()) - float(reference_loss.detach())),
        "all_gradients_finite": True,
    }


def _attention_check(device: torch.device) -> dict[str, Any]:
    """Compare the fused attention kernel with the reference path on this device."""
    torch.manual_seed(2718)
    model_config = ModelConfig(vocab_size=272, width=32, layers=2, heads=4, context=16)
    reference = GPT(model_config).to(device)
    fused = GPT(model_config).to(device)
    fused.load_state_dict(reference.state_dict())
    fused.set_fused_attention(True)
    inputs = torch.randint(0, model_config.vocab_size, (4, 16), device=device)
    targets = torch.randint(0, model_config.vocab_size, (4, 16), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        reference_loss = cross_entropy(reference(inputs), targets)
        fused_loss = cross_entropy(fused(inputs), targets)
    reference_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    fused_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
    torch.testing.assert_close(fused_loss, reference_loss, atol=LOSS_ATOL * 10, rtol=LOSS_RTOL * 10)
    relative_errors: list[float] = []
    for left, right in zip(fused.parameters(), reference.parameters(), strict=True):
        if left.grad is None or right.grad is None or not bool(torch.isfinite(left.grad).all()):
            raise AssertionError("fused attention produced a missing or nonfinite gradient")
        reference_norm = float(right.grad.norm())
        error = float((left.grad - right.grad).norm())
        relative = error / reference_norm if reference_norm else error
        if relative > GRADIENT_RELATIVE_NORM_MAX:
            raise AssertionError("fused attention gradient relative norm error exceeds 0.03")
        relative_errors.append(relative)
    with torch.no_grad():
        changed = inputs.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % model_config.vocab_size
        torch.testing.assert_close(fused(changed)[:, :8], fused(inputs)[:, :8], atol=0, rtol=0)
    return {
        "status": "passed",
        "same_initial_weights": True,
        "autocast_dtype": "bfloat16",
        "loss_absolute_error": abs(float(fused_loss.detach()) - float(reference_loss.detach())),
        "max_gradient_relative_norm_error": max(relative_errors),
        "gradient_relative_norm_max": GRADIENT_RELATIVE_NORM_MAX,
        "causality_exact": True,
        "note": "fused CUDA backward kernels may not be bitwise deterministic across runs",
    }


def _compile_check(device: torch.device) -> dict[str, Any]:
    """Compare ``torch.compile`` with eager training forward and backward on a tiny shape.

    Both models start from the same weights with fused attention, as rehearsal training
    runs, and the compiled one is wrapped exactly as the trainer wraps it. Each precision
    compares the loss and every parameter gradient within its tolerance.
    """
    torch.manual_seed(1618)
    model_config = ModelConfig(vocab_size=272, width=32, layers=2, heads=4, context=16)
    inputs = torch.randint(0, model_config.vocab_size, (4, 16), device=device)
    targets = torch.randint(0, model_config.vocab_size, (4, 16), device=device)
    initial = GPT(model_config).state_dict()
    results: dict[str, Any] = {}
    for precision, tolerance in COMPILE_TOLERANCES.items():
        eager, compiled = GPT(model_config).to(device), GPT(model_config).to(device)
        eager.load_state_dict(initial)
        compiled.load_state_dict(initial)
        eager.set_fused_attention(True)
        compiled.set_fused_attention(True)
        forward = torch.compile(compiled, dynamic=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            eager_loss = cross_entropy(eager(inputs), targets)
            compiled_loss = cross_entropy(forward(inputs), targets)
        eager_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
        compiled_loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
        torch.testing.assert_close(
            compiled_loss, eager_loss, atol=tolerance["loss_atol"], rtol=tolerance["loss_rtol"]
        )
        relative_errors: list[float] = []
        for left, right in zip(compiled.parameters(), eager.parameters(), strict=True):
            if left.grad is None or right.grad is None or not bool(torch.isfinite(left.grad).all()):
                raise AssertionError("compiled training produced a missing or nonfinite gradient")
            reference_norm = float(right.grad.norm())
            error = float((left.grad - right.grad).norm())
            relative = error / reference_norm if reference_norm else error
            if relative > tolerance["gradient_relative_norm_max"]:
                raise AssertionError(f"compiled {precision} gradient differs beyond tolerance")
            relative_errors.append(relative)
        results[precision] = {
            **tolerance,
            "loss_absolute_error": abs(float(compiled_loss.detach()) - float(eager_loss.detach())),
            "max_gradient_relative_norm_error": max(relative_errors),
        }
    return {
        "status": "passed",
        "same_initial_weights": True,
        "compile_mode": "torch.compile(model, dynamic=True), default backend",
        "attention": "fused",
        "precisions": results,
        "torch_version": str(torch.__version__),
    }


def _make_trainer(config: RunConfig, root: Path, name: str, device: torch.device) -> Trainer:
    rng = random.Random(config.train.seed)
    sources = {
        domain: [
            [256, *(rng.randrange(1, 256) for _ in range(config.model.context + 17))]
            for _ in range(8)
        ]
        for domain in sorted(config.data_mix)
    }
    data_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    loader = MixedLoader(
        sources,
        config.data_mix,
        sequence_length=config.train.sequence_length,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    return Trainer(
        config,
        loader,
        run_id=name,
        run_root=root,
        data_hash=data_hash,
        tokenizer_hash="authored-synthetic-token-IDs",
        device=str(device),
    )


def _resume_check(root: Path, device: torch.device) -> dict[str, Any]:
    config = configuration("tiny", "bfloat16", seed=23, steps=RESUME_STEPS)
    run_root = root / "runs"
    full = _make_trainer(config, run_root, "full", device)
    full_losses = full.train()
    full_path = root / "full.pt"
    full.checkpoint(full_path)
    expected_next = torch.rand(8, device=device).cpu()
    full_state = torch.load(full_path, map_location="cpu", weights_only=True)
    del full

    partial = _make_trainer(config, run_root, "partial", device)
    split = RESUME_STEPS // 2
    first_losses = partial.train(stop_after=split)
    assert_identical(first_losses, full_losses[:split])
    partial_path = root / "partial.pt"
    partial.checkpoint(partial_path)
    del partial

    resumed = _make_trainer(config, run_root, "resumed", device)
    torch.manual_seed(999)
    resumed.resume(partial_path)
    remaining_losses = resumed.train()
    assert_identical(first_losses + remaining_losses, full_losses)
    resumed_path = root / "resumed.pt"
    resumed.checkpoint(resumed_path)
    for parameter in resumed.model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise AssertionError("resumed model contains a nonfinite parameter")
    resumed_state = torch.load(resumed_path, map_location="cpu", weights_only=True)
    for field in ("config", "model", "optimizer", "step", "rank_states"):
        assert_identical(resumed_state[field], full_state[field])
    actual_next = torch.rand(8, device=device).cpu()
    torch.testing.assert_close(actual_next, expected_next, atol=0, rtol=0)
    return {
        "status": "passed",
        "steps": RESUME_STEPS,
        "split_step": split,
        "accumulation_steps": config.train.accumulation_steps,
        "precision": config.train.precision,
        "losses_match_exactly": True,
        "model_optimizer_loader_and_rng_checkpoint_state_match_exactly": True,
        "next_device_rng_draw_matches_exactly": True,
        "compared_checkpoint_fields": ["config", "model", "optimizer", "step", "rank_states"],
    }


def run_preflight(
    output: Path, *, device: str = "cpu", compile_check: bool = False
) -> dict[str, Any]:
    """Run authored tiny precision and resume checks; never downloads or provisions.

    ``compile_check`` adds the compiled-against-eager comparison that the rehearsal
    worker requires; the pilot's preflight leaves it off.
    """
    selected = resolve_device(device)
    if output.exists() or output.is_symlink():
        raise FileExistsError("preflight output must be a new directory")
    hardware = hardware_description(selected)
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "device": str(selected),
        "hardware": hardware,
        "hardware_precision_and_resume_checked": False,
        "data_kind": "authored-synthetic-token-IDs",
        "model_parameters": expected_parameters(ModelConfig(272, 16, 1, 2, 8)),
        "gpu_readiness_established": False,
        "limitations": [
            "tiny-authored-token-fixture-only",
            "does-not-establish-GPU-correctness-throughput-or-launch-readiness",
            "does-not-test-real-corpus-or-coding-capability",
        ],
    }
    report_path = output / "report.json"
    atomic_json(report_path, report)
    previous_threads = torch.get_num_threads()
    started = time.perf_counter()
    torch.set_num_threads(1)
    try:
        report["precision"] = _precision_check(selected)
        report["fused_attention"] = _attention_check(selected)
        if compile_check:
            report["compile_equivalence"] = _compile_check(selected)
        report["exact_resume"] = _resume_check(output / "resume", selected)
        report["hardware_precision_and_resume_checked"] = selected.type == "cuda"
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["ended_at"] = utc_now()
        try:
            atomic_json(report_path, report)
        finally:
            torch.set_num_threads(previous_threads)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cpu", help="cpu or explicit CUDA ordinal such as cuda:0"
    )
    parser.add_argument(
        "--compile-check",
        action="store_true",
        help="also compare torch.compile with eager training (required by the rehearsal runs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_preflight(args.output, device=args.device, compile_check=args.compile_check)
    except (FileExistsError, OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"pilot-preflight: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"Pilot preflight {report['status']} on {report['device']}: {args.output / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
