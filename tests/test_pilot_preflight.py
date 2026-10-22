import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.pilot_preflight import (
    GRADIENT_RELATIVE_NORM_MAX,
    LOGITS_ATOL,
    LOGITS_RTOL,
    LOSS_ATOL,
    LOSS_RTOL,
    hardware_description,
    resolve_device,
    run_preflight,
)


def test_cpu_preflight_runs_authored_precision_and_exact_resume(tmp_path: Path) -> None:
    output = tmp_path / "cpu-preflight"
    report = run_preflight(output)

    assert report["status"] == "completed"
    assert report["device"] == "cpu"
    assert report["gpu_readiness_established"] is False
    assert report["hardware"] == {"device_type": "cpu", "device": "cpu", "gpu_checked": False}
    assert report["hardware_precision_and_resume_checked"] is False
    assert report["data_kind"] == "authored-synthetic-token-IDs"
    assert report["precision"]["same_initial_weights"] is True
    assert report["precision"]["logits_atol"] == LOGITS_ATOL == 0.005
    assert report["precision"]["logits_rtol"] == LOGITS_RTOL == 0.03
    assert report["precision"]["loss_atol"] == LOSS_ATOL == 0.001
    assert report["precision"]["loss_rtol"] == LOSS_RTOL == 0.001
    assert report["precision"]["gradient_relative_norm_max"] == GRADIENT_RELATIVE_NORM_MAX
    assert report["fused_attention"]["status"] == "passed"
    assert report["fused_attention"]["causality_exact"] is True
    assert report["fused_attention"]["max_gradient_relative_norm_error"] <= 0.03
    assert report["exact_resume"]["steps"] == 4
    assert report["exact_resume"]["accumulation_steps"] == 2
    assert report["exact_resume"]["losses_match_exactly"] is True
    assert report["exact_resume"]["model_optimizer_loader_and_rng_checkpoint_state_match_exactly"]
    assert report["exact_resume"]["next_device_rng_draw_matches_exactly"] is True
    assert json.loads((output / "report.json").read_text())["status"] == "completed"


def test_cuda_unavailable_fails_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="unavailable"):
        run_preflight(output, device="cuda:0")

    assert not output.exists()


def test_invalid_cuda_ordinal_fails_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="unavailable"):
        run_preflight(output, device="cuda:1")

    assert not output.exists()


@pytest.mark.parametrize("value", ["cuda", "cuda:-1", "cuda:0.5", "mps", "CPU", "--device"])
def test_only_cpu_or_explicit_cuda_ordinal_is_accepted(value: str) -> None:
    with pytest.raises(ValueError, match="device must be"):
        resolve_device(value)


def test_cuda_ordinal_resolution_is_testable_without_cuda_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_device("cuda:1") == torch.device("cuda:1")


def _mock_a40(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int = 1,
    uuid: str = "GPU-test",
    total_memory: int = 48 * 1024**3,
) -> None:
    properties = SimpleNamespace(
        name="NVIDIA A40",
        uuid=uuid,
        total_memory=total_memory,
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 6))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)


def test_cuda_hardware_description_checks_torch_and_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_a40(monkeypatch)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA A40, GPU-test, 550.54, 49152\n"),
    )

    report = hardware_description(torch.device("cuda:0"))

    assert report["device"] == "cuda:0"
    assert report["name"] == report["nvidia_smi_name"] == "NVIDIA A40"
    assert report["uuid"] == "GPU-test"
    assert report["capability"] == [8, 6]
    assert report["total_memory_bytes"] == 48 * 1024**3
    assert report["nvidia_smi_total_memory_mib"] == 49152
    assert report["nvidia_smi_total_memory_bytes"] == 48 * 1024**3
    assert report["driver_version"] == "550.54"
    assert report["bf16_supported"] is True


@pytest.mark.parametrize(
    ("count", "uuid", "stdout", "message"),
    [
        (2, "GPU-test", "NVIDIA A40, GPU-test, 550.54, 49152\n", "exactly one"),
        (1, "GPU-test", "NVIDIA A40, GPU-other, 550.54, 49152\n", "different.*UUIDs"),
        (1, "GPU-test", "NVIDIA RTX, GPU-test, 550.54, 49152\n", "non-A40"),
    ],
)
def test_cuda_hardware_mismatch_rejected(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    uuid: str,
    stdout: str,
    message: str,
) -> None:
    _mock_a40(monkeypatch, count=count, uuid=uuid)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=stdout),
    )

    with pytest.raises(ValueError, match=message):
        hardware_description(torch.device("cuda:0"))


@pytest.mark.parametrize("capability,bf16", [((7, 5), True), ((8, 0), False)])
def test_cuda_requires_ampere_and_bf16(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    bf16: bool,
) -> None:
    _mock_a40(monkeypatch)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)

    with pytest.raises(ValueError, match="Ampere-or-newer|does not support BF16"):
        hardware_description(torch.device("cuda:0"))


def test_cuda_hardware_rejection_happens_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _mock_a40(monkeypatch, count=2)
    output = tmp_path / "not-created"

    with pytest.raises(ValueError, match="exactly one"):
        run_preflight(output, device="cuda:0")

    assert not output.exists()


def test_nvidia_smi_query_uses_argument_array_and_five_second_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_a40(monkeypatch)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="NVIDIA A40, GPU-test, 550.54, 49152\n")

    monkeypatch.setattr("tools.pilot_preflight.subprocess.run", fake_run)
    hardware_description(torch.device("cuda:0"))

    assert len(calls) == 1
    assert isinstance(calls[0][0][0], list)
    assert calls[0][1]["timeout"] == 5
    assert calls[0][1]["check"] is True


def test_cuda_memory_comparison_allows_small_driver_reservation_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_memory = 48 * 1024**3 - 128 * 1024**2
    _mock_a40(monkeypatch, total_memory=torch_memory)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA A40, GPU-test, 550.54, 49152\n"),
    )

    report = hardware_description(torch.device("cuda:0"))

    assert report["total_memory_bytes"] == torch_memory
    assert report["nvidia_smi_total_memory_bytes"] == 48 * 1024**3


def test_cuda_memory_comparison_rejects_large_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_a40(monkeypatch, total_memory=47 * 1024**3)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA A40, GPU-test, 550.54, 49152\n"),
    )

    with pytest.raises(ValueError, match="materially different"):
        hardware_description(torch.device("cuda:0"))


def test_cuda_memory_comparison_rejects_capacity_below_a40_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_a40(monkeypatch, total_memory=43 * 1024**3)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA A40, GPU-test, 550.54, 44032\n"),
    )

    with pytest.raises(ValueError, match="below the minimum"):
        hardware_description(torch.device("cuda:0"))


@pytest.mark.parametrize(
    ("torch_memory", "smi_memory", "message"),
    [
        (0, "49152", "must be positive"),
        (48 * 1024**3, "0", "must be positive"),
        (48 * 1024**3, "-1", "must be positive"),
    ],
)
def test_cuda_memory_comparison_rejects_nonpositive_values(
    monkeypatch: pytest.MonkeyPatch,
    torch_memory: int,
    smi_memory: str,
    message: str,
) -> None:
    _mock_a40(monkeypatch, total_memory=torch_memory)
    monkeypatch.setattr(
        "tools.pilot_preflight.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"NVIDIA A40, GPU-test, 550.54, {smi_memory}\n"
        ),
    )

    with pytest.raises(ValueError, match=message):
        hardware_description(torch.device("cuda:0"))


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep")

    with pytest.raises(FileExistsError):
        run_preflight(output, device="cpu")

    assert marker.read_text() == "keep"
