"""The checked-in A40 specification must encode the agreed pilot exactly."""

import json
from pathlib import Path

from scratchcoder.model.gpt import expected_parameters
from scratchcoder.train.pilot import accumulation_steps
from tools.gpu_pilot import Specification

SPEC = Path(__file__).resolve().parents[1] / "configs" / "pilot-a40-s.json"


def test_specification_matches_the_agreed_experiment() -> None:
    spec = Specification.load(SPEC)
    model = spec.config.model
    assert (model.vocab_size, model.width, model.layers, model.heads, model.context) == (
        65536,
        768,
        12,
        12,
        1024,
    )
    assert expected_parameters(model) == 135_285_504
    assert spec.config.train.precision == "bfloat16"
    assert spec.config.train.attention == "fused"  # Equivalence-tested kernel; no score matrices.
    assert spec.config.train.optimizer_foreach is True  # Bitwise-identical update, fewer kernels.
    assert spec.config.train.z_loss == 0 and spec.config.model.qk_norm is False  # Agreed recipe.
    assert spec.config.train.compile is False and spec.config.train.schedule == "cosine"
    # Owner decision 2026-10-24: probes and arms train eagerly; one separate 20-update
    # compiled probe at the chosen microbatch measures the speedup and compile cost.
    assert spec.compile_probe_steps == 20 and spec.compile_probe_warmup_updates == 2
    assert spec.compile_probe_seconds == 240
    assert spec.config.train.sequence_length == 1024
    assert spec.config.train.validation_interval == 0
    assert spec.config.train.validation_documents == 32
    assert spec.tokens_per_update == 131_072
    assert spec.microbatches[:3] == (16, 32, 8)
    assert [accumulation_steps(b, 1024, 131_072) for b in spec.microbatches] == [8, 4, 16, 32]
    assert spec.probe_steps == 20
    assert spec.learning_rates == (0.0003, 0.0006, 0.0012)
    assert spec.screen_steps == 100 and spec.config.train.steps == 500
    arms = len(spec.learning_rates)
    total_updates = arms * spec.screen_steps + (spec.config.train.steps - spec.screen_steps)
    assert total_updates == 700
    assert total_updates * spec.tokens_per_update == 91_750_400
    limits = spec.limits
    assert (limits.probe_seconds, limits.training_seconds, limits.pod_seconds) == (
        300,
        3600,
        5400,
    )
    assert limits.approved_cost_cap == 5.0
    assert limits.quoted_total_hourly >= spec.quoted_gpu_hourly == 0.59
    assert limits.estimated_total_cost == 0.9
    assert limits.cost_stop <= 5.0
    assert limits.idle_seconds <= 600
    assert limits.finalization_reserve_seconds == 900
    assert spec.expected_tokens_per_second is None  # Measured probe baseline, not a guess.
    # The 65.5M winner tokens (500 x 131,072) stay below one pass over the reviewed shard.
    assert spec.config.train.steps * spec.tokens_per_update == 65_536_000 < 66_030_000


def test_improved_variant_changes_only_the_declared_recipe_options() -> None:
    base = Specification.load(SPEC)
    improved = Specification.load(SPEC.with_name("pilot-a40-s-improved.json"))
    assert improved.config.model.qk_norm is True
    assert expected_parameters(improved.config.model) == 135_285_504 + 12 * 2 * 64
    assert improved.config.train.z_loss == 0.0001
    assert improved.config.train.compile is True and improved.config.train.attention == "fused"
    assert improved.compile_probe_steps == 0  # Compiles every phase; no eager comparison.
    assert improved.config.train.schedule == "cosine"
    for name in ("steps", "sequence_length", "learning_rate", "warmup_steps", "precision"):
        assert getattr(improved.config.train, name) == getattr(base.config.train, name)
    assert improved.limits == base.limits
    assert improved.microbatches == base.microbatches
    assert improved.learning_rates == base.learning_rates
    assert improved.tokens_per_update == base.tokens_per_update


def test_specification_file_is_plain_json_without_private_material() -> None:
    raw = json.loads(SPEC.read_text())
    assert set(raw) >= {"config", "limits", "microbatches", "learning_rates"}
    assert "sha256" not in SPEC.read_text() and "/Users/" not in SPEC.read_text()
