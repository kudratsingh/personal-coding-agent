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


def test_specification_file_is_plain_json_without_private_material() -> None:
    raw = json.loads(SPEC.read_text())
    assert set(raw) >= {"config", "limits", "microbatches", "learning_rates"}
    assert "sha256" not in SPEC.read_text() and "/Users/" not in SPEC.read_text()
