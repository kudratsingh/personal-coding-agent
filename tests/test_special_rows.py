"""The special-token rows: their fresh draw and their own learning rate in AdamW.

Reference tests: the draw is recomputed by hand in float64 from the formula; the scaled
update is recomputed by hand from the AdamW formula; a scale of 1 reproduces the plain
update bit for bit, in the per-tensor loop and in the list kernels.
"""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.optim.adamw import AdamW
from scratchcoder.tokenizer.bpe import FIRST_MERGE
from scratchcoder.train.special_rows import (
    BOS_ID,
    SPECIAL_ROW_LR_SCALE,
    SPECIAL_ROWS,
    ordinary_mask,
    reinitialize_special_rows,
    row_scale,
)
from scratchcoder.train.trainer import Trainer


def embedding(vocabulary: int = FIRST_MERGE + 20, width: int = 6, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocabulary, width, generator=generator) * 0.3 + 0.5


def test_the_rows_are_the_chat_specials_and_reserved_tokens_never_bos() -> None:
    assert SPECIAL_ROWS == tuple(range(257, 272))
    assert BOS_ID not in SPECIAL_ROWS
    assert SPECIAL_ROW_LR_SCALE == 10.0
    mask = ordinary_mask(FIRST_MERGE + 20)
    assert not mask[256:272].any() and mask[:256].all() and mask[272:].all()


def test_the_draw_matches_the_formula_by_hand() -> None:
    weight = embedding()
    original = weight.clone()
    record = reinitialize_special_rows(weight, seed=7)
    ordinary = [index for index in range(weight.shape[0]) if not 256 <= index < 272]
    rows = original[ordinary].double()
    mean = rows.sum(dim=0) / len(ordinary)
    spread = ((rows - mean) ** 2).sum(dim=0).div(len(ordinary)).sqrt()
    noise = torch.randn(15, 6, generator=torch.Generator().manual_seed(7)).double()
    expected = mean + spread * noise
    torch.testing.assert_close(weight[257:272].double(), expected, rtol=1e-6, atol=1e-6)
    # Every other row, BOS included, is untouched.
    assert torch.equal(weight[:257], original[:257])
    assert torch.equal(weight[272:], original[272:])
    assert record["rows"] == list(range(257, 272)) and record["seed"] == 7
    assert record["ordinary_rows"] == len(ordinary)
    assert record["before_sha256"] != record["after_sha256"]


def test_the_draw_is_deterministic_and_seed_dependent() -> None:
    first, second, other = embedding(), embedding(), embedding()
    one = reinitialize_special_rows(first, seed=3)
    two = reinitialize_special_rows(second, seed=3)
    three = reinitialize_special_rows(other, seed=4)
    assert torch.equal(first, second) and one == two
    assert not torch.equal(first[257:272], other[257:272])
    assert three["after_sha256"] != one["after_sha256"]


def test_fresh_rows_are_as_alike_as_the_ordinary_statistics_predict() -> None:
    # Ordinary rows with mean m and per-dimension spread s give fresh rows whose expected
    # pairwise cosine is about |m|^2 / (|m|^2 + sum s^2): 0.5 here, far below the gate's 0.9.
    weight = torch.randn(4_000, 256, generator=torch.Generator().manual_seed(1))
    weight += 1.0
    collapsed = weight.clone()
    collapsed[257:272] = weight[272:].mean(dim=0)  # Run 1's failure: rows on one direction.
    record = reinitialize_special_rows(collapsed, seed=0)
    assert record["max_cosine_after"] < 0.75
    assert 0.3 < record["mean_cosine_after"] < 0.7


def test_the_draw_refuses_malformed_input() -> None:
    with pytest.raises(ValueError):
        reinitialize_special_rows(torch.zeros(300), seed=0)
    with pytest.raises(ValueError):
        reinitialize_special_rows(embedding(), seed=-1)
    with pytest.raises(ValueError):
        reinitialize_special_rows(embedding(), seed=0, rows=(260, 259))
    with pytest.raises(ValueError):
        reinitialize_special_rows(embedding(vocabulary=FIRST_MERGE), seed=0, rows=(999,))


def test_row_scale_is_ten_on_the_special_rows_and_one_elsewhere() -> None:
    scale = row_scale(FIRST_MERGE + 4)
    assert scale.shape == (FIRST_MERGE + 4, 1)
    assert torch.equal(scale[257:272], torch.full((15, 1), 10.0))
    assert torch.equal(scale[:257], torch.ones(257, 1))
    assert torch.equal(scale[272:], torch.ones(4, 1))
    for bad in (0.0, -1.0, math.inf, True):
        with pytest.raises(ValueError):
            row_scale(300, scale=bad)


def run_steps(
    *, foreach: bool, scale: torch.Tensor | None, steps: int = 6, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    rows = torch.randn(8, 3, dtype=dtype, requires_grad=True)
    other = torch.randn(4, 2, dtype=dtype, requires_grad=True)
    optimizer = AdamW(
        [rows, other],
        learning_rate=0.01,
        weight_decay=0.1,
        decay_mask=[False, True],
        foreach=foreach,
        lr_scales=[scale, None],
    )
    for _ in range(steps):
        rows.grad = torch.randn_like(rows)
        other.grad = torch.randn_like(other)
        optimizer.step()
    return rows.detach(), other.detach()


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_scale_of_one_reproduces_the_plain_update_exactly(
    foreach: bool, dtype: torch.dtype
) -> None:
    plain = run_steps(foreach=foreach, scale=None, dtype=dtype)
    ones = run_steps(foreach=foreach, scale=torch.ones(8, 1), dtype=dtype)
    assert torch.equal(plain[0], ones[0]) and torch.equal(plain[1], ones[1])


def test_the_list_kernels_match_the_loop_with_a_scale() -> None:
    scale = torch.ones(8, 1)
    scale[2:5] = 10.0
    loop = run_steps(foreach=False, scale=scale)
    lists = run_steps(foreach=True, scale=scale)
    assert torch.equal(loop[0], lists[0]) and torch.equal(loop[1], lists[1])


def test_scaled_rows_follow_the_adamw_formula_by_hand() -> None:
    """Three float64 updates recomputed from the formula: row r steps by
    -lr x scale_r x m_hat / (sqrt(v_hat) + eps)."""
    torch.manual_seed(2)
    weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([[1.0], [10.0], [1.0], [10.0], [0.0]], dtype=torch.float64)
    optimizer = AdamW(
        [weight], learning_rate=0.02, weight_decay=0.0, decay_mask=[False], lr_scales=[scale]
    )
    start = weight.detach().clone()
    expected = start.clone()
    m = torch.zeros_like(expected)
    v = torch.zeros_like(expected)
    beta1, beta2, eps = 0.9, 0.95, 1e-8
    for step in range(1, 4):
        gradient = torch.randn_like(expected)
        weight.grad = gradient.clone()
        optimizer.step()
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        update = (m / (1 - beta1**step)) / ((v / (1 - beta2**step)).sqrt() + eps)
        expected = expected - 0.02 * scale * update
        torch.testing.assert_close(weight.detach(), expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(weight.detach()[4], start[4])  # Scale 0: the row never moves.


def test_ten_times_the_scale_is_ten_times_the_step() -> None:
    scale = torch.ones(8, 1)
    scale[3] = 10.0
    start = run_steps(foreach=False, scale=None, steps=0)[0]
    plain = run_steps(foreach=False, scale=None, steps=1)[0]
    scaled = run_steps(foreach=False, scale=scale, steps=1)[0]
    torch.testing.assert_close(scaled[3] - start[3], 10 * (plain[3] - start[3]))
    assert torch.equal(scaled[:3], plain[:3]) and torch.equal(scaled[4:], plain[4:])


def test_scales_are_refused_on_decayed_or_mismatched_parameters() -> None:
    weight = torch.zeros(4, 3, requires_grad=True)
    with pytest.raises(ValueError, match="without decay"):
        AdamW([weight], decay_mask=[True], lr_scales=[torch.ones(4, 1)])
    for bad in (torch.ones(3, 1), torch.full((4, 1), math.nan), -torch.ones(4, 1)):
        with pytest.raises(ValueError, match="scale"):
            AdamW([weight], decay_mask=[False], lr_scales=[bad])
    with pytest.raises(ValueError, match="one entry"):
        AdamW([weight], decay_mask=[False], lr_scales=[])
    caller = torch.ones(4, 1)
    optimizer = AdamW([weight], decay_mask=[False], lr_scales=[caller])
    caller.fill_(99.0)  # The optimizer owns its copy.
    assert optimizer.lr_scales[0] is not None and torch.equal(
        optimizer.lr_scales[0], torch.ones(4, 1)
    )


def tiny_trainer(tmp_path: Path, **train: object) -> Trainer:
    from scratchcoder.data.loader import BatchSource

    class Fixed(BatchSource):
        def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
            return [[1, 2, 3, 4]], [[2, 3, 4, 270]], ["x"]

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            return None

    model = ModelConfig(vocab_size=FIRST_MERGE + 8, width=8, layers=1, heads=2, context=8)
    options = TrainConfig(steps=2, batch_size=1, sequence_length=4, warmup_steps=1)
    config = RunConfig(model=model, train=replace(options, **train))  # type: ignore[arg-type]
    return Trainer(
        config,
        Fixed(),
        run_id="special-rows",
        run_root=tmp_path / "runs",
        data_hash="a" * 64,
        tokenizer_hash="b" * 64,
    )


def test_the_trainer_defaults_leave_the_embedding_and_the_update_alone(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path)
    assert trainer.special_rows is None
    assert all(scale is None for scale in trainer.optimizer.lr_scales)


def test_the_trainer_redraws_and_scales_the_special_rows(tmp_path: Path) -> None:
    plain = tiny_trainer(tmp_path / "plain")
    trainer = tiny_trainer(
        tmp_path / "special",
        special_rows_init="ordinary-mean-noise",
        special_rows_seed=11,
        special_rows_lr_scale=10.0,
    )
    assert trainer.special_rows is not None and trainer.special_rows["seed"] == 11
    before = plain.model.embedding.weight.detach()
    after = trainer.model.embedding.weight.detach()
    assert torch.equal(before[:257], after[:257]) and torch.equal(before[272:], after[272:])
    assert not torch.equal(before[257:272], after[257:272])
    names = [name for name, _ in trainer.model.named_parameters()]
    scales = dict(zip(names, trainer.optimizer.lr_scales, strict=True))
    embedding_scale = scales["embedding.weight"]
    assert embedding_scale is not None and float(embedding_scale[270]) == 10.0
    assert all(scale is None for name, scale in scales.items() if name != "embedding.weight")
    trainer.train()
    assert trainer.step == 2


def test_older_configs_resume_with_the_defaults() -> None:
    from scratchcoder.train.trainer import ADDED_TRAIN_DEFAULTS

    assert ADDED_TRAIN_DEFAULTS["special_rows_init"] == "none"
    assert ADDED_TRAIN_DEFAULTS["special_rows_lr_scale"] == 1.0
    bad: dict[str, str] | dict[str, int]
    for bad in (
        {"special_rows_init": "zeros"},
        {"special_rows_seed": -1},
        {"special_rows_lr_scale": 0},
    ):
        with pytest.raises(ValueError):
            TrainConfig(**bad)
