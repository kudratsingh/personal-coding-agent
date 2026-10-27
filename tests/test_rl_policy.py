"""Reference checks for group-mean advantages and the token-normalized policy-gradient loss."""

import math

import pytest
import torch
from torch import Tensor

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.policy import (
    PolicySequence,
    group_advantages,
    policy_gradient_backward,
    policy_sequences,
    token_log_probs,
)

CPU = torch.device("cpu")


class TablePolicy(torch.nn.Module):
    """A toy policy whose next-token logits ignore the input: logits = theta everywhere.

    Then log pi(y) = theta_y - logsumexp(theta), and its gradient is e_y - softmax(theta),
    so the policy-gradient loss's gradient can be written by hand.
    """

    def __init__(self, theta: list[float]) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=torch.float64))

    def forward(self, ids: Tensor) -> Tensor:
        return self.theta.expand(*ids.shape, self.theta.shape[0])


def worked_example() -> tuple[list[PolicySequence], int]:
    """Three groups over a 3-token vocabulary; the prompt is token 0 for every attempt.

    - Group A: rewards (1, 0), mean 0.5, advantages (+0.5, -0.5); attempts [0, 2] and [1].
    - Group B: rewards (0.5, 0.5): all equal, skipped; attempts [2, 2, 2] and [1].
    - Group C: rewards (1, 0, 0.5), mean 0.5, advantages (+0.5, -0.5, 0); attempts [2], [0]
      and [1, 1, 1]. The last has zero advantage: it counts in N but adds no gradient.

    N = 2 + 1 (A) + 1 + 1 + 3 (C) = 8; group B's 4 tokens are excluded.
    """
    groups = [
        (group_advantages("A", [1.0, 0.0]), [[0, 2], [1]]),
        (group_advantages("B", [0.5, 0.5]), [[2, 2, 2], [1]]),
        (group_advantages("C", [1.0, 0.0, 0.5]), [[2], [0], [1, 1, 1]]),
    ]
    prompts, attempts, advantages, skipped = [], [], [], []
    for group, tokens in groups:
        for advantage, attempt in zip(group.advantages, tokens, strict=True):
            prompts.append([0])
            attempts.append(attempt)
            advantages.append(advantage)
            skipped.append(group.skipped)
    return policy_sequences(prompts, attempts, advantages, skipped)


def test_group_advantages_subtract_the_mean_without_std_normalization() -> None:
    group = group_advantages("t", [1.0, 0.0, 0.5, 0.5])
    assert group.mean == 0.5 and not group.skipped
    assert group.advantages == (0.5, -0.5, 0.0, 0.0)
    wide = group_advantages("t", [0.0, 0.0, 0.0, 1.0])
    assert wide.advantages == (-0.25, -0.25, -0.25, 0.75)  # not divided by the std


@pytest.mark.parametrize("rewards", [[0.0] * 8, [1.0] * 8, [0.3] * 8, [0.25]])
def test_an_all_equal_group_is_skipped_with_exact_zero_advantages(rewards: list[float]) -> None:
    group = group_advantages("t", rewards)
    assert group.skipped and group.advantages == (0.0,) * len(rewards)


def test_group_advantages_reject_empty_or_nonfinite_rewards() -> None:
    with pytest.raises(ValueError):
        group_advantages("t", [])
    with pytest.raises(ValueError):
        group_advantages("t", [0.0, math.nan])


def test_the_denominator_counts_attempt_tokens_of_non_skipped_groups_only() -> None:
    sequences, denominator = worked_example()
    assert denominator == 8
    # The zero-advantage attempt is counted in N but not forwarded.
    assert [(s.tokens, s.advantage) for s in sequences] == [
        ((0, 0, 2), 0.5),
        ((0, 1), -0.5),
        ((0, 2), 0.5),
        ((0, 0), -0.5),
    ]
    assert sum(s.attempt_tokens for s in sequences) == denominator - 3  # C2 has 3 tokens


def test_policy_sequences_reject_misaligned_rows() -> None:
    with pytest.raises(ValueError):
        policy_sequences([[0]], [[1], [2]], [0.5], [False])


def test_hand_computed_gradient_and_loss_of_the_worked_example() -> None:
    """theta = (0, ln 2, ln 3), so pi = (1/6, 1/3, 1/2).

    dL/dtheta = -(1/N) sum_i A_i sum_t (e_y - pi). The advantage-weighted token count is
    sum_i A_i n_i = 0.5 x 2 - 0.5 x 1 + 0.5 x 1 - 0.5 x 1 = 0.5, so the weighted sum is
    S = 0.5 (e0 + e2) - 0.5 e1 + 0.5 e2 - 0.5 e0 - 0.5 pi
      = -0.5 e1 + e2 - 0.5 pi = (-1/12, -2/3, 3/4), and dL/dtheta = -S / 8
      = (1/96, 1/12, -3/32).
    L = -(1/8)(0.5 ln(1/6) + 0.5 ln(1/2) - 0.5 ln(1/3) + 0.5 ln(1/2) - 0.5 ln(1/6))
      = (ln 2 - 0.5 ln 3) / 8.
    """
    policy = TablePolicy([0.0, math.log(2), math.log(3)])
    sequences, denominator = worked_example()
    loss = policy_gradient_backward(
        policy, sequences, denominator=denominator, micro_batch=2, device=CPU
    )
    assert policy.theta.grad is not None
    expected = torch.tensor([1 / 96, 1 / 12, -3 / 32], dtype=torch.float64)
    torch.testing.assert_close(policy.theta.grad, expected, rtol=0, atol=1e-15)
    assert loss == pytest.approx((math.log(2) - 0.5 * math.log(3)) / 8, abs=1e-15)


def test_the_loss_matches_autograd_of_the_written_out_sum() -> None:
    """The same gradient from autograd on L written term by term, without padding."""
    policy = TablePolicy([0.3, -1.2, 0.7])
    sequences, denominator = worked_example()
    policy_gradient_backward(policy, sequences, denominator=denominator, micro_batch=1, device=CPU)
    assert policy.theta.grad is not None
    measured = policy.theta.grad.clone()
    theta = torch.tensor([0.3, -1.2, 0.7], dtype=torch.float64, requires_grad=True)
    log_pi = torch.log_softmax(theta, dim=0)
    total = sum(
        s.advantage * sum(log_pi[token] for token in s.tokens[s.prompt_length :]) for s in sequences
    )
    assert isinstance(total, Tensor)
    (-total / denominator).backward()  # type: ignore[no-untyped-call]
    assert theta.grad is not None
    torch.testing.assert_close(measured, theta.grad, rtol=0, atol=1e-15)


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(ModelConfig(vocab_size=272, width=16, layers=2, heads=2, context=24)).double()


def random_sequences(count: int, seed: int) -> tuple[list[PolicySequence], int]:
    generator = torch.Generator().manual_seed(seed)
    sequences = []
    for index in range(count):
        prompt = torch.randint(0, 272, (int(torch.randint(1, 8, (1,), generator=generator)),))
        attempt = torch.randint(0, 272, (int(torch.randint(1, 10, (1,), generator=generator)),))
        tokens = (*prompt.tolist(), *attempt.tolist())
        sequences.append(PolicySequence(tokens, len(prompt), (-1.0) ** index * (0.1 + index)))
    return sequences, sum(s.attempt_tokens for s in sequences) + 5


def reference_gradients(
    model: GPT, sequences: list[PolicySequence], denominator: int
) -> list[Tensor]:
    """Each sequence alone, unpadded, through the real causal model."""
    model.zero_grad()
    for sequence in sequences:
        ids = torch.tensor([sequence.tokens])
        logits = model(ids[:, :-1])
        chosen = token_log_probs(logits, ids[:, 1:])[0, sequence.prompt_length - 1 :]
        (-(sequence.advantage * chosen.sum()) / denominator).backward()  # type: ignore[no-untyped-call]
    return [p.grad.clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("micro_batch", [1, 2, 3, 7, 64])
def test_accumulated_micro_batches_equal_the_full_batch(micro_batch: int) -> None:
    """Right padding and the attempt mask leave every scored logit as the unpadded forward
    computes it, and the shared 1/N makes the accumulated gradient the full-batch one."""
    model = tiny_model()
    sequences, denominator = random_sequences(7, seed=3)
    expected = reference_gradients(model, sequences, denominator)
    model.zero_grad()
    policy_gradient_backward(
        model, sequences, denominator=denominator, micro_batch=micro_batch, device=CPU, pad_id=256
    )
    measured = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(measured) == len(expected)
    for left, right in zip(measured, expected, strict=True):
        # The model normalizes and takes the attention softmax in float32 even when its
        # weights are float64, so batching changes rounding at float32 level on some BLAS
        # builds (Linux, not macOS). A padding or masking error is orders of magnitude larger.
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)


def test_the_padding_token_does_not_change_the_gradient() -> None:
    sequences, denominator = random_sequences(5, seed=8)
    gradients = []
    for pad in (0, 256, 271):
        model = tiny_model()
        policy_gradient_backward(
            model, sequences, denominator=denominator, micro_batch=5, device=CPU, pad_id=pad
        )
        gradients.append([p.grad for p in model.parameters() if p.grad is not None])
    for other in gradients[1:]:
        for left, right in zip(gradients[0], other, strict=True):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-14)


def test_a_nonfinite_micro_batch_loss_raises_before_backward() -> None:
    policy = TablePolicy([0.0, 1.0, 2.0])

    def broken(ids: Tensor) -> Tensor:
        logits: Tensor = policy(ids)
        return logits * torch.tensor(math.nan, dtype=torch.float64)

    sequences, denominator = worked_example()
    with pytest.raises(FloatingPointError):
        policy_gradient_backward(
            broken, sequences, denominator=denominator, micro_batch=4, device=CPU
        )
    assert policy.theta.grad is None


def test_policy_gradient_rejects_bad_settings() -> None:
    policy = TablePolicy([0.0, 0.0, 0.0])
    sequences, _ = worked_example()
    for options in (
        {"denominator": 0, "micro_batch": 1},
        {"denominator": 4, "micro_batch": 0},
        {"denominator": 4, "micro_batch": 1, "precision": "float16"},
        {"denominator": 4, "micro_batch": 1, "temperature": 0.0},
    ):
        with pytest.raises(ValueError):
            policy_gradient_backward(policy, sequences, device=CPU, **options)


def test_cpu_bfloat16_autocast_stays_close_to_float32() -> None:
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=2, heads=2, context=24))
    sequences, denominator = random_sequences(4, seed=1)
    full = policy_gradient_backward(
        model, sequences, denominator=denominator, micro_batch=4, device=CPU
    )
    reference = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad()
    half = policy_gradient_backward(
        model, sequences, denominator=denominator, micro_batch=4, device=CPU, precision="bfloat16"
    )
    assert half == pytest.approx(full, rel=0.05, abs=0.05)
    measured = [p.grad for p in model.parameters() if p.grad is not None]
    for left, right in zip(measured, reference, strict=True):
        torch.testing.assert_close(left, right, rtol=0.1, atol=0.05)


def test_policy_sequence_validation() -> None:
    with pytest.raises(ValueError):
        PolicySequence((1, 2), 0, 0.5)
    with pytest.raises(ValueError):
        PolicySequence((1, 2), 3, 0.5)
    with pytest.raises(ValueError):
        PolicySequence((1, 2), 1, math.inf)
    assert PolicySequence((1, 2, 3), 1, 0.5).attempt_tokens == 2
