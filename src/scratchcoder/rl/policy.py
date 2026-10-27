"""Group-mean advantages and the token-normalized policy-gradient loss.

For one update, task ``k`` has ``G`` attempts with rewards ``r_k1 .. r_kG``. Its advantage
is ``A_ki = r_ki - mean_i(r_ki)``, with no division by the standard deviation. A group whose
rewards are all equal is skipped: it is left out of the loss and out of its denominator.

With ``y_ki`` the attempt tokens of attempt ``i`` (the generated tokens up to and including
the stop token when one was produced; never the prompt or padding) and ``N`` the number of
attempt tokens over every non-skipped group, the loss is::

    L = -(1 / N) * sum over non-skipped (k, i) of A_ki * sum over t of log pi(y_kit | prefix)

``log pi`` comes from the trainer's own forward pass over ``prompt + attempt`` on the
policy being updated, never from the sampler. ``policy_gradient_backward`` splits the
sequences into micro-batches and backpropagates each one's share of ``L`` with the same
``1 / N``, so the accumulated gradient equals one full-batch step. See
``docs/walkthroughs/rl-update.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Group:
    """One task's rewards in an update, their mean and advantages, and the skip flag."""

    task_id: str
    rewards: tuple[float, ...]
    mean: float
    advantages: tuple[float, ...]
    skipped: bool


def group_advantages(task_id: str, rewards: Sequence[float]) -> Group:
    """``A_i = r_i - mean(r)``; an all-equal group is skipped and its advantages are 0."""
    values = tuple(float(value) for value in rewards)
    if not values:
        raise ValueError("a group needs at least one reward")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("group rewards must be finite")
    mean = math.fsum(values) / len(values)
    skipped = all(value == values[0] for value in values)
    advantages = (0.0,) * len(values) if skipped else tuple(value - mean for value in values)
    return Group(task_id, values, mean, advantages, skipped)


@dataclass(frozen=True)
class PolicySequence:
    """``tokens`` is the prompt followed by the attempt; the first ``prompt_length`` tokens
    are the prompt. Every later token is an attempt token weighted by ``advantage``."""

    tokens: tuple[int, ...]
    prompt_length: int
    advantage: float

    def __post_init__(self) -> None:
        if type(self.prompt_length) is not int or not 1 <= self.prompt_length <= len(self.tokens):
            raise ValueError("a policy sequence needs a nonempty prompt within its tokens")
        if not math.isfinite(self.advantage):
            raise ValueError("an advantage must be finite")

    @property
    def attempt_tokens(self) -> int:
        return len(self.tokens) - self.prompt_length


def policy_sequences(
    prompts: Sequence[Sequence[int]],
    attempts: Sequence[Sequence[int]],
    advantages: Sequence[float],
    skipped: Sequence[bool],
) -> tuple[list[PolicySequence], int]:
    """The sequences the loss is taken over, and its denominator ``N``.

    Rows are aligned: attempt ``j`` has prompt ``prompts[j]``, generated tokens
    ``attempts[j]``, advantage ``advantages[j]`` and its group's skip flag ``skipped[j]``.
    ``N`` counts the attempt tokens of every attempt in a non-skipped group. A sequence
    with zero advantage or no attempt tokens contributes nothing to the gradient, so it is
    counted in ``N`` but not returned for a forward pass.
    """
    if not len(prompts) == len(attempts) == len(advantages) == len(skipped):
        raise ValueError("prompts, attempts, advantages and skip flags must be aligned")
    sequences: list[PolicySequence] = []
    denominator = 0
    for prompt, attempt, advantage, skip in zip(
        prompts, attempts, advantages, skipped, strict=True
    ):
        if skip:
            continue
        denominator += len(attempt)
        if attempt and advantage != 0.0:
            sequences.append(PolicySequence((*prompt, *attempt), len(prompt), float(advantage)))
    return sequences, denominator


def _microbatch(
    chunk: Sequence[PolicySequence], pad_id: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Right-padded inputs and targets [m, L], the attempt-target mask and the weights.

    Input position ``t`` predicts ``tokens[t + 1]``; it is an attempt target when
    ``tokens[t + 1]`` is an attempt token, that is ``prompt_length - 1 <= t``. Padding
    follows every real position, so causal attention keeps it from changing a scored logit.
    """
    width = max(len(sequence.tokens) for sequence in chunk) - 1
    inputs, targets, mask, weights = [], [], [], []
    for sequence in chunk:
        real = len(sequence.tokens) - 1
        gap = width - real
        inputs.append([*sequence.tokens[:-1], *([pad_id] * gap)])
        targets.append([*sequence.tokens[1:], *([pad_id] * gap)])
        row = [sequence.prompt_length - 1 <= position < real for position in range(width)]
        mask.append(row)
        weights.append([sequence.advantage if keep else 0.0 for keep in row])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.bool, device=device),
        torch.tensor(weights, dtype=torch.float64, device=device),
    )


def token_log_probs(logits: Tensor, targets: Tensor, *, temperature: float = 1.0) -> Tensor:
    """``log softmax(logits / T)`` at each target: [B, L, V] logits to [B, L].

    Half-precision logits are widened to float32 first; float64 logits stay float64.
    """
    wide = logits if logits.dtype == torch.float64 else logits.float()
    scaled = wide if temperature == 1.0 else wide / temperature
    return torch.log_softmax(scaled, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def policy_gradient_backward(
    forward: Callable[[Tensor], Tensor],
    sequences: Sequence[PolicySequence],
    *,
    denominator: int,
    micro_batch: int,
    device: torch.device,
    precision: str = "float32",
    pad_id: int = 0,
    temperature: float = 1.0,
) -> float:
    """Accumulate the gradient of ``L`` into the parameters ``forward`` uses; return ``L``.

    Sequences are sorted longest first (ties by input order) and split into micro-batches
    of at most ``micro_batch``. Each micro-batch backpropagates
    ``-(sum of A x log pi over its attempt targets) / N``, so the accumulated gradient is
    the full-batch gradient. ``forward`` maps ids [m, L] to logits [m, L, V]; with
    ``precision="bfloat16"`` it runs under bfloat16 autocast and the log-softmax stays in
    float32. A non-finite micro-batch loss raises ``FloatingPointError`` before its
    backward pass.
    """
    if type(denominator) is not int or denominator < 1:
        raise ValueError("the loss denominator must be a positive token count")
    if type(micro_batch) is not int or micro_batch < 1:
        raise ValueError("micro_batch must be a positive integer")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    order = sorted(range(len(sequences)), key=lambda index: (-len(sequences[index].tokens), index))
    total = 0.0
    for start in range(0, len(order), micro_batch):
        chunk = [sequences[index] for index in order[start : start + micro_batch]]
        inputs, targets, mask, weights = _microbatch(chunk, pad_id, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            logits = forward(inputs)
        chosen = token_log_probs(logits, targets, temperature=temperature)
        weighted = torch.where(mask, chosen * weights.to(chosen.dtype), torch.zeros_like(chosen))
        loss = -weighted.sum() / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite policy-gradient loss")
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
        total += float(loss.detach())
    return total
