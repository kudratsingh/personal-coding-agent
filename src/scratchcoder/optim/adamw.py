"""Hand-written AdamW, cross-entropy, scheduling, and gradient clipping."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def _finite_real(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_hyperparameters(lr: Any, betas: Any, eps: Any, weight_decay: Any) -> None:
    if (
        not _finite_real(lr)
        or lr < 0
        or not _finite_real(eps)
        or eps <= 0
        or not _finite_real(weight_decay)
        or weight_decay < 0
        or not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(not _finite_real(beta) or not 0 <= beta < 1 for beta in betas)
    ):
        raise ValueError("invalid optimizer hyperparameters")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Stable mean next-token loss without delegating to a loss framework.

    The log-probabilities are formed with the max-shifted log-softmax primitive
    and the target is gathered by hand. Mathematically this is the same
    ``log(sum(exp(x - max))) - (x_target - max)`` as the earlier explicit
    formulation, but autograd keeps one fp32 ``[B,T,V]`` tensor for backward
    instead of three (4 instead of 12 bytes per logit), which matters at a
    65,536-entry vocabulary. ``tests/test_optimizer.py`` checks both the
    reference values and the saved-memory bound.
    """
    if logits.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("logits/target shape mismatch or empty targets")
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    chosen = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -chosen.mean()


def learning_rate(
    step: int, *, total_steps: int, warmup_steps: int, peak: float, minimum_ratio: float = 0.1
) -> float:
    if not 0 <= step < total_steps or not 0 <= warmup_steps <= total_steps:
        raise ValueError("invalid schedule position")
    if not 0 <= minimum_ratio <= 1 or not 0 < peak < float("inf"):
        raise ValueError("invalid learning rate bounds")
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return peak * (minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def clip_gradients(parameters: Iterable[Tensor], maximum: float) -> float:
    if not 0 < maximum < float("inf"):
        raise ValueError("maximum norm must be finite and positive")
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    norm = math.sqrt(sum(float(gradient.detach().double().square().sum()) for gradient in grads))
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    scale = min(1.0, maximum / (norm + 1e-6))
    for gradient in grads:
        gradient.mul_(scale)
    return norm


class AdamW:
    def __init__(
        self,
        parameters: Iterable[Tensor],
        *,
        learning_rate: float = 0.0006,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        decay_mask: list[bool] | None = None,
    ) -> None:
        _validate_hyperparameters(learning_rate, betas, eps, weight_decay)
        self.parameters = list(parameters)
        mask = decay_mask if decay_mask is not None else [True] * len(self.parameters)
        if (
            not isinstance(mask, (list, tuple))
            or len(mask) != len(self.parameters)
            or any(type(flag) is not bool for flag in mask)
        ):
            raise ValueError("decay mask must contain one boolean per parameter")
        self.lr, self.eps, self.weight_decay = learning_rate, eps, weight_decay
        self.betas = (betas[0], betas[1])
        self.decay_mask = list(mask)
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.steps = [0] * len(self.parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = self.betas
        for i, parameter in enumerate(self.parameters):
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise ValueError("sparse gradients are unsupported")
            self.steps[i] += 1
            step = self.steps[i]
            if self.decay_mask[i]:
                parameter.mul_(1 - self.lr * self.weight_decay)
            self.m[i].mul_(beta1).add_(gradient, alpha=1 - beta1)
            self.v[i].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            denominator = self.v[i].sqrt() / math.sqrt(1 - beta2**step) + self.eps
            parameter.addcdiv_(self.m[i], denominator, value=-self.lr / (1 - beta1**step))

    def state_dict(self) -> dict[str, Any]:
        return {
            "m": [value.clone() for value in self.m],
            "v": [value.clone() for value in self.v],
            "steps": list(self.steps),
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "decay_mask": list(self.decay_mask),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        fields = {"m", "v", "steps", "lr", "betas", "eps", "weight_decay", "decay_mask"}
        if not isinstance(state, dict) or set(state) != fields:
            raise ValueError("invalid optimizer state fields")
        if any(
            not isinstance(state[name], list) or len(state[name]) != len(self.parameters)
            for name in ("m", "v", "steps", "decay_mask")
        ):
            raise ValueError("optimizer parameter count mismatch")
        if any(type(step) is not int or step < 0 for step in state["steps"]):
            raise ValueError("optimizer counters must be nonnegative integers")
        if any(type(flag) is not bool for flag in state["decay_mask"]):
            raise ValueError("optimizer decay mask must contain booleans")
        betas = state["betas"]
        _validate_hyperparameters(state["lr"], betas, state["eps"], state["weight_decay"])
        for name in ("m", "v"):
            if any(
                not isinstance(value, Tensor)
                or value.layout != torch.strided
                or value.is_meta
                or not value.is_floating_point()
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not bool(torch.isfinite(value).all())
                or (name == "v" and bool((value < 0).any()))
                for value, parameter in zip(state[name], self.parameters, strict=True)
            ):
                raise ValueError("optimizer moments require matching finite dense tensors; v >= 0")
        # Finish every copy/transfer before committing any live field.
        moments = {
            name: [
                value.detach().to(parameter.device).clone()
                for value, parameter in zip(state[name], self.parameters, strict=True)
            ]
            for name in ("m", "v")
        }
        steps, mask = list(state["steps"]), list(state["decay_mask"])
        new_betas = tuple(betas)
        self.m, self.v, self.steps, self.decay_mask = moments["m"], moments["v"], steps, mask
        self.lr, self.betas = state["lr"], new_betas
        self.eps, self.weight_decay = state["eps"], state["weight_decay"]
