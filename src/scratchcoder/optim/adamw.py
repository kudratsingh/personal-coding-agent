"""Hand-written AdamW, cross-entropy, scheduling, and gradient clipping."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Stable mean next-token loss without delegating to a loss framework."""
    if logits.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("logits/target shape mismatch or empty targets")
    shifted = logits.float() - logits.float().amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    chosen = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - chosen).mean()


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
        if not 0 <= learning_rate < float("inf") or not 0 < eps < float("inf"):
            raise ValueError("invalid optimizer rate/epsilon")
        if not all(0 <= beta < 1 for beta in betas) or not 0 <= weight_decay < float("inf"):
            raise ValueError("invalid optimizer decay/betas")
        self.parameters = list(parameters)
        self.lr, self.betas, self.eps, self.weight_decay = learning_rate, betas, eps, weight_decay
        self.decay_mask = decay_mask if decay_mask is not None else [True] * len(self.parameters)
        if len(self.decay_mask) != len(self.parameters):
            raise ValueError("decay mask must match parameters")
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
            "decay_mask": self.decay_mask,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            len(state["m"]) != len(self.parameters)
            or len(state["v"]) != len(self.parameters)
            or len(state["steps"]) != len(self.parameters)
        ):
            raise ValueError("optimizer parameter count mismatch")
        for name in ("m", "v"):
            values = state[name]
            if any(
                value.shape != parameter.shape
                for value, parameter in zip(values, self.parameters, strict=True)
            ):
                raise ValueError("optimizer state shape mismatch")
            setattr(
                self,
                name,
                [
                    value.to(parameter.device).clone()
                    for value, parameter in zip(values, self.parameters, strict=True)
                ],
            )
        self.steps = list(state["steps"])
        self.lr, self.betas = state["lr"], tuple(state["betas"])
        self.eps, self.weight_decay = state["eps"], state["weight_decay"]
        self.decay_mask = list(state["decay_mask"])
