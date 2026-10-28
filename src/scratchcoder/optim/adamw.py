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


def _broadcasts(scale: Tensor, parameter: Tensor) -> bool:
    """Whether ``scale`` broadcasts to ``parameter``'s shape without growing it."""
    if scale.ndim > parameter.ndim:
        return False
    return all(
        size in (1, full)
        for size, full in zip(reversed(scale.shape), reversed(parameter.shape), strict=False)
    )


def _scaled(moment: Tensor, scale: Tensor | None) -> Tensor:
    """The first moment a parameter steps by: itself, or times its learning-rate scale.

    Multiplying by 1.0 is exact in floating point, so a scale of ones gives the plain step
    bit for bit; ``None`` skips the multiply altogether."""
    return moment if scale is None else moment * scale


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


def cross_entropy_sum(
    logits: Tensor, targets: Tensor, ignore_index: int = -100
) -> tuple[Tensor, Tensor]:
    """Summed next-token loss over targets that are not ``ignore_index``, and their count.

    The caller chooses the denominator. A one-pass final update divides by its
    full row capacity, not by the count, so a padded update gives each real target
    the same gradient weight as every other update. Ignored positions gather class
    zero and are then zeroed, which keeps the gather in range and gives them no
    gradient; when every target is ignored the sum is an exact zero that still
    backpropagates. Autograd keeps one fp32 ``[B,T,V]`` tensor, as for
    ``cross_entropy``.
    """
    if logits.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("logits/target shape mismatch or empty targets")
    keep = targets != ignore_index
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    chosen = log_probabilities.gather(-1, targets.masked_fill(~keep, 0).unsqueeze(-1)).squeeze(-1)
    return (-chosen).masked_fill(~keep, 0.0).sum(), keep.sum()


def cross_entropy_and_z(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Mean next-token loss and the z-loss term ``mean(log Z^2)``.

    ``log Z`` is the log-partition function of each position's softmax. Adding
    a small multiple of its square (``1e-4`` in Wortsman et al. 2023 and PaLM)
    keeps output logits from drifting without changing the predicted
    distribution. Autograd keeps one fp32 ``[B,T,V]`` tensor, as for
    ``cross_entropy``.
    """
    if logits.shape[:-1] != targets.shape or targets.numel() == 0:
        raise ValueError("logits/target shape mismatch or empty targets")
    values = logits.float()
    log_partition = torch.logsumexp(values, dim=-1)
    chosen = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - chosen).mean(), log_partition.square().mean()


def learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum_ratio: float = 0.1,
    schedule: str = "cosine",
    decay_fraction: float = 0.2,
) -> float:
    """Warmup then cosine decay, or warmup-stable-decay (WSD) with a linear tail.

    WSD holds the peak after warmup and decays linearly over the final
    ``decay_fraction`` of the run, which lets a stable-phase checkpoint be
    continued without re-warming and is the schedule used by recent small-model
    pretraining such as SmolLM2 and MiniCPM.
    """
    if not 0 <= step < total_steps or not 0 <= warmup_steps <= total_steps:
        raise ValueError("invalid schedule position")
    if not 0 <= minimum_ratio <= 1 or not 0 < peak < float("inf"):
        raise ValueError("invalid learning rate bounds")
    if schedule not in {"cosine", "wsd"}:
        raise ValueError("schedule must be cosine or wsd")
    if not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in (0, 1]")
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if schedule == "wsd":
        decay_steps = max(1, round(decay_fraction * total_steps))
        decay_start = max(warmup_steps, total_steps - decay_steps)
        if step < decay_start:
            return peak
        progress = (step - decay_start) / max(1, total_steps - decay_start - 1)
        return peak * (1 - (1 - minimum_ratio) * min(1.0, progress))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return peak * (minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def clip_gradients(parameters: Iterable[Tensor], maximum: float) -> float:
    """Scale gradients to a global L2 norm; one device sync for all tensors."""
    if not 0 < maximum < float("inf"):
        raise ValueError("maximum norm must be finite and positive")
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    # Per-tensor float64 partial sums are reduced on the device, so the only
    # host synchronization is the final scalar read, not one per parameter.
    squares = torch.stack([gradient.detach().double().square().sum() for gradient in grads])
    norm = float(squares.sum().sqrt())
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
        foreach: bool = False,
        lr_scales: list[Tensor | None] | None = None,
    ) -> None:
        """``lr_scales`` gives a parameter its own per-element learning-rate multiplier: a
        finite, nonnegative tensor broadcastable to the parameter (``[V, 1]`` scales the rows
        of a ``[V, D]`` embedding), or ``None`` for the plain update. Only a parameter without
        weight decay may carry one, so decoupled decay never needs rescaling. Like
        ``foreach`` it is a construction setting, not optimizer state: a resumed trainer
        builds it again from its configuration."""
        _validate_hyperparameters(learning_rate, betas, eps, weight_decay)
        if type(foreach) is not bool:
            raise ValueError("foreach must be a boolean")
        self.foreach = foreach
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
        scales = lr_scales if lr_scales is not None else [None] * len(self.parameters)
        if not isinstance(scales, (list, tuple)) or len(scales) != len(self.parameters):
            raise ValueError("learning-rate scales must give one entry per parameter")
        owned: list[Tensor | None] = []
        for scale, parameter, decayed in zip(scales, self.parameters, mask, strict=True):
            if scale is None:
                owned.append(None)
                continue
            if (
                not isinstance(scale, Tensor)
                or not scale.is_floating_point()
                or scale.layout != torch.strided
                or not _broadcasts(scale, parameter)
                or not bool(torch.isfinite(scale).all())
                or bool((scale < 0).any())
            ):
                raise ValueError(
                    "a learning-rate scale is a finite nonnegative float tensor that "
                    "broadcasts to its parameter"
                )
            if decayed:
                raise ValueError("a learning-rate scale applies only to a parameter without decay")
            owned.append(scale.detach().to(device=parameter.device, dtype=parameter.dtype).clone())
        self.lr_scales = owned
        self.m = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.v = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.steps = [0] * len(self.parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        if self.foreach:
            self._step_foreach()
            return
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
            parameter.addcdiv_(
                _scaled(self.m[i], self.lr_scales[i]),
                denominator,
                value=-self.lr / (1 - beta1**step),
            )

    @torch.no_grad()
    def _step_foreach(self) -> None:
        """The same update as ``step`` issued as one kernel per operation.

        Each elementwise operation runs over the whole parameter list at once
        instead of once per tensor, which removes most launch overhead on a GPU.
        ``tests/test_optimizer.py`` checks the result against the per-tensor
        loop exactly. Bias corrections stay per tensor because a parameter
        that skipped an update keeps its own step count. A parameter with a
        learning-rate scale steps by its scaled first moment, as in ``step``.
        """
        beta1, beta2 = self.betas
        indices = [i for i, parameter in enumerate(self.parameters) if parameter.grad is not None]
        if not indices:
            return
        grads: list[Tensor] = []
        for i in indices:
            gradient = self.parameters[i].grad
            assert gradient is not None
            if gradient.is_sparse:
                raise ValueError("sparse gradients are unsupported")
            grads.append(gradient)
            self.steps[i] += 1
        params = [self.parameters[i] for i in indices]
        first = [self.m[i] for i in indices]
        second = [self.v[i] for i in indices]
        decayed = [self.parameters[i] for i in indices if self.decay_mask[i]]
        if decayed:
            torch._foreach_mul_(decayed, 1 - self.lr * self.weight_decay)
        torch._foreach_mul_(first, beta1)
        torch._foreach_add_(first, grads, alpha=1 - beta1)
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, grads, grads, value=1 - beta2)
        denominators = torch._foreach_sqrt(second)
        torch._foreach_div_(denominators, [math.sqrt(1 - beta2 ** self.steps[i]) for i in indices])
        torch._foreach_add_(denominators, self.eps)
        numerators = [
            _scaled(moment, self.lr_scales[i]) for i, moment in zip(indices, first, strict=True)
        ]
        torch._foreach_addcdiv_(
            params,
            numerators,
            denominators,
            [-self.lr / (1 - beta1 ** self.steps[i]) for i in indices],
        )

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
