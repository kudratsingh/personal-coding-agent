"""Readable decoder-only transformer built from tensor primitives."""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratchcoder.common.config import ModelConfig


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight)


def rotary(x: Tensor, *, offset: int = 0) -> Tensor:
    """Rotate adjacent feature pairs; x is [batch, heads, time, head_width]."""
    width = x.shape[-1]
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, width, 2, device=x.device).float() / width
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, *, offset: int = 0) -> Tensor:
    """q [B,H,T,D], k/v [B,H,S,D]; allow key positions <= absolute query position."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    query_positions = torch.arange(offset, offset + q.shape[-2], device=q.device)
    key_positions = torch.arange(k.shape[-2], device=q.device)
    mask = key_positions[None, :] > query_positions[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return probabilities @ v


def fused_causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Same mathematics as ``causal_attention`` without materializing ``[B,H,T,T]``.

    The scaled-dot-product primitive applies the causal mask, the ``1/sqrt(D)``
    scale and the softmax inside one kernel, so autograd keeps no score or
    probability matrices. It is only used for the uncached training/validation
    path (query and key positions coincide). ``tests/test_model.py`` checks its
    outputs and gradients against the reference formulation; the A40 preflight
    repeats that comparison on the actual device. Fused CUDA backward kernels
    may not be bitwise deterministic across runs.
    """
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("fused attention requires equal query and key lengths")
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.fused = False  # Set through GPT.set_fused_attention; never persisted.
        # Optional per-head query/key normalization keeps attention logits bounded
        # at high learning rates (Wortsman et al. 2023); gains are shared by heads.
        self.q_norm = RMSNorm(config.width // config.heads) if config.qk_norm else None
        self.k_norm = RMSNorm(config.width // config.heads) if config.qk_norm else None

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_cached(x, None)[0]

    def forward_cached(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, time, width = x.shape
        offset = 0 if cache is None else cache[0].shape[-2]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        k = k.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = rotary(q, offset=offset)
        k = rotary(k, offset=offset)
        v = v.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        if self.fused and cache is None:
            mixed = fused_causal_attention(q, k, v)
        else:
            mixed = causal_attention(q, k, v, offset=offset)
        attended = mixed.transpose(1, 2).contiguous().view(batch, time, width)
        return cast(Tensor, self.output(attended)), (k, v)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = RMSNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width, bias=False)
        self.down = nn.Linear(4 * config.width, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_cached(x, None)[0]

    def forward_cached(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, updated = self.attention.forward_cached(self.norm1(x), cache)
        x = x + attended
        hidden = self.up(self.norm2(x))
        # Exact GELU expressed with erf rather than a ready-made transformer component.
        activated = hidden * 0.5 * (1 + torch.erf(hidden / math.sqrt(2)))
        return cast(Tensor, x + self.down(activated)), updated


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.fused_attention = False
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
        # GPT-2 depth scaling: the two projections that write into the residual
        # stream start at std 0.02 / sqrt(2L), so the residual variance does not
        # grow with depth at initialization.
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        for block in self.blocks:
            nn.init.normal_(cast(Block, block).attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(cast(Block, block).down.weight, mean=0.0, std=residual_std)

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or ids.shape[1] > self.config.context or ids.shape[1] == 0:
            raise ValueError("tokens must be nonempty [batch,time] within context")
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        # Shared embedding/output matrix; logits are [batch,time,vocabulary].
        return cast(Tensor, self.norm(x) @ self.embedding.weight.T)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_fused_attention(self, enabled: bool) -> None:
        """Select the fused kernel for the uncached path; cached generation stays reference."""
        if type(enabled) is not bool:
            raise ValueError("fused attention flag must be a boolean")
        self.fused_attention = enabled
        for block in self.blocks:
            cast(Block, block).attention.fused = enabled

    def forward_cached(
        self, ids: Tensor, cache: list[tuple[Tensor, Tensor]] | None = None
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("tokens must be nonempty [batch,time]")
        if cache is not None and len(cache) != self.config.layers:
            raise ValueError("cache layer count differs")
        offset = 0 if cache is None else cache[0][0].shape[-2]
        if offset + ids.shape[1] > self.config.context:
            raise ValueError("cached sequence exceeds context")
        if cache is not None and any(
            key.shape[-2] != offset or key.shape[0] != ids.shape[0] or value.shape != key.shape
            for key, value in cache
        ):
            raise ValueError("inconsistent cache dimensions")
        x = self.embedding(ids)
        updated: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            x, layer = cast(Block, block).forward_cached(x, None if cache is None else cache[index])
            updated.append(layer)
        return cast(Tensor, self.norm(x) @ self.embedding.weight.T), updated


def expected_parameters(config: ModelConfig) -> int:
    qk_gains = 2 * (config.width // config.heads) if config.qk_norm else 0
    return (
        config.vocab_size * config.width
        + config.layers * (12 * config.width**2 + 2 * config.width + qk_gains)
        + config.width
    )
