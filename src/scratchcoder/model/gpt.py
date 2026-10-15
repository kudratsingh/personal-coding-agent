"""Readable decoder-only transformer built from tensor primitives."""

import math
from typing import cast

import torch
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


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rotary(q.view(batch, time, self.heads, width // self.heads).transpose(1, 2))
        k = rotary(k.view(batch, time, self.heads, width // self.heads).transpose(1, 2))
        v = v.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        attended = causal_attention(q, k, v).transpose(1, 2).contiguous().view(batch, time, width)
        return cast(Tensor, self.output(attended))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = RMSNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width, bias=False)
        self.down = nn.Linear(4 * config.width, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        hidden = self.up(self.norm2(x))
        # Exact GELU expressed with erf rather than a ready-made transformer component.
        activated = hidden * 0.5 * (1 + torch.erf(hidden / math.sqrt(2)))
        return cast(Tensor, x + self.down(activated))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

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


def expected_parameters(config: ModelConfig) -> int:
    return (
        config.vocab_size * config.width
        + config.layers * (12 * config.width**2 + 2 * config.width)
        + config.width
    )
