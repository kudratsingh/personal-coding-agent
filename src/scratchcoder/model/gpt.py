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


def rotary_table(
    width: int, start: int, stop: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Float32 cosine and sine [stop-start, width/2] for positions start..stop-1."""
    positions = torch.arange(start, stop, device=device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, width, 2, device=device).float() / width
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate adjacent feature pairs of x [..., T, D] by angles broadcast as [..., T, D/2]."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def rotary(x: Tensor, *, offset: int = 0) -> Tensor:
    """Rotate adjacent feature pairs; x is [batch, heads, time, head_width]."""
    cosine, sine = rotary_table(x.shape[-1], offset, offset + x.shape[-2], device=x.device)
    return apply_rotary(x, cosine.to(x.dtype), sine.to(x.dtype))


def masked_attention(q: Tensor, k: Tensor, v: Tensor, hidden: Tensor) -> Tensor:
    """q [B,H,T,D], k/v [B,H,S,D]; ``hidden`` (broadcast to [B,H,T,S]) marks unseen keys."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(hidden, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return probabilities @ v


def causal_attention(q: Tensor, k: Tensor, v: Tensor, *, offset: int = 0) -> Tensor:
    """q [B,H,T,D], k/v [B,H,S,D]; allow key positions <= absolute query position."""
    query_positions = torch.arange(offset, offset + q.shape[-2], device=q.device)
    key_positions = torch.arange(k.shape[-2], device=q.device)
    return masked_attention(q, k, v, key_positions[None, :] > query_positions[:, None])


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


class KVCache:
    """Preallocated keys and values for left-padded rows that decode in lockstep.

    Every row occupies the same slots ``0..capacity-1``. Row ``b`` starts with
    ``pad[b]`` padding slots, so slot ``s`` holds its token at position
    ``s - pad[b]``. The rotary angles of each row's positions and the masks are
    built once here; a decoding step only slices them.
    """

    def __init__(self, config: ModelConfig, pad: Tensor, *, width: int, capacity: int) -> None:
        if pad.ndim != 1 or pad.numel() == 0 or pad.dtype != torch.long:
            raise ValueError("padding must be a nonempty int64 [batch] tensor")
        if not 0 < width <= capacity:
            raise ValueError("prompt width must be positive and fit the capacity")
        self.batch, self.width, self.capacity, self.length = pad.numel(), width, capacity, 0
        slots = torch.arange(capacity, device=pad.device)
        # The capacity can exceed the context (a long prompt beside a long budget), so
        # slot numbers could pass the table; each row's own positions stay inside it
        # while the row is active. Positions past the context occur only in rows that
        # already finished; the clamp keeps their (discarded) work finite and in range.
        positions = (slots[None, :] - pad[:, None]).clamp(0, config.context - 1)
        cosine, sine = rotary_table(
            config.width // config.heads, 0, config.context, device=pad.device
        )
        # Per-row angles [B, capacity, D/2], cast once per activation dtype.
        self.angles = {torch.float32: (cosine[positions], sine[positions])}
        self.padding = slots[None, :] < pad[:, None]  # [B, capacity]
        self.future = slots[None, :] > slots[:, None]  # [capacity, capacity]
        self.keys: list[Tensor | None] = [None] * config.layers
        self.values: list[Tensor | None] = [None] * config.layers

    def rotation(self, time: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Cosine and sine [B,1,T,D/2] of the next ``time`` slots, in ``dtype``."""
        if dtype not in self.angles:
            cosine, sine = self.angles[torch.float32]
            self.angles[dtype] = cosine.to(dtype), sine.to(dtype)
        cosine, sine = self.angles[dtype]
        stop = self.length + time
        return cosine[:, None, self.length : stop], sine[:, None, self.length : stop]

    def hidden(self, time: int) -> Tensor:
        """Keys that the next ``time`` query slots must not see, as [B,1,T,S]."""
        start, stop = self.length, self.length + time
        if time == 1 and start >= self.width:
            # Decoding: the query is a real token and every earlier slot is in its past.
            return self.padding[:, None, None, :stop]
        # Real queries skip padding keys. A padding query sees its own causal past
        # instead, so its softmax row is never empty; real rows never read its output.
        skip = self.padding[:, None, :stop] & ~self.padding[:, start:stop, None]
        return (self.future[start:stop, :stop] | skip)[:, None]

    def store(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write k/v [B,H,T,D] at the next slots; return the filled [B,H,length+T,D] views.

        Both buffers use the dtype of ``v``, the activation dtype. Under autocast,
        query/key normalization's float32 gain promotes ``k`` to float32; attention
        would round it to the activation dtype at every step, so it is rounded once
        here, with the same result, instead of keeping a float32 key cache.
        """
        stop = self.length + k.shape[-2]
        keys, values = self.keys[layer], self.values[layer]
        if keys is None or values is None:
            keys = v.new_empty((*k.shape[:2], self.capacity, k.shape[-1]))
            values = v.new_empty((*v.shape[:2], self.capacity, v.shape[-1]))
            self.keys[layer], self.values[layer] = keys, values
        keys[:, :, self.length : stop] = k.to(keys.dtype)
        values[:, :, self.length : stop] = v
        return keys[:, :, :stop], values[:, :, :stop]


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

    def decode(self, x: Tensor, cache: KVCache, layer: int) -> Tensor:
        """Attend the next ``T`` slots of x [B,T,C] through a preallocated cache."""
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        k = k.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary(q, *cache.rotation(time, q.dtype))
        k = apply_rotary(k, *cache.rotation(time, k.dtype))
        v = v.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
        keys, values = cache.store(layer, k, v)
        mixed = masked_attention(q, keys, values, cache.hidden(time))
        attended = mixed.transpose(1, 2).contiguous().view(batch, time, width)
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
        return self.forward_cached(x, None)[0]

    def forward_cached(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, updated = self.attention.forward_cached(self.norm1(x), cache)
        return self.feed_forward(x + attended), updated

    def decode(self, x: Tensor, cache: KVCache, layer: int) -> Tensor:
        return self.feed_forward(x + self.attention.decode(self.norm1(x), cache, layer))

    def feed_forward(self, x: Tensor) -> Tensor:
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

    def decode(self, ids: Tensor, cache: KVCache) -> Tensor:
        """Run ids [B,T] at the cache's next slots; return float32 logits [B,V] of the last.

        Only the last slot reaches the vocabulary projection, so a long prefill never
        materializes ``[B,T,V]``. That projection runs in float32 even under bfloat16
        autocast, so sampling sees unrounded logits.
        """
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.shape[0] != cache.batch:
            raise ValueError("tokens must be nonempty [batch,time] matching the cache")
        if cache.length + ids.shape[1] > cache.capacity:
            raise ValueError("decoding exceeds the cache capacity")
        x = self.embedding(ids)
        for index, block in enumerate(self.blocks):
            x = cast(Block, block).decode(x, cache, index)
        cache.length += ids.shape[1]
        last = self.norm(x[:, -1])
        with torch.autocast(device_type=last.device.type, enabled=False):
            return cast(Tensor, last.float() @ self.embedding.weight.float().T)


def expected_parameters(config: ModelConfig) -> int:
    qk_gains = 2 * (config.width // config.heads) if config.qk_norm else 0
    return (
        config.vocab_size * config.width
        + config.layers * (12 * config.width**2 + 2 * config.width + qk_gains)
        + config.width
    )
