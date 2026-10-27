"""Seeded sampling, batched cached generation, and batched continuation scoring."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from scratchcoder.common.config import finite_number
from scratchcoder.model.gpt import GPT, KVCache

SAMPLER_VERSION = "philox4x32-10-inverse-cdf-v1"
"""Names the seeded sampler below; record it with every sampled result."""

PRECISIONS = ("float32", "bfloat16")
# Temperatures this small scale in float64: (logit - max) / T stays exact there.
TINY_TEMPERATURE = 2.0**-100
# Prefill chunks keep each layer's attention scores below this many elements.
PREFILL_SCORE_ELEMENTS = 2**27
_MASK32 = 0xFFFFFFFF
_PHILOX_MULTIPLIERS = (0xD2511F53, 0xCD9E8D57)
_PHILOX_WEYL = (0x9E3779B9, 0xBB67AE85)


def _multiply_high_low(multiplier: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """High and low 32-bit words of ``multiplier * x`` for x in [0, 2^32).

    The 64-bit product is assembled from 16-bit halves, so no int64 intermediate
    exceeds 2^34 and nothing relies on signed overflow.
    """
    high, low = multiplier >> 16, multiplier & 0xFFFF
    x_high, x_low = x >> 16, x & 0xFFFF
    middle = high * x_low + low * x_high
    bottom = low * x_low + ((middle & 0xFFFF) << 16)
    top = high * x_high + (middle >> 16) + (bottom >> 32)
    return top & _MASK32, bottom & _MASK32


def philox4x32(
    counter: tuple[Tensor, Tensor, Tensor, Tensor], key: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Philox4x32-10 (Salmon et al. 2011) on broadcastable int64 tensors of 32-bit words."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(10):
        if round_index:
            k0, k1 = (k0 + _PHILOX_WEYL[0]) & _MASK32, (k1 + _PHILOX_WEYL[1]) & _MASK32
        high0, low0 = _multiply_high_low(_PHILOX_MULTIPLIERS[0], c0)
        high1, low1 = _multiply_high_low(_PHILOX_MULTIPLIERS[1], c2)
        c0, c1, c2, c3 = high1 ^ c1 ^ k0, low1, high0 ^ c3 ^ k1, low0
    return c0, c1, c2, c3


def sampling_uniforms(seeds: Tensor, steps: int, *, first: int = 0) -> Tensor:
    """Float64 uniforms [B, steps] in [0, 1); entry (b, t) depends only on (seeds[b], first + t).

    The key is the row's 64-bit seed and the counter is ``(step, 0, 0, 0)``; two
    output words give 53 random bits. No generator state exists, so a row draws
    the same numbers whatever batch it shares.
    """
    if seeds.ndim != 1 or seeds.dtype != torch.long:
        raise ValueError("seeds must be an int64 [batch] tensor")
    if type(steps) is not int or type(first) is not int or steps < 0 or first < 0:
        raise ValueError("steps and first must be nonnegative integers")
    if first + steps > 2**32:
        raise ValueError("sampling step exceeds the 32-bit counter")
    key = (seeds[:, None] & _MASK32, seeds[:, None] >> 32)
    step = torch.arange(first, first + steps, dtype=torch.long, device=seeds.device)[None, :]
    zero = torch.zeros_like(step)
    words = philox4x32((step, zero, zero, zero), key)
    bits = (words[0] >> 5) * 2**26 + (words[1] >> 6)
    return bits.double() / 2.0**53


def _controls(
    temperature: float, top_k: int | None, top_p: float
) -> tuple[float, int | None, float]:
    temperature = finite_number(temperature, "temperature")
    top_p = finite_number(top_p, "top_p")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid temperature/top_p")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    return temperature, top_k, top_p


def _check_logits(logits: Tensor) -> None:
    if (
        logits.layout != torch.strided
        or logits.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or logits.ndim not in {1, 2}
        or logits.numel() == 0
    ):
        raise ValueError("sampling logits must be nonempty dense floating [vocab] or [batch,vocab]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("sampling logits must be finite")


def _filters(vocabulary: int, top_k: int | None, top_p: float) -> bool:
    return (top_k is not None and top_k < vocabulary) or top_p < 1


def _weights(logits: Tensor, temperature: float, top_k: int | None, top_p: float) -> Tensor:
    """Scaled, filtered log weights for a positive temperature; removed tokens are -inf."""
    wide = logits.dtype == torch.float64 or temperature < TINY_TEMPERATURE
    scaled = logits.double() if wide else logits.float()
    if temperature >= 1:
        # Divide first: |logit / T| <= |logit|, so opposite finite extremes cannot overflow.
        scaled = scaled / temperature
    else:
        # Center first: the maximum stays an exact zero and overflow can only reach -inf,
        # the limit of a negligible probability.
        scaled = (scaled - scaled.amax(dim=-1, keepdim=True)) / temperature
    vocabulary = scaled.shape[-1]
    if top_k is not None and top_k < vocabulary:
        values, indices = scaled.topk(top_k, dim=-1)
        scaled = torch.full_like(scaled, float("-inf")).scatter(-1, indices, values)
    if top_p < 1:
        # The nucleus keeps each token whose higher-ranked mass is still below top_p.
        values, order = scaled.sort(dim=-1, descending=True, stable=True)
        probabilities = values.softmax(dim=-1)
        removed = probabilities.cumsum(dim=-1) - probabilities >= top_p
        scaled = scaled.masked_fill(
            torch.zeros_like(removed).scatter(-1, order, removed), -torch.inf
        )
    return scaled


def inverse_cdf(weights: Tensor, uniform: Tensor, *, filtered: bool = True) -> Tensor:
    """Token [..., 1] whose cumulative probability first exceeds ``uniform * total``.

    ``weights`` are log weights [..., V] and ``uniform`` is float64 [...] in [0, 1).
    The cumulative sum is float64 in vocabulary order. With ``filtered``, entries of
    zero probability are pinned to the previous running maximum, so a parallel scan's
    rounding can never select a removed token.
    """
    probabilities = weights.softmax(dim=-1)
    cumulative = probabilities.cumsum(dim=-1, dtype=torch.float64)
    if filtered:
        cumulative = cumulative.masked_fill(probabilities == 0, -torch.inf).cummax(dim=-1).values
    total = cumulative[..., -1:]
    # Stay strictly below the total so rounding can never step past the last kept token.
    below = torch.nextafter(total, torch.zeros_like(total))
    target = torch.minimum(uniform[..., None] * total, below)
    return torch.searchsorted(cumulative, target, right=True).clamp_(max=weights.shape[-1] - 1)


def _choose(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float,
    uniform: Tensor | None,
    log_probs: bool,
) -> tuple[Tensor, Tensor | None]:
    """Token [..., 1] and, on request, its float32 log probability under the sampler."""
    if temperature == 0 or uniform is None:
        token = logits.argmax(dim=-1, keepdim=True)
        # Greedy decoding samples from a point mass: its chosen token has log probability 0.
        return token, (torch.zeros(token.shape, device=token.device) if log_probs else None)
    weights = _weights(logits, temperature, top_k, top_p)
    token = inverse_cdf(weights, uniform, filtered=_filters(weights.shape[-1], top_k, top_p))
    if not log_probs:
        return token, None
    chosen = weights.gather(-1, token) - weights.logsumexp(dim=-1, keepdim=True)
    return token, chosen.float()


def _seed_tensor(seeds: Sequence[int], device: torch.device) -> Tensor:
    if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("seeds must be integers in [0, 2^63)")
    return torch.tensor(list(seeds), dtype=torch.long, device=device)


def sampling_log_probs(
    logits: Tensor, *, temperature: float, top_k: int | None = None, top_p: float = 1.0
) -> Tensor:
    """Float log probabilities [..., V] of the distribution ``sample`` draws from."""
    temperature, top_k, top_p = _controls(temperature, top_k, top_p)
    _check_logits(logits)
    if temperature == 0:
        point = torch.full(logits.shape, -torch.inf, device=logits.device)
        return point.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return _weights(logits, temperature, top_k, top_p).log_softmax(dim=-1)


def sample(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seed: int | Sequence[int] = 0,
    step: int = 0,
) -> Tensor:
    """Choose one token per row of [V] or [B,V] logits, returned as [..., 1].

    Temperature zero is greedy (the first maximal logit). Otherwise row ``r`` draws
    the uniform of ``(seed_r, step)`` and inverts the cumulative distribution of its
    filtered probabilities. An integer seed gives row ``r`` the seed ``seed + r``.
    """
    temperature, top_k, top_p = _controls(temperature, top_k, top_p)
    _check_logits(logits)
    if type(step) is not int or not 0 <= step < 2**32:
        raise ValueError("step must be an integer in [0, 2^32)")
    rows = 1 if logits.ndim == 1 else logits.shape[0]
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer or one integer per row")
    seeds = [seed + row for row in range(rows)] if isinstance(seed, int) else list(seed)
    if len(seeds) != rows:
        raise ValueError("sampling needs one seed per row")
    uniform = sampling_uniforms(_seed_tensor(seeds, logits.device), 1, first=step)[:, 0]
    token, _ = _choose(
        logits,
        temperature,
        top_k,
        top_p,
        uniform[0] if logits.ndim == 1 else uniform,
        False,
    )
    return token


@dataclass(frozen=True)
class Completion:
    """One prompt's new tokens; a stop token, when produced, is the last token.

    ``finish_reason`` is ``"stop"`` after a stop token and ``"length"`` once the
    row's ``budget`` (``min(max_new_tokens, context - prompt length)``) is spent.
    ``log_probs`` holds each token's log probability under the sampling
    distribution actually used (0 for greedy), when requested.
    """

    tokens: list[int]
    finish_reason: str
    budget: int
    log_probs: list[float] | None = None


def _decode(
    model: GPT,
    ids: Tensor,
    pad: Tensor,
    budgets: list[int],
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seeds: Tensor | None,
    stops: Tensor | None,
    log_probs: bool,
    precision: str,
    prefill_chunk: int | None,
    stop_check_interval: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Decode left-padded ids [B,W]; return tokens [B,N], counts [B], stopped [B], log probs.

    The loop launches device work only: stops and budgets update device flags, and
    the host reads whether any row is still active once every
    ``stop_check_interval`` steps. Finiteness is accumulated per row and checked
    once at the end.
    """
    batch, width = ids.shape
    steps, device = max(budgets), ids.device
    tokens = torch.zeros((batch, steps), dtype=torch.long, device=device)
    chosen = torch.zeros((batch, steps), device=device) if log_probs else None
    counts = torch.zeros(batch, dtype=torch.long, device=device)
    stopped = torch.zeros(batch, dtype=torch.bool, device=device)
    if steps == 0:
        return tokens, counts, stopped, chosen
    budget = torch.tensor(budgets, dtype=torch.long, device=device)
    active, nonfinite = budget > 0, torch.zeros_like(stopped)
    draws = None if temperature == 0 or seeds is None else sampling_uniforms(seeds, steps)
    # The last chosen token is never fed back, so steps - 1 decoding slots follow the prompt.
    cache = KVCache(model.config, pad, width=width, capacity=width + steps - 1)
    chunk = prefill_chunk or max(
        1, PREFILL_SCORE_ELEMENTS // (batch * model.config.heads * cache.capacity)
    )
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
    ):
        for start in range(0, width, chunk):
            logits = model.decode(ids[:, start : start + chunk], cache)
        for step in range(steps):
            with torch.autocast(device_type=device.type, enabled=False):
                nonfinite |= active & ~torch.isfinite(logits).all(dim=-1)
                uniform = None if draws is None else draws[:, step]
                token, log_prob = _choose(logits, temperature, top_k, top_p, uniform, log_probs)
                token = torch.where(active, token[:, 0], 0)
                tokens[:, step] = token
                if chosen is not None and log_prob is not None:
                    chosen[:, step] = torch.where(active, log_prob[:, 0], 0.0)
                counts += active
                if stops is not None:
                    hit = active & (token[:, None] == stops).any(dim=-1)
                    stopped |= hit
                    active &= ~hit
                active &= counts < budget
            if step + 1 == steps:
                break
            if (step + 1) % stop_check_interval == 0 and not bool(active.any()):
                break
            logits = model.decode(token[:, None], cache)
    if bool(nonfinite.any()):
        raise ValueError("sampling logits must be finite")
    used = int(counts.max())
    return tokens[:, :used], counts, stopped, None if chosen is None else chosen[:, :used]


def _options(
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    precision: str,
    prefill_chunk: int | None,
    stop_check_interval: int,
) -> tuple[float, int | None, float]:
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if precision not in PRECISIONS:
        raise ValueError("precision must be float32 or bfloat16")
    if prefill_chunk is not None and (type(prefill_chunk) is not int or prefill_chunk < 1):
        raise ValueError("prefill_chunk must be a positive integer")
    if type(stop_check_interval) is not int or stop_check_interval < 1:
        raise ValueError("stop_check_interval must be a positive integer")
    return _controls(temperature, top_k, top_p)


@torch.no_grad()
def generate_batch(
    model: GPT,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seeds: Sequence[int] | None = None,
    stops: Sequence[Collection[int]] | None = None,
    log_probs: bool = False,
    precision: str = "float32",
    prefill_chunk: int | None = None,
    stop_check_interval: int = 8,
) -> list[Completion]:
    """Generate for prompts of different lengths as one left-padded batch.

    Row ``r`` has its own seed ``seeds[r]`` (required when sampling), its own stop
    token set ``stops[r]`` and its own budget ``min(max_new_tokens, context -
    len(prompts[r]))``, so its output does not depend on the other rows.
    ``precision="bfloat16"`` runs the transformer under bfloat16 autocast; the
    vocabulary projection and the sampler stay in float32/float64.
    """
    temperature, top_k, top_p = _options(
        max_new_tokens, temperature, top_k, top_p, precision, prefill_chunk, stop_check_interval
    )
    if type(log_probs) is not bool:
        raise ValueError("log_probs must be a boolean")
    config, device = model.config, next(model.parameters()).device
    rows = [list(prompt) for prompt in prompts]
    if not rows or any(not row for row in rows):
        raise ValueError("prompts must be a nonempty list of nonempty token lists")
    if any(
        type(token) is not int or not 0 <= token < config.vocab_size
        for row in rows
        for token in row
    ):
        raise ValueError("prompt tokens must be vocabulary IDs")
    if any(len(row) > config.context for row in rows):
        raise ValueError("prompt exceeds context")
    if temperature > 0 and (seeds is None or len(seeds) != len(rows)):
        raise ValueError("sampled generation needs one seed per prompt")
    if stops is not None and len(stops) != len(rows):
        raise ValueError("stops need one token set per prompt")
    stop_sets = [sorted(set(row)) for row in stops] if stops is not None else []
    if any(
        type(token) is not int or not 0 <= token < config.vocab_size
        for row in stop_sets
        for token in row
    ):
        raise ValueError("stop tokens must be vocabulary IDs")
    width = max(len(row) for row in rows)
    pad = [width - len(row) for row in rows]
    budgets = [min(max_new_tokens, config.context - len(row)) for row in rows]
    ids = torch.tensor([[0] * gap + row for gap, row in zip(pad, rows, strict=True)], device=device)
    stop_width = max((len(row) for row in stop_sets), default=0)
    stop_table = (
        torch.tensor([row + [-1] * (stop_width - len(row)) for row in stop_sets], device=device)
        if stop_width
        else None
    )
    tokens, counts, stopped, chosen = _decode(
        model,
        ids,
        torch.tensor(pad, dtype=torch.long, device=device),
        budgets,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seeds=None if temperature == 0 or seeds is None else _seed_tensor(seeds, device),
        stops=stop_table,
        log_probs=log_probs,
        precision=precision,
        prefill_chunk=prefill_chunk,
        stop_check_interval=stop_check_interval,
    )
    token_rows, count_rows, stop_rows = tokens.tolist(), counts.tolist(), stopped.tolist()
    log_prob_rows = None if chosen is None else chosen.tolist()
    return [
        Completion(
            tokens=token_rows[row][: count_rows[row]],
            finish_reason="stop" if stop_rows[row] else "length",
            budget=budgets[row],
            log_probs=None if log_prob_rows is None else log_prob_rows[row][: count_rows[row]],
        )
        for row in range(len(rows))
    ]


def generate_many(
    model: GPT,
    prompts: Sequence[Sequence[int]],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seeds: Sequence[int] | None = None,
    stops: Sequence[Collection[int]] | None = None,
    log_probs: bool = False,
    precision: str = "float32",
    prefill_chunk: int | None = None,
    stop_check_interval: int = 8,
) -> list[Completion]:
    """Generate in length-sorted batches of at most ``batch_size``; results keep input order.

    Prompts are ordered longest first (ties by input index), so each batch pads
    little and the largest cache is allocated first.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not prompts:
        raise ValueError("prompts must be nonempty")
    for name, values in (("seeds", seeds), ("stops", stops)):
        if values is not None and len(values) != len(prompts):
            raise ValueError(f"{name} need one entry per prompt")
    order = sorted(range(len(prompts)), key=lambda index: (-len(prompts[index]), index))
    results: dict[int, Completion] = {}
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        completions = generate_batch(
            model,
            [prompts[index] for index in chunk],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seeds=None if seeds is None else [seeds[index] for index in chunk],
            stops=None if stops is None else [stops[index] for index in chunk],
            log_probs=log_probs,
            precision=precision,
            prefill_chunk=prefill_chunk,
            stop_check_interval=stop_check_interval,
        )
        results.update(zip(chunk, completions, strict=True))
    return [results[index] for index in range(len(prompts))]


@torch.no_grad()
def generate(
    model: GPT,
    prompts: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_id: int | None = None,
    seed: int = 0,
) -> Tensor:
    """Continue equal-length prompts [B,T]; return [B, T + N] with N <= max_new_tokens.

    Row ``r`` samples with seed ``seed + r``, through the same sampler and decoding
    loop as ``generate_batch``. After a row emits ``eos_id`` it is filled with
    ``eos_id``; generation ends once every row has stopped.
    """
    if prompts.ndim != 2 or prompts.shape[1] < 1 or prompts.shape[0] < 1:
        raise ValueError("prompts must be nonempty [batch,time]")
    if max_new_tokens < 0 or prompts.shape[1] + max_new_tokens > model.config.context:
        raise ValueError("generation exceeds context")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("invalid stop token")
    temperature, top_k, top_p = _controls(temperature, top_k, top_p)
    output = prompts.clone()
    if max_new_tokens == 0:
        return output
    batch, device = prompts.shape[0], prompts.device
    tokens, counts, _, _ = _decode(
        model,
        prompts,
        torch.zeros(batch, dtype=torch.long, device=device),
        [max_new_tokens] * batch,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seeds=_seed_tensor([seed + row for row in range(batch)], device),
        stops=None if eos_id is None else torch.full((batch, 1), eos_id, device=device),
        log_probs=False,
        precision="float32",
        prefill_chunk=None,
        stop_check_interval=8,
    )
    if eos_id is not None:
        written = torch.arange(tokens.shape[1], device=device)[None, :] < counts[:, None]
        tokens = torch.where(written, tokens, eos_id)
    return torch.cat((output, tokens), dim=1)


@torch.no_grad()
def score_answers(model: GPT, prompt: list[int], answers: list[list[int]]) -> list[float]:
    """Sum log probabilities over answer tokens, never over the prompt/padding."""
    if not prompt or not answers or any(not answer for answer in answers):
        raise ValueError("prompt and answers must be nonempty")
    length = max(len(prompt) + len(answer) for answer in answers)
    if length > model.config.context:
        raise ValueError("answer exceeds context")
    device = next(model.parameters()).device
    ids = torch.tensor(
        [prompt + answer + [0] * (length - len(prompt) - len(answer)) for answer in answers],
        device=device,
    )
    logits = model(ids).float()
    log_probabilities = logits - logits.logsumexp(dim=-1, keepdim=True)
    scores = []
    for index, answer in enumerate(answers):
        positions = log_probabilities[index, len(prompt) - 1 : len(prompt) + len(answer) - 1]
        selected = positions.gather(-1, torch.tensor(answer, device=device)[:, None])
        scores.append(float(selected.sum()))
    return scores
