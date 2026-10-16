"""Sampling, cached generation, and batched continuation scoring."""

import torch
from torch import Tensor

from scratchcoder.common.config import finite_number
from scratchcoder.model.gpt import GPT


def sample(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    temperature = finite_number(temperature, "temperature")
    top_p = finite_number(top_p, "top_p")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid temperature/top_p")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if (
        logits.layout != torch.strided
        or logits.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        or logits.ndim not in {1, 2}
        or logits.numel() == 0
    ):
        raise ValueError("sampling logits must be nonempty dense floating [vocab] or [batch,vocab]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("sampling logits must be finite")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = logits.float() / temperature
    if not bool(torch.isfinite(filtered).all()):
        # Keep ordinary seeded draws unchanged. Only extreme scaling takes this path.
        wide = logits.double()
        if temperature >= 1:
            # Divide first so opposite finite extremes do not overflow on subtraction.
            wide = wide / temperature
            filtered = wide - wide.amax(dim=-1, keepdim=True)
        else:
            # Center first so the maximum remains zero even at subnormal temperatures.
            filtered = (wide - wide.amax(dim=-1, keepdim=True)) / temperature
    if top_k is not None:
        values, indices = filtered.topk(min(top_k, filtered.shape[-1]), dim=-1)
        filtered = torch.full_like(filtered, float("-inf")).scatter(-1, indices, values)
    values, indices = filtered.sort(dim=-1, descending=True)
    probabilities = values.softmax(dim=-1)
    removed = probabilities.cumsum(dim=-1) - probabilities >= top_p
    values = values.masked_fill(removed, float("-inf"))
    selected = torch.multinomial(values.softmax(dim=-1), 1, generator=generator)
    return indices.gather(-1, selected)


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
    if prompts.ndim != 2 or prompts.shape[1] < 1 or prompts.shape[0] < 1:
        raise ValueError("prompts must be nonempty [batch,time]")
    if max_new_tokens < 0 or prompts.shape[1] + max_new_tokens > model.config.context:
        raise ValueError("generation exceeds context")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("invalid stop token")
    output = prompts.clone()
    if max_new_tokens == 0:
        return output
    generator = torch.Generator(device=prompts.device).manual_seed(seed)
    logits, cache = model.forward_cached(prompts)
    finished = torch.zeros(prompts.shape[0], dtype=torch.bool, device=prompts.device)
    for step in range(max_new_tokens):
        token = sample(
            logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p, generator=generator
        )
        if eos_id is not None:
            token = torch.where(finished[:, None], eos_id, token)
            finished |= token[:, 0] == eos_id
        output = torch.cat((output, token), dim=1)
        if bool(finished.all()) or step + 1 == max_new_tokens:
            break
        logits, cache = model.forward_cached(token, cache)
    return output


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
