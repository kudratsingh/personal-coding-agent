"""Policies for the loop: the checkpoint (generation on the host, never execution) and the
reference-fix oracle that checks the harness end to end."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

from scratchcoder.agent import workspace
from scratchcoder.agent.cases import Case
from scratchcoder.agent.loop import Decoding, Proposal
from scratchcoder.agent.prompt import Prompt
from scratchcoder.data.chat import CODE_STOP_IDS, TextDecoder, extract_program
from scratchcoder.eval.harness import attempt_seed

AGENT_SEED = 2**41
"""Base seed of sampled agent turns; a turn's seed is
``attempt_seed(AGENT_SEED, f"{case}/turn-{turn}", sample index)``."""

Generate = Callable[[Sequence[int], int, float, float, int | None], tuple[list[int], str]]
"""(prompt ids, max new tokens, temperature, top-p, seed) -> (new tokens, finish reason)."""


def turn_seed(case_id: str, turn: int, index: int) -> int:
    return attempt_seed(AGENT_SEED, f"{case_id}/turn-{turn}", index)


def model_generate(model: object) -> Generate:
    """Batch-1 generation with ``generate_batch`` and the chat-v1 code stop tokens."""
    import torch

    from scratchcoder.engine.generation import generate_batch
    from scratchcoder.model.gpt import GPT

    assert isinstance(model, GPT)

    def run(
        ids: Sequence[int], cap: int, temperature: float, top_p: float, seed: int | None
    ) -> tuple[list[int], str]:
        with torch.inference_mode():
            completion = generate_batch(
                model,
                [list(ids)],
                max_new_tokens=cap,
                temperature=temperature,
                top_p=top_p,
                seeds=None if temperature == 0 or seed is None else [seed],
                stops=[CODE_STOP_IDS],
            )[0]
        return list(completion.tokens), completion.finish_reason

    return run


class ModelPolicy:
    """The checkpoint answers each prompt with one generation. Generations are cached by
    (prompt, cap, decoding, seed): the single condition's turn equals the repair condition's
    first turn, so it is generated once and reused (``cached`` in the trace)."""

    def __init__(
        self,
        generate: Generate,
        decoder: TextDecoder,
        *,
        name: str = "model",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.generate = generate
        self.decoder = decoder
        self.clock = clock
        self.cache: dict[tuple[object, ...], Proposal] = {}

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: Prompt,
        max_new_tokens: int,
        decoding: Decoding,
        files: Mapping[str, str],
    ) -> Proposal:
        seed = None if decoding.temperature == 0 else turn_seed(case.id, turn, decoding.index)
        key = (prompt.ids, max_new_tokens, decoding.temperature, decoding.top_p, seed)
        if key in self.cache:
            hit = self.cache[key]
            return Proposal(
                hit.code, hit.text, hit.generated_tokens, hit.finish_reason, 0.0, cached=True
            )
        started = self.clock()
        tokens, finish = self.generate(
            prompt.ids, max_new_tokens, decoding.temperature, decoding.top_p, seed
        )
        seconds = self.clock() - started
        proposal = Proposal(
            extract_program(tokens, self.decoder),
            self.decoder.decode(tokens),
            len(tokens),
            finish,
            seconds,
        )
        self.cache[key] = proposal
        return proposal


class OraclePolicy:
    """Writes the reference version of the first target function that still differs from
    it. A harness check: it must solve every validated case through the full loop."""

    name = "oracle"

    def propose(
        self,
        case: Case,
        turn: int,
        prompt: Prompt,
        max_new_tokens: int,
        decoding: Decoding,
        files: Mapping[str, str],
    ) -> Proposal:
        for key in case.targets:
            path, name = key.split("::")
            current = workspace.function_source(
                files[path], workspace.functions(files[path], path)[name]
            )
            reference = case.reference_function(key)
            if current != reference:
                return Proposal(reference, reference, 0, "oracle")
        return Proposal(None, "", 0, "oracle")
