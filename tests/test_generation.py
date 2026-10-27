from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from scratchcoder.common.config import ModelConfig
from scratchcoder.engine import generation
from scratchcoder.engine.generation import (
    Completion,
    generate,
    generate_batch,
    generate_many,
    sample,
    sampling_log_probs,
    score_answers,
)
from scratchcoder.model.gpt import GPT, KVCache

CONFIG = ModelConfig(vocab_size=272, width=32, layers=2, heads=4, context=32)


def model_fixture(config: ModelConfig = CONFIG) -> GPT:
    torch.manual_seed(2)
    return GPT(config)


def prompt_fixture(lengths: list[int], *, seed: int = 3) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 272, (length,), generator=generator).tolist() for length in lengths]


@torch.no_grad()
def legacy_generate(
    model: GPT, prompts: Tensor, *, max_new_tokens: int, eos_id: int | None = None
) -> Tensor:
    """The pre-batching greedy loop (growing cache, full prefill logits), kept as the reference."""
    output = prompts.clone()
    logits, cache = model.forward_cached(prompts)
    finished = torch.zeros(prompts.shape[0], dtype=torch.bool)
    for step in range(max_new_tokens):
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        if eos_id is not None:
            token = torch.where(finished[:, None], eos_id, token)
            finished |= token[:, 0] == eos_id
        output = torch.cat((output, token), dim=1)
        if bool(finished.all()) or step + 1 == max_new_tokens:
            break
        logits, cache = model.forward_cached(token, cache)
    return output


def test_cached_logits_equal_full_logits() -> None:
    model = model_fixture()
    ids = torch.randint(0, 272, (2, 12))
    reference = model(ids)
    first, cache = model.forward_cached(ids[:, :5])
    pieces = [first]
    for index in range(5, 12):
        logits, cache = model.forward_cached(ids[:, index : index + 1], cache)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_preallocated_cache_prefill_and_decode_equal_full_forward(qk_norm: bool) -> None:
    model = model_fixture(replace(CONFIG, qk_norm=qk_norm))
    ids = torch.randint(0, 272, (2, 12))
    with torch.no_grad():
        reference = model(ids)
        cache = KVCache(model.config, torch.zeros(2, dtype=torch.long), width=5, capacity=12)
        pieces = [model.decode(ids[:, :5], cache)]
        for index in range(5, 12):
            pieces.append(model.decode(ids[:, index : index + 1], cache))
    assert cache.length == 12 and pieces[0].dtype == torch.float32
    torch.testing.assert_close(torch.stack(pieces, dim=1), reference[:, 4:], atol=1e-5, rtol=1e-4)
    # Keys and values are written into one buffer per layer, never regrown.
    keys = cache.keys[0]
    assert keys is not None and keys.shape == (2, 4, 12, 8)
    with pytest.raises(ValueError, match="capacity"), torch.no_grad():
        model.decode(ids[:, :1], cache)


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_left_padded_rows_match_their_unpadded_logits(chunk: int) -> None:
    model = model_fixture()
    prompts = prompt_fixture([9, 1, 4, 9, 6])
    width = 9
    pad = torch.tensor([width - len(row) for row in prompts])
    ids = torch.tensor([[0] * (width - len(row)) + row for row in prompts])
    follow = torch.randint(0, 272, (5, 4))
    with torch.no_grad():
        cache = KVCache(model.config, pad, width=width, capacity=width + 4)
        for start in range(0, width, chunk):
            logits = model.decode(ids[:, start : start + chunk], cache)
        steps = [logits]
        for index in range(4):
            steps.append(model.decode(follow[:, index : index + 1], cache))
        for row, prompt in enumerate(prompts):
            # Rotary uses the row's own positions 0..len-1, never its padded slots.
            full = model(torch.tensor([prompt + follow[row].tolist()]))[0]
            expected = full[len(prompt) - 1 : len(prompt) + 4]
            actual = torch.stack([step[row] for step in steps])
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_padding_tokens_cannot_change_real_logits() -> None:
    model = model_fixture()
    with torch.no_grad():
        outputs = []
        for filler in (0, 271):
            ids = torch.tensor([[filler, filler, filler, 5, 6, 7]])
            cache = KVCache(model.config, torch.tensor([3]), width=6, capacity=6)
            outputs.append(model.decode(ids, cache))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def test_greedy_batches_equal_the_reference_single_prompt_path() -> None:
    model = model_fixture()
    lengths = [1, 7, 3, 12, 30, 5, 31, 2]
    prompts = prompt_fixture(lengths)
    completions = generate_batch(model, prompts, max_new_tokens=6)
    for prompt, completion in zip(prompts, completions, strict=True):
        budget = min(6, 32 - len(prompt))
        expected = legacy_generate(model, torch.tensor([prompt]), max_new_tokens=budget)
        assert completion.tokens == expected[0, len(prompt) :].tolist()
        assert completion.budget == budget == len(completion.tokens)
        assert completion.finish_reason == "length" and completion.log_probs is None
    # The legacy entry point now runs through the same loop and still matches.
    rows = torch.tensor(prompt_fixture([4, 4, 4], seed=9))
    assert torch.equal(
        generate(model, rows, max_new_tokens=9), legacy_generate(model, rows, max_new_tokens=9)
    )


def test_legacy_stop_token_semantics_are_unchanged() -> None:
    model = model_fixture()
    rows = torch.tensor(prompt_fixture([5, 5, 5], seed=4))
    unstopped = legacy_generate(model, rows, max_new_tokens=10)
    eos_id = int(unstopped[1, 5 + 2])  # Row 1's third token stops it.
    expected = legacy_generate(model, rows, max_new_tokens=10, eos_id=eos_id)
    assert torch.equal(generate(model, rows, max_new_tokens=10, eos_id=eos_id), expected)
    assert expected.shape[1] <= 15


@pytest.mark.parametrize("settings", [{"temperature": 1.0}, {"temperature": 0.7, "top_p": 0.9}])
def test_seeded_sampling_per_row_is_independent_of_batching(settings: dict[str, Any]) -> None:
    model = model_fixture()
    prompts = prompt_fixture([3, 11, 1, 6, 29, 8, 2])
    seeds = [101 + index for index in range(len(prompts))]
    batched = generate_batch(model, prompts, max_new_tokens=8, seeds=seeds, **settings)
    for prompt, seed, completion in zip(prompts, seeds, batched, strict=True):
        single = generate(
            model, torch.tensor([prompt]), max_new_tokens=completion.budget, seed=seed, **settings
        )
        assert completion.tokens == single[0, len(prompt) :].tolist()
    for batch_size in (1, 2, 7):
        assert (
            generate_many(
                model, prompts, batch_size=batch_size, max_new_tokens=8, seeds=seeds, **settings
            )
            == batched
        )
    other = generate_batch(
        model, prompts, max_new_tokens=8, seeds=[seed + 50 for seed in seeds], **settings
    )
    assert other != batched


def test_budgets_are_per_row_and_unchanged_by_sorted_batching() -> None:
    model = model_fixture()
    lengths = [32, 31, 20, 2, 29, 1]
    prompts = prompt_fixture(lengths)
    alone = [generate_batch(model, [prompt], max_new_tokens=5)[0] for prompt in prompts]
    assert [completion.budget for completion in alone] == [0, 1, 5, 5, 3, 5]
    assert [len(completion.tokens) for completion in alone] == [0, 1, 5, 5, 3, 5]
    for batch_size in (1, 2, 4, 6):
        assert generate_many(model, prompts, batch_size=batch_size, max_new_tokens=5) == alone


def test_many_sorts_longest_first_and_restores_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_fixture()
    prompts = prompt_fixture([2, 9, 4, 9, 1])
    seen: list[list[int]] = []
    original = generation.generate_batch

    def recording(model: GPT, rows: list[list[int]], **options: Any) -> list[Completion]:
        seen.append([len(row) for row in rows])
        return original(model, rows, **options)

    monkeypatch.setattr(generation, "generate_batch", recording)
    results = generate_many(model, prompts, batch_size=2, max_new_tokens=3)
    assert seen == [[9, 9], [4, 2], [1]]
    assert results == [generate_batch(model, [prompt], max_new_tokens=3)[0] for prompt in prompts]


def test_stop_sets_end_each_row_and_all_stopped_rows_exit_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_fixture()
    prompts = prompt_fixture([4, 8, 2])
    free = generate_batch(model, prompts, max_new_tokens=20)
    stops = [{free[0].tokens[2]}, {free[1].tokens[0], 271}, set()]
    stopped = generate_batch(model, prompts, max_new_tokens=20, stops=stops)
    for completion, unstopped, row_stops in zip(stopped, free, stops, strict=True):
        cut = next(
            (index + 1 for index, token in enumerate(unstopped.tokens) if token in row_stops),
            len(unstopped.tokens),
        )
        assert completion.tokens == unstopped.tokens[:cut]
        assert completion.finish_reason == ("stop" if cut < len(unstopped.tokens) else "length")
    assert [completion.finish_reason for completion in stopped[:2]] == ["stop", "stop"]
    calls: list[int] = []
    original = model.decode

    def counting(ids: Tensor, cache: KVCache) -> Tensor:
        calls.append(ids.shape[1])
        return original(ids, cache)

    monkeypatch.setattr(model, "decode", counting)
    every_stop = [{token} for token in (row.tokens[1] for row in free)]
    early = generate_batch(
        model, prompts, max_new_tokens=20, stops=every_stop, stop_check_interval=4
    )
    assert all(completion.finish_reason == "stop" for completion in early)
    assert all(len(completion.tokens) <= 2 for completion in early)
    # One prefill, then decoding stops at the first check after every row stopped.
    assert calls == [8, 1, 1, 1]


def test_log_probs_follow_the_distribution_actually_sampled() -> None:
    model = model_fixture()
    prompts = prompt_fixture([3, 6])
    settings: dict[str, Any] = {"temperature": 0.8, "top_k": 7, "top_p": 0.9}
    completions = generate_batch(
        model, prompts, max_new_tokens=5, seeds=[4, 5], log_probs=True, **settings
    )
    for prompt, completion in zip(prompts, completions, strict=True):
        assert completion.log_probs is not None and len(completion.log_probs) == 5
        full = model(torch.tensor([prompt + completion.tokens]))[0].detach()
        for index, token in enumerate(completion.tokens):
            expected = sampling_log_probs(full[len(prompt) - 1 + index], **settings)[token]
            assert completion.log_probs[index] == pytest.approx(float(expected), abs=1e-4)
            assert completion.log_probs[index] > -float("inf")
    greedy = generate_batch(model, prompts, max_new_tokens=3, log_probs=True)
    assert all(completion.log_probs == [0.0, 0.0, 0.0] for completion in greedy)


def test_bfloat16_autocast_runs_and_keeps_float32_logits() -> None:
    model = model_fixture()
    prompts = prompt_fixture([5, 2])
    completions = generate_batch(
        model, prompts, max_new_tokens=4, precision="bfloat16", seeds=[1, 2], temperature=1.0
    )
    assert [len(completion.tokens) for completion in completions] == [4, 4]
    pad = torch.tensor([0, 3])
    ids = torch.tensor([prompts[0], [0, 0, 0, *prompts[1]]])
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.decode(ids, KVCache(model.config, pad, width=5, capacity=5))
    assert logits.dtype == torch.float32
    with torch.no_grad():
        reference = model.decode(ids, KVCache(model.config, pad, width=5, capacity=5))
    torch.testing.assert_close(logits, reference, atol=0.05, rtol=0.05)


def test_decoding_loop_makes_no_per_step_host_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    model = model_fixture()
    prompts = prompt_fixture([3, 9, 5])
    reads: list[str] = []

    def counted(name: str) -> Callable[..., Any]:
        original = getattr(Tensor, name)

        def wrapper(self: Tensor, *args: Any) -> Any:
            reads.append(name)
            return original(self, *args)

        return wrapper

    for name in ("__bool__", "item", "tolist", "__int__", "__float__", "__index__"):
        monkeypatch.setattr(Tensor, name, counted(name))
    for steps in (4, 20):
        reads.clear()
        generate_batch(
            model,
            prompts,
            max_new_tokens=steps,
            temperature=1.0,
            top_p=0.9,
            seeds=[1, 2, 3],
            stops=[{271}, {270}, set()],
            stop_check_interval=100,
        )
        # Only the final finiteness check, the trim and the conversion to lists.
        assert sorted(reads) == ["__bool__", "__int__", "tolist", "tolist", "tolist"]


def test_no_vocabulary_sort_or_cache_growth_without_nucleus_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_fixture()
    prompts = prompt_fixture([3, 9])

    def forbidden(*args: Any, **options: Any) -> Any:
        raise AssertionError("unexpected call")

    for name in ("sort", "argsort", "topk"):
        monkeypatch.setattr(Tensor, name, forbidden)
    monkeypatch.setattr(torch, "cat", forbidden)
    generate_batch(model, prompts, max_new_tokens=6, temperature=1.0, seeds=[1, 2])
    with pytest.raises(AssertionError, match="unexpected"):
        generate_batch(model, prompts, max_new_tokens=6, temperature=1.0, top_p=0.5, seeds=[1, 2])


def test_nonfinite_logits_fail_after_the_loop() -> None:
    model = model_fixture()
    with torch.no_grad():
        model.norm.weight[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        generate_batch(model, prompt_fixture([3]), max_new_tokens=3)
    with pytest.raises(ValueError, match="finite"):
        generate(model, torch.tensor([[1, 2]]), max_new_tokens=2)


@pytest.mark.parametrize(
    "options",
    [
        {"prompts": []},
        {"prompts": [[]]},
        {"prompts": [[272]]},
        {"prompts": [[True]]},
        {"prompts": [[1] * 33]},
        {"max_new_tokens": -1},
        {"temperature": 1.0},
        {"temperature": 1.0, "seeds": [-1]},
        {"temperature": 1.0, "seeds": [2**63]},
        {"temperature": 1.0, "seeds": [1, 2]},
        {"stops": [{1}, {2}]},
        {"stops": [{272}]},
        {"precision": "float16"},
        {"prefill_chunk": 0},
        {"stop_check_interval": 0},
        {"log_probs": 1},
        {"top_k": 0},
    ],
)
def test_batched_generation_rejects_invalid_requests(options: dict[str, Any]) -> None:
    request: dict[str, Any] = {"prompts": [[1, 2]], "max_new_tokens": 2, **options}
    with pytest.raises(ValueError):
        generate_batch(model_fixture(), request.pop("prompts"), **request)


def test_zero_budget_returns_without_running_the_model() -> None:
    model = model_fixture()
    assert generate_batch(model, [[1, 2]], max_new_tokens=0) == [Completion([], "length", 0)]
    assert torch.equal(
        generate(model, torch.tensor([[1, 2]]), max_new_tokens=0), torch.tensor([[1, 2]])
    )


def test_answer_scoring_matches_slow_full_prefix_reference() -> None:
    model = model_fixture()
    prompt, answers = [1, 2, 3], [[4, 5], [6, 7, 8]]
    actual = score_answers(model, prompt, answers)
    expected = []
    for answer in answers:
        total = 0.0
        for index, token in enumerate(answer):
            logits = model(torch.tensor([prompt + answer[:index]]))[0, -1].float()
            total += float((logits[token] - logits.logsumexp(-1)).detach())
        expected.append(total)
    assert actual == pytest.approx(expected, abs=1e-5)


def test_sampling_bounds_and_reproducibility() -> None:
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    assert int(sample(logits, top_k=1)) == 1
    assert int(sample(logits, top_p=0.01)) == 1
    first = sample(logits, seed=5, step=3)
    assert torch.equal(first, sample(logits, seed=5, step=3))
    draws = {int(sample(logits, seed=seed)) for seed in range(40)}
    assert draws == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        sample(logits, temperature=-1)
