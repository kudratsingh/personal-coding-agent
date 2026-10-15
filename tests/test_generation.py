import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.engine.generation import generate, sample, score_answers
from scratchcoder.model.gpt import GPT


def model_fixture() -> GPT:
    torch.manual_seed(2)
    return GPT(ModelConfig(vocab_size=272, width=32, layers=2, heads=4, context=32))


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


def test_batched_greedy_generation_equals_individual() -> None:
    model = model_fixture()
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batched = generate(model, prompts, max_new_tokens=5)
    separate = torch.cat([generate(model, row[None], max_new_tokens=5) for row in prompts])
    assert torch.equal(batched, separate)


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
    generator = torch.Generator().manual_seed(5)
    first = sample(logits, generator=generator)
    second = sample(logits, generator=torch.Generator().manual_seed(5))
    assert torch.equal(first, second)
    with pytest.raises(ValueError):
        sample(logits, temperature=-1)
