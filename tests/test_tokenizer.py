import json
import random
from collections import Counter
from pathlib import Path

import pytest

from scratchcoder.tokenizer.bpe import CHUNKS, FIRST_MERGE, ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder


def test_single_space_attaches_without_splitting_code_whitespace_runs() -> None:
    text = "the quick brown\n\n    return value\t+  other\n "
    assert CHUNKS.findall(text) == [
        "the",
        " quick",
        " brown",
        "\n\n",
        "    ",
        "return",
        " value",
        "\t",
        "+",
        "  ",
        "other",
        "\n",
        " ",
    ]
    assert "".join(CHUNKS.findall(text)) == text
    trained = ByteBPE.train(["the quick brown fox jumps over the lazy dog"] * 3, 400)
    assert trained.pre_split_version == 2
    assert trained.decode(trained.encode(text)) == text
    assert FastEncoder(trained).encode(text) == trained.encode(text)
    tokens = trained.byte_tokens()
    assert any(tokens[token].startswith(b" quick") for token in trained.encode("the quick"))


def test_split_protocol_is_hashed_and_legacy_encoding_is_preserved(tmp_path: Path) -> None:
    legacy = ByteBPE(((32, 97),))
    current = ByteBPE(legacy.merges, pre_split_version=2)
    assert legacy.encode(" a") == FastEncoder(legacy).encode(" a") == [32, 97]
    assert current.encode(" a") == FastEncoder(current).encode(" a") == [FIRST_MERGE]
    assert legacy.sha256 != current.sha256
    for tokenizer in (legacy, current):
        path = tmp_path / f"v{tokenizer.pre_split_version}.json"
        tokenizer.save(path)
        loaded = ByteBPE.load(path)
        assert loaded == tokenizer and loaded.encode(" a") == tokenizer.encode(" a")


def slow_reference(corpus: list[str], merges: int) -> tuple[tuple[int, int], ...]:
    """Recount the whole corpus each iteration; intentionally slow and independent."""
    sequences = [list(chunk.encode()) for text in corpus for chunk in CHUNKS.findall(text)]
    result: list[tuple[int, int]] = []
    for number in range(merges):
        counts: Counter[tuple[int, int]] = Counter()
        for ids in sequences:
            counts.update(zip(ids, ids[1:], strict=False))
        if not counts:
            break
        pair = min(counts, key=lambda value: (-counts[value], value))
        result.append(pair)
        for index, ids in enumerate(sequences):
            rebuilt: list[int] = []
            offset = 0
            while offset < len(ids):
                if ids[offset : offset + 2] == list(pair):
                    rebuilt.append(FIRST_MERGE + number)
                    offset += 2
                else:
                    rebuilt.append(ids[offset])
                    offset += 1
            sequences[index] = rebuilt
    return tuple(result)


def test_incremental_training_matches_slow_reference() -> None:
    rng = random.Random(4)
    for _ in range(15):
        corpus = ["".join(rng.choice("abcc \n") for _ in range(100)) for _ in range(5)]
        actual = ByteBPE.train(corpus, FIRST_MERGE + 25)
        assert actual.merges == slow_reference(corpus, 25)
        assert actual.sha256 == ByteBPE.train(reversed(corpus), FIRST_MERGE + 25).sha256


def test_unicode_and_code_roundtrip_and_save(tmp_path: Path) -> None:
    corpus = ["def add(a, b):\n    return a + b\n", "English prose and emoji 🐍 café 中文"]
    tokenizer = ByteBPE.train(corpus, 350)
    rng = random.Random(17)
    examples = corpus + ["", "\t  \n\n", "<|bos|>"]
    for _ in range(100):
        examples.append(
            "".join(
                chr(rng.choice([rng.randrange(0xD800), rng.randrange(0xE000, 0x110000)]))
                for _ in range(20)
            )
        )
    for example in examples:
        assert tokenizer.decode(tokenizer.encode(example)) == example
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPE.load(path)
    assert loaded == tokenizer
    assert loaded.sha256 == tokenizer.sha256


def test_special_tokens_require_explicit_opt_in() -> None:
    tokenizer = ByteBPE()
    assert tokenizer.bos_id not in tokenizer.encode("<|bos|>")
    assert tokenizer.encode("<|bos|>", allow_special=True) == [tokenizer.bos_id]
    assert tokenizer.decode([tokenizer.bos_id], skip_special=True) == ""


def test_invalid_merge_and_unknown_token_rejected() -> None:
    with pytest.raises(ValueError):
        ByteBPE(((9999, 1),))
    with pytest.raises(ValueError):
        ByteBPE().decode([100000])


def test_tokenizer_loader_rejects_extra_or_mistyped_merge_values(tmp_path: Path) -> None:
    raw = json.loads(ByteBPE().serialized())
    path = tmp_path / "bad.json"
    for merges in ([[1, 2, 3]], [[True, 2]], [[1]], "not-a-list"):
        raw["merges"] = merges
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ByteBPE.load(path)
