import random

import pytest

from tools.baselines.tokenizer_reference import verify_bpe

pytest.importorskip("tokenizers", reason="optional baseline reference; dedicated CI installs it")


def test_independent_library_training_matches_tie_heavy_random_corpora() -> None:
    rng = random.Random(4)
    for _ in range(15):
        corpus = ["".join(rng.choice("abcc \n") for _ in range(100)) for _ in range(5)]
        report = verify_bpe(corpus, corpus + ["abacbc\n  cab"], merges=25)
        assert report["examples"] == 6 and report["merge_count"] <= 25


def test_independent_library_unicode_code_whitespace_and_unseen_bytes() -> None:
    corpus = [
        "def add(a, b):\n    return a + b\n" * 4,
        "English prose and emoji 🐍 café 中文\n" * 4,
        "aab aaa ab aba aaa\t\t\n" * 3,
    ]
    report = verify_bpe(
        corpus, corpus + ["", "\t  \n\n", "<|bos|>", "\x00\x85🦀\U0010ffff"], merges=60
    )
    assert report["examples"] == 7 and report["reference"] == "huggingface/tokenizers"
