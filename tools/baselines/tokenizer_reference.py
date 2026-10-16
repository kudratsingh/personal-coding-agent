"""Independent library BPE training/encoding check; never used in core tokenization."""

import importlib
import json
from collections.abc import Sequence
from typing import Any

from scratchcoder.tokenizer.bpe import CHUNKS, FIRST_MERGE, ByteBPE


def verify_bpe(corpus: list[str], examples: Sequence[str], *, merges: int) -> dict[str, Any]:
    if not corpus or merges < 1:
        raise ValueError("nonempty corpus and positive merge count required")
    library = importlib.import_module("tokenizers")
    models = importlib.import_module("tokenizers.models")
    trainers = importlib.import_module("tokenizers.trainers")
    own = ByteBPE.train(corpus, FIRST_MERGE + merges)
    reference = library.Tokenizer(models.BPE())
    # Each UTF-8 byte becomes one Latin-1 code point; no library normalization or
    # pretokenization can change our explicitly supplied chunk boundaries.
    reference.train_from_iterator(
        (
            chunk.encode("utf-8").decode("latin1")
            for text in corpus
            for chunk in CHUNKS.findall(text)
        ),
        trainer=trainers.BpeTrainer(
            vocab_size=256 + len(own.merges),
            min_frequency=1,
            initial_alphabet=[chr(value) for value in range(256)],
            show_progress=False,
        ),
    )
    byte_tokens = own.byte_tokens()
    expected = [
        [byte_tokens[left].decode("latin1"), byte_tokens[right].decode("latin1")]
        for left, right in own.merges
    ]
    actual = json.loads(reference.to_str())["model"]["merges"]
    if actual != expected:
        raise AssertionError("independently trained BPE merge tables differ")
    for text in examples:
        encoded: list[int] = []
        for chunk in CHUNKS.findall(text):
            ids = reference.encode(
                chunk.encode("utf-8").decode("latin1"), add_special_tokens=False
            ).ids
            # The library reserves no specials: its merge 256 is our merge 272.
            encoded.extend(token if token < 256 else token + FIRST_MERGE - 256 for token in ids)
        if encoded != own.encode(text):
            raise AssertionError("library and handwritten BPE token IDs differ")
    return {
        "reference": "huggingface/tokenizers",
        "version": library.__version__,
        "merge_count": len(own.merges),
        "examples": len(examples),
        "tokenizer_hash": own.sha256,
    }
