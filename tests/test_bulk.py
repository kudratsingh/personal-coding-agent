import random

from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder, encode_many


def test_fast_and_reference_encoders_match() -> None:
    rng = random.Random(5)
    corpus = ["banana bandana\n  def add(a, b): return a+b\n🐍 café 中文"]
    tokenizer = ByteBPE.train(corpus, 350)
    examples = corpus + ["".join(rng.choices("abc 🐍\t\n", k=100)) for _ in range(50)]
    fast = FastEncoder(tokenizer)
    assert [fast.encode(text) for text in examples] == [tokenizer.encode(text) for text in examples]


def test_process_workers_preserve_document_order() -> None:
    corpus = [f"document {i} with indentation\n    return {i}" for i in range(40)]
    tokenizer = ByteBPE.train(corpus, 320)
    assert list(encode_many(tokenizer, corpus, workers=2)) == list(encode_many(tokenizer, corpus))
