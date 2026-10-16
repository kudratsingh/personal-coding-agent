# Byte-level BPE

## A small example

`banana banana` begins as UTF-8 bytes. Chunks separate the two words and the space. Count adjacent pairs within chunks; repeated words contribute their frequency. Merge the most frequent pair into a new token, resolving ties by the numeric pair. Repeat until the vocabulary limit or no remaining pairs.

Bytes occupy IDs 0–255. Sixteen special-token IDs follow, with merges beginning at 272. Special spellings in ordinary input remain ordinary bytes unless explicitly enabled, so untrusted text cannot insert conversation delimiters implicitly.

## Data flow

Text becomes a list of byte lists, one per whitespace/non-whitespace chunk. A merge shortens a chunk by replacing non-overlapping matching adjacent pairs. Encoding applies merges in their learned order; decoding concatenates the bytes represented by each ID. Every UTF-8 input therefore round-trips, including code indentation and Unicode.

Training deduplicates chunks with weights. Pair counts and chunk-location sets update only where a selected merge appears. A heap selects the next candidate; stale heap entries are ignored. This avoids recounting every chunk for every merge.

## Verification and limits

`uv run pytest tests/test_tokenizer.py` compares the incremental trainer with a separate slow full-recount reference on randomized corpora, checks input-order determinism, save/load hashes, special-token behavior, and random Unicode round-trips. Invalid generated UTF-8 decodes with replacement characters; exact round-trip is guaranteed for valid text passed to encode.

No production tokenizer has been trained or frozen. Large-corpus throughput and held-out compression are separate measurements.

An additional reference-only check independently trains Hugging Face Tokenizers BPE on the same synthetic byte chunks, comparing learned merge order/bytes and encoded IDs. Run `uv run --locked --extra baseline pytest tests/test_tokenizer_external.py`; dedicated CPU CI installs the locked extra so this check actually runs. No pretrained model/tokenizer or dataset is downloaded. The core implementation does not depend on the library.

The adapter represents each raw byte as one Latin-1 code point and supplies our chunk boundaries explicitly, with no library normalizer/pretokenizer. The reference trainer receives all 256 initial symbols and minimum pair frequency one, as documented in the [reference trainer API](https://huggingface.co/docs/tokenizers/v0.22.2/api/trainers). It reserves no special IDs, so its merge IDs are shifted by sixteen for comparison. Random tie-heavy corpora, Unicode/code, whitespace, and unseen bytes exercise both independent training and encoding. This validates the BPE algorithm under matched preprocessing, not identity with an unrelated pretrained tokenizer.

`FastEncoder` uses learned pair ranks to visit only merges present in a chunk and caches repeated chunks. `encode_many` can distribute documents to spawn-based workers while preserving order. Reference/fast identity and worker ordering are tested; `tools/benchmark_tokenizer.py` reports synthetic throughput and checks identity over the requested byte count. Synthetic repetition benefits caching and must not be represented as real-corpus throughput.

## Questions

- Why bytes? Every valid UTF-8 string is representable without unknown tokens.
- Why weighted chunks? Repeated words contribute fully without storing repeated sequences.
- Why forbid merges across chunks? It preserves deliberate whitespace boundaries.
- Why a fixed special-token prefix? Later training stages keep the same token IDs.
- Why hashes? A run can identify the exact merge table, not merely its filename.
