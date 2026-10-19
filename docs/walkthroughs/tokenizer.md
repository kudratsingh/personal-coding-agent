# Byte-level BPE

## A small example

`banana banana` begins as UTF-8 bytes. Chunks separate the two words and the space. Count adjacent pairs within chunks; repeated words contribute their frequency. Merge the most frequent pair into a new token, resolving ties by the numeric pair. Repeat until the vocabulary limit or no remaining pairs.

Bytes occupy IDs 0–255. Sixteen special-token IDs follow, with merges beginning at 272. Special spellings in ordinary input remain ordinary bytes unless explicitly enabled, so untrusted text cannot insert conversation delimiters implicitly.

## Data flow

Text becomes a list of byte lists, one per whitespace/non-whitespace chunk. A merge shortens a chunk by replacing non-overlapping matching adjacent pairs. Encoding applies merges in their learned order; decoding concatenates the bytes represented by each ID. Every UTF-8 input therefore round-trips, including code indentation and Unicode.

Training deduplicates chunks with weights. Pair counts and chunk-location sets update only where a selected merge appears. A heap selects the next candidate; stale heap entries are ignored. This avoids recounting every chunk for every merge.

## Verification and limits

`uv run pytest tests/test_tokenizer.py` compares the incremental trainer with a separate slow full-recount reference on randomized corpora, checks input-order determinism, save/load hashes, special-token behavior, and random Unicode round-trips. Invalid generated UTF-8 decodes with replacement characters; exact round-trip is guaranteed for valid text passed to encode.

The production freeze is a separate, single authorized run on reviewed real samples; its evidence is recorded only after measurement. Fixture tests and synthetic throughput cannot establish held-out real-source compression.

`tools/freeze_tokenizer.py` selects equal UTF-8 byte prefixes from stage-1 web and Python training documents, never validation. It preserves earlier document boundaries and shortens only a final prefix to a common valid UTF-8 byte boundary. The smaller available source limits both domains; the declared cap is 100 MB per domain. Training uses the existing from-scratch BPE trainer with 32,768 vocabulary entries. It journals the selection/source hashes before the one training call, then saves the frozen tokenizer identity, file SHA and merge-only SHA before any encoding measurement. An existing output directory blocks retraining. The encoded token count excludes BOS, with a separate count including one BOS per document. Held-out bytes per token use the hash-split validation text only; all retained samples are recounted with four ordered CPU encoding workers. Small real validation probes also check reference/fast IDs and round-trip equality, not a 100 MB reference-equivalence claim. `tests/test_freeze_tokenizer.py` exercises these policies on authored text, including UTF-8 balance, held-out exclusion, a single training call, identity records and refusal to retrain the same output.

The tokenizer owns an immutable tuple of merge pairs, even when a runtime caller supplies lists. A frozen dataclass alone would not protect caller-owned lists: editing one could change the tokenizer hash after a fast encoder had already cached the old ranks. Construction copies both levels, rejects malformed/duplicate/forward-reference pairs and enforces the same 65,536-token ceiling as training and uint16 shards. `tests/test_tokenizer_integrity.py` checks caller-mutation isolation, reference/fast identity, format-1 hash compatibility and the exact vocabulary boundary. Serialization and valid existing tokenizer hashes are unchanged. These structural checks are not a memory sandbox for arbitrary tokenizer files or expanded token bytes.

An additional reference-only check independently trains Hugging Face Tokenizers BPE on the same synthetic byte chunks, comparing learned merge order/bytes and encoded IDs. Run `uv run --locked --extra baseline pytest tests/test_tokenizer_external.py`; dedicated CPU CI installs the locked extra so this check actually runs. No pretrained model/tokenizer or dataset is downloaded. The core implementation does not depend on the library.

The adapter represents each raw byte as one Latin-1 code point and supplies our chunk boundaries explicitly, with no library normalizer/pretokenizer. The reference trainer receives all 256 initial symbols and minimum pair frequency one, as documented in the [reference trainer API](https://huggingface.co/docs/tokenizers/v0.22.2/api/trainers). It reserves no special IDs, so its merge IDs are shifted by sixteen for comparison. Random tie-heavy corpora, Unicode/code, whitespace, and unseen bytes exercise both independent training and encoding. This validates the BPE algorithm under matched preprocessing, not identity with an unrelated pretrained tokenizer.

`FastEncoder` uses learned pair ranks to visit only merges present in a chunk and caches repeated chunks. `encode_many` can distribute documents to spawn-based workers while preserving order. Reference/fast identity and worker ordering are tested; `tools/benchmark_tokenizer.py` reports synthetic throughput and checks identity over the requested byte count. Synthetic repetition benefits caching and must not be represented as real-corpus throughput.

The serial iterator consumes one document at a time. Parallel encoding feeds windows of at most 16 documents per worker to the existing 16-document chunks, exhausting each ordered window before reading the next. Using an unrestricted `Pool.imap` directly would let the producer run ahead and completed results accumulate behind a paused consumer. This window bounds document-count prefetch, not bytes/RSS: callers must still bound individual documents and their own retained outputs. A source error aborts the current window; no crash-recovery or partial-output transaction is promised. Close the returned generator when stopping early so its worker pool is terminated. `tests/test_bulk_prefetch.py` checks producer consumption, boundaries/order, short/empty inputs, strict worker counts and cleanup after early close or producer/worker errors.

## Questions

The completed [real-source freeze](../tokenizer-freeze.md) records one 192,446,084-byte balanced training run, vocabulary 32,768, held-out English/Python compression 2.74195/3.37015 bytes per token, and the immutable identities. It is not a benchmark-trained vocabulary or a full-data/model-training run. Production retraining now requires explicit owner approval.

- Why bytes? Every valid UTF-8 string is representable without unknown tokens.
- Why weighted chunks? Repeated words contribute fully without storing repeated sequences.
- Why forbid merges across chunks? It preserves deliberate whitespace boundaries.
- Why a fixed special-token prefix? Later training stages keep the same token IDs.
- Why hashes? A run can identify the exact merge table, not merely its filename.
