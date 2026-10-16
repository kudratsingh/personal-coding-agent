# Token shards and integrity

A shard stores documents as little-endian unsigned 16-bit token IDs. For two documents `[1, 2]` and `[3]` with BOS 256, the file holds `[256, 1, 2, 256, 3]`; offsets are `[(0, 3), (3, 2)]`, measured in tokens. A hand-rendered binary reference test verifies byte order and exact token sequences independently of the writer's packing helper.

The JSON header records version/dtype, vocabulary, tokenizer identity, file SHA-256 and contiguous document offsets. New headers additionally record `bos_id`. The writer requires integer IDs and a vocabulary that fits uint16, inserts BOS once per document, exclusively creates the binary file, and publishes its header after flushing the content. It does not overwrite existing shards.

On open, the reader rejects malformed/unknown header fields, non-integer declarations, invalid vocabulary/BOS, broken offsets, wrong tokenizer identity, and byte-hash mismatches. Optional `vocab_size` and `bos_id` arguments let a caller verify the header against its frozen tokenizer; training and held-out validation supply these. This prevents a matching tokenizer-hash label from concealing an incompatible declared vocabulary.

Documents are read on demand. Before returning their IDs, the reader verifies the vocabulary range and, when known, the initial BOS. This avoids a second whole-corpus token scan while preventing invalid IDs from reaching model embeddings. A hash-consistent file can still be semantically invalid, so hashes alone do not replace these checks.

Legacy version-1 headers without `bos_id` remain supported without modification. Their original metadata hash and exact-resume identity are unchanged. Without explicit tokenizer context, the reader reports BOS as unknown; it cannot reconstruct an unrecorded custom BOS. With that context, it checks every read document against the supplied value. This compatibility policy is recorded in [ADR 0008](../adr/0008-shard-semantic-integrity.md).

`tests/test_shard_integrity.py` covers malformed fields, vocabulary/BOS mismatches, out-of-range IDs despite a valid content hash, custom BOS and legacy identity. `tests/test_shards.py` checks unchanged bytes, loader source shares, rank partitioning and exact resume. Train/validation command tests require semantic mismatch rejection before producing run/report artifacts. The synthetic smoke exercises both stages without ingesting external data or executing generated code.

SHA-256 provides content identity, not an authenticated publisher signature. Shards must remain immutable while a run uses them; these checks do not establish security against concurrent file modification. Real-corpus throughput, production tokenizer freeze and full-build approval remain separate milestones.

## Loader resume state

`MixedLoader` saves four fields: a configuration/source signature, its Python RNG state, per-source document positions and leftover token buffers. A position counts consumed documents across epochs, so it may exceed the source size. A buffer may exceed the sequence length because the loader appends whole documents before consuming a sequence. Neither is truncated on resume.

The loader copies the caller's source-name map and sampling-share map before initialization. Replacing, adding or removing entries in those original maps cannot silently change batches while leaving the saved signature unchanged. Corpus objects are not deep-copied: their contents and shard files must still remain immutable during use. This does not prevent deliberate modification of loader internals or concurrent source changes. `tests/test_loader_ownership.py` compares exact batches and native/JSON continuation on both ranks after caller-map mutations, and preserves a pre-fix signature/batch reference.

Loading first checks the exact field/source sets, matching signature, nonnegative integer positions and list buffers containing nonnegative integer token IDs. Booleans and floating-point tokens are rejected instead of being silently converted later. Current native RNG tuples and their JSON-list representation are supported; word ranges/types and finite cache values are checked, and Python validates the RNG shape/index in a temporary generator. Only after all validation and copying succeeds does the live loader change. A rejected load leaves its next batches unchanged. The saved format and source/configuration signature are unchanged.

`tests/test_loader_state.py` compares valid native/JSON continuation on both rank partitions and exercises malformed fields, rollback, long tails, multi-epoch positions and caller-copy independence. Existing trainer tests and smoke runs cover normal checkpoint integration. This transaction covers **only the loader**: it does not make an entire trainer restore transactional, authenticate checkpoints or make unsafe checkpoint deserialization safe. Tokenizer/vocabulary compatibility still comes from shard and caller checks, not this structural state validation.
