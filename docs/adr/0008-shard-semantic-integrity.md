# ADR 0008: semantic shard checks with legacy compatibility

Status: accepted. Date: 2026-10-16.

## Context

A content hash detects changed bytes but does not establish that token IDs fit the declared vocabulary or that document starts contain BOS. The original writer did not record custom BOS values and accepted a floating vocabulary declaration. The reader ignored declared vocabulary bounds. Existing local version-1 shards and their exact-resume identities must remain usable.

## Decision

Keep the uint16 little-endian representation and version-1 required fields. Newly written headers add optional `bos_id`; old readers already ignore this extra field. New readers validate known header fields/types, vocabulary and offsets before use, and validate each decoded document's IDs and declared/explicitly supplied BOS on demand. Training and held-out validation supply the actual frozen tokenizer's vocabulary/BOS expectations.

Legacy headers without BOS retain their original identity and are never rewritten. A reader without explicit caller context leaves legacy BOS unknown, rather than guessing a custom value. A caller with the actual tokenizer can supply that expectation. New headers naturally have different metadata identities; resume checks must not conflate them with old headers.

## Consequences

Valid token sequences and byte layout do not change. No second whole-corpus token scan is introduced: content hashing still occurs at open, with semantic ID checks when a document is read. Invalid declarations and out-of-range IDs fail before reaching an embedding lookup. This is integrity validation, not authentication or protection against concurrent malicious file modification. Existing full-build, tokenizer-freeze and hardware approval gates remain separate.

## Verification

Authored regressions reproduced accepted float declarations, hash-consistent out-of-vocabulary IDs, and an iterator-raised `FileExistsError` leaving a partial binary before their fixes. Cleanup now applies only after this writer exclusively opens its own new file; existing binaries remain untouched on open failure. Header/BOS/range/legacy tests, unchanged loader resume tests, train/validation preflight tests, and the synthetic two-stage smoke verify the change. See the [shard walkthrough](../walkthroughs/shards.md).
