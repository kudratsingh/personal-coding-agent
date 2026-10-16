# Data policy

Training sources must have verified licenses and documented provenance before use. An aggregate dataset description is not a substitute for checking its individual components and source terms.

For each source record its identifier, pinned revision, purpose, license and attribution, terms acceptance where required, content hash, and transformation steps. Use synthetic or redistributable fixtures for automated tests.

Keep benchmark prompts, reference solutions, hidden tests, and held-out tasks out of every training stage. Apply duplicate and contamination checks before splitting; make splits deterministic and record their hashes.

Raw corpora, tokenized shards, and credentials remain outside Git. Publish a data card listing the sources actually used when a training milestone is released. No production training dataset is currently included or approved by this starter.

The [current data card](data-card.md) records the synthetic-only development state. The manifest downloader fails closed on missing approval, unpinned revisions, disallowed training licenses, absent gated terms acceptance, or mismatched bytes.

Filtering records a primary rejection reason per source for length, code shape, generated markers, exact duplicates, and held-out n-gram matches. The default contamination index uses normalized 13-word sequences from supplied held-out text. This catches planted verbatim spans, not every paraphrase; real corpora still need further review and near-duplicate analysis. Document hash splits are deterministic and keep stage-1, stage-2, and validation web material separate.

Token shards store little-endian uint16 IDs with an explicit BOS at every document start, document offsets, a tokenizer hash, and content integrity hash. The loader reads documents on demand, partitions document indices across ranks, mixes fixed-length sequences by source share, and serializes RNG/buffer/position state for exact resume. Supply separate, decontaminated source/split shard sets; the loader cannot infer provenance from token IDs.

Normalized task JSONL keeps prompt/public tests separate from hidden tests and optional reference solutions. Source adapters require explicit field mappings; real-source schemas are verified when ingestion is approved, not guessed from dataset names. Pool admission rejects benchmark IDs, prompt near-duplicates, contaminated spans, and references that fail either test partition. Missing isolation aborts admission. Assign held-out sizes after deduplication and preserve immutable upstream holdouts.

Conservative reviewed-schema adapters cover OpenCodeInstruct assertion lists, AceCode recorded completions, full MBPP rows, and evaluation-only HumanEval rows. They require a validated provenance entry, preserve source revision, canonicalize MBPP benchmark IDs, and reject unsupported test statements instead of executing source setup on the host. Passing upstream attempts still require local isolated verification. Other source variants and full-corpus admission remain separate ingestion work.
