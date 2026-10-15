# Data policy

Training sources must have verified licenses and documented provenance before use. An aggregate dataset description is not a substitute for checking its individual components and source terms.

For each source record its identifier, pinned revision, purpose, license and attribution, terms acceptance where required, content hash, and transformation steps. Use synthetic or redistributable fixtures for automated tests.

Keep benchmark prompts, reference solutions, hidden tests, and held-out tasks out of every training stage. Apply duplicate and contamination checks before splitting; make splits deterministic and record their hashes.

Raw corpora, tokenized shards, and credentials remain outside Git. Publish a data card listing the sources actually used when a training milestone is released. No production training dataset is currently included or approved by this starter.

The [current data card](data-card.md) records the synthetic-only development state. The manifest downloader fails closed on missing approval, unpinned revisions, disallowed training licenses, absent gated terms acceptance, or mismatched bytes.

Filtering records a primary rejection reason per source for length, code shape, generated markers, exact duplicates, and held-out n-gram matches. The default contamination index uses normalized 13-word sequences from supplied held-out text. This catches planted verbatim spans, not every paraphrase; real corpora still need further review and near-duplicate analysis. Document hash splits are deterministic and keep stage-1, stage-2, and validation web material separate.
