# Data policy

Training sources must have verified licenses and documented provenance before use. An aggregate dataset description is not a substitute for checking its individual components and source terms.

For each source record its identifier, pinned revision, purpose, license and attribution, terms acceptance where required, content hash, and transformation steps. Use synthetic or redistributable fixtures for automated tests.

Keep benchmark prompts, reference solutions, hidden tests, and held-out tasks out of every training stage. Apply duplicate and contamination checks before splitting; make splits deterministic and record their hashes.

Raw corpora, tokenized shards, and credentials remain outside Git. Publish a data card listing the sources actually used when a training milestone is released. No production training dataset is currently included or approved by this starter.
