# Data card: development fixtures

No external training corpus or benchmark has been downloaded or used. Tests use short synthetic strings and in-memory byte fixtures authored for this repository. These tests establish software behavior, not corpus quality or model performance.

`data/manifest.json` intentionally contains no approved sources. External sources require a pinned immutable revision, a verified license and attribution, SHA-256, exact byte count, declared purpose, and owner terms acceptance where applicable. An unverified or incomplete entry fails before a network request.

The downloader verifies size and hash before atomically replacing the target. Test fixtures verify deterministic bytes, rejected licenses, absent terms acceptance, corruption, truncation, and oversized downloads.
