# Data card: development fixtures and evaluation-only reference

No external training corpus has been ingested. Ordinary tests use synthetic strings and byte fixtures authored for this repository. These establish software behavior, not corpus quality or model performance. A separately dispatched reference workflow downloads HumanEval for evaluation only; it never feeds benchmark material into training.

`data/manifest.json` intentionally contains no approved sources. External sources require a pinned immutable revision, a verified license and attribution, SHA-256, exact byte count, declared purpose, and owner terms acceptance where applicable. An unverified or incomplete entry fails before a network request.

The downloader verifies size and hash before atomically replacing the target. Test fixtures verify deterministic bytes, rejected licenses, absent terms acceptance, corruption, truncation, and oversized downloads.

## Evaluation-only source actually used

- OpenAI HumanEval: 164 tasks, [revision and MIT attribution](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d). Pinned gzip SHA-256: `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` (44,877 bytes).
- Purpose: isolated external-model comparison and canonical known-answer validation, not training or tokenizer fitting. Prompts, reference solutions and hidden tests remain evaluation material.
- The optional reference model is `bigcode/tiny_starcoder_py`, revision `8547527bef0bc927268c1653cce6948c5c242dd1`, under BigCode OpenRAIL-M v1. Its weights are comparison-only, not project model weights or a project training dependency. Caches, weights and generated programs are not committed or uploaded as workflow artifacts.

See [reference evaluation](reference-evaluation.md) for observed outcomes. Real-source adapters and schema tests do not mean the corresponding corpora have been downloaded, rights-cleared, or admitted to training.

The pinned [HumanEval+/MBPP+ releases](evalplus.md) have additionally been read for schema, dependency, and source-term review, not used for training or reported model scoring. Their gzip fingerprints and component attributions are recorded in that guide. Actual isolated canonical validation remains separate work; ordinary CI uses synthetic loader fixtures.
