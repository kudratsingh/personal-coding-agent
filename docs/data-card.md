# Data card: development fixtures, real samples and evaluation

The **2026-10-20 replacement [production tokenizer freeze](tokenizer-freeze.md)**
used the identical **96,223,042-byte** non-validation selection per domain at
**65,536 total entries**, including specials. Merge SHA-256 is
`b0297f478dd5172a690752ceac8f997fccbd95e36e5787ca56a2c555083c7b7b`.
Original-review corpus totals are **100,709,712 web / 20,389,757 Python tokens**,
no BOS. Original held-outs measure **4.9966354130 English / 4.3333298596 Python
bytes/token**, versus GPT-2 **4.6675797737 / 2.1991506376**; 50/50 byte-mixture
**4.6414040782 / 2.9896939068**. This one owner-authorized replacement supersedes
the October 19 freeze without overwriting it. Further retraining needs approval;
no full corpus has been admitted or model pretrained on these samples.

Real-source sample review is complete, but no full production training corpus has been admitted. Ordinary tests use synthetic strings and byte fixtures authored for this repository. These establish software behavior, not corpus quality or model performance. The reference workflow downloads HumanEval for evaluation only; benchmark material never enters training or tokenizer input.

`data/manifest.json` intentionally contains no full-build-approved sources. The real sample review keeps its raw-container manifests/terms receipts local and ignored, with public source/hash evidence below. External sources require a pinned immutable revision, a verified license and attribution, SHA-256, exact byte count, declared purpose, and owner terms acceptance where applicable. An unverified or incomplete downloader entry fails before a network request.

The [real sample review](real-samples.md) records FineWeb-Edu `sample-10BT` (Hugging Face; Lozhkov et al., 2024; ODC-By 1.0 plus Common Crawl terms) and StarCoderData Python (BigCode/The Stack plus original repository authors; accepted dataset terms and original file licenses). Original notices and repository/path provenance are retained locally; data is not redistributed here. Kept text is **498,123,158 web bytes** and **96,638,864 Python bytes**. Python admission is limited to explicit allowlisted declarations in each retained original file, not a dataset license tag. The final pool holdout does not exist yet, so its exclusion coverage is explicitly **0**. These samples support the separately authorized tokenizer work, not a full data build. Production BPE counts/freeze are reported separately once measured.

The [OpenCodeInstruct component pilot](opencode-pilot.md) separately hash/schema-verified one 133,476,478-byte shard for evaluation-only inspection of its first 200 ordered rows. Actual isolated execution retained 60 references passing both partitions after 35 unsupported-schema, 13 contamination/duplicate and 92 failing-reference rejections. This is not a representative sample, training admission or full-source build. The guide records explicit governing terms, source credit, artifact identity, observed schema, fixed runtime profile and limitations. No source program has run on the host or entered tokenizer/training data.

The downloader verifies size and hash before atomically replacing the target. Test fixtures verify deterministic bytes, rejected licenses, absent terms acceptance, corruption, truncation, and oversized downloads.

## Header-classified sample revision (2026-10-20)

The current [header-classifier review](real-samples.md) retains **13,833 Python
files / 122,701,374 UTF-8 bytes**, not the earlier 96,638,864-byte selection used
to fit the already-frozen tokenizer. This is a new review output, not retroactive
relabeling of the tokenizer's training corpus. Each retained row carries a
recognized license ID, original repository/path and unchanged notices/text.

| Recognized ID | Retained files |
| --- | ---: |
| Apache-2.0 | 8,776 |
| MIT | 3,635 |
| BSD-2-Clause | 280 |
| BSD-3-Clause | 1,009 |
| ISC | 82 |
| PSF-2.0 | 0 |
| Unlicense | 30 |
| CC0-1.0 | 21 |

Credit BigCode/The Stack, the original repository authors and the preserved
per-file copyright holders. Full local notices/provenance are not redistributed
here. Copyleft, restrictive and unrecognized headers are separately dropped;
an "all rights reserved" line does not override a recognized BSD grant. Header
recognition is not a comprehensive component-rights warranty or full-build approval.

The repository fallback correction separates **185,680 files / 786,817,959
bytes** with no inline license and no repository metadata from **8,737 files /
69,830,333 bytes** with unrecognized inline licensing. The actual five-column
Parquet has no repository-license field. No such files are admitted by inference;
off-allowlist repository-ID counts are empty because metadata is absent. When
verified metadata is supplied, retained no-header files record its allowlisted
ID for attribution. The current sample still retains the same 13,833 files and
the ID counts above. See the [current class table](real-samples.md).

With the replacement 65,536-entry tokenizer, this review contains
**26,328,493 Python tokens / 122,701,374 bytes** and **100,709,712 web tokens /
498,123,158 bytes**, excluding BOS. Its newer Python validation selection is
**52 files / 475,430 bytes / 108,883 tokens**. The original 45-file holdout
remains the compression comparison above; see [P1-6](full-build-plan.md) for
the revised conditional capacity forecast.

## Evaluation-only source actually used

- OpenAI HumanEval: 164 tasks, [revision and MIT attribution](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d). Pinned gzip SHA-256: `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` (44,877 bytes).
- Purpose: isolated external-model comparison and canonical known-answer validation, not training or tokenizer fitting. Prompts, reference solutions and hidden tests remain evaluation material.
- The optional reference model is `bigcode/tiny_starcoder_py`, revision `8547527bef0bc927268c1653cce6948c5c242dd1`, under BigCode OpenRAIL-M v1. Its weights are comparison-only, not project model weights or a project training dependency. Caches, weights and generated programs are not committed or uploaded as workflow artifacts.

See [reference evaluation](reference-evaluation.md) for observed outcomes. Real-source adapters and schema tests do not mean the corresponding corpora have been downloaded, rights-cleared, or admitted to training.

The pinned [HumanEval+/MBPP+ releases](evalplus.md) have additionally been downloaded, hash/size checked, parsed and used for isolated canonical preflight, never training or tokenizer fitting. Their fingerprints, component attributions and observed validation outcomes are recorded in that guide. A failed canonical preflight blocks project-model scoring; ordinary CI uses synthetic loader fixtures. The optional nano checkpoint is trained only on repository-authored synthetic documents, independently of these releases.

The pinned [ARC-Easy release](arc-easy.md) has been downloaded and verified for evaluation-only intake: 2,376 test questions, plus 2,251 train and 570 validation rows used only for schema/answer-key and cross-split ID checks. No ARC text is used for training, tokenizer fitting, prompt examples or checkpoint selection. The guide records the independent original-distribution license declaration, component attribution, hashes, scoring limitations and complete synthetic-only nano result: 403/2,376, including 664 context-limit failures. No useful capability or real-corpus training is claimed.

The original [Google Research MBPP source](mbpp-audit.md) is additionally pinned and parsed for an evaluation-only reference-intake audit, not training. It retains original partition information, excludes actual MBPP+ IDs before execution and preserves primary/challenge assertions through conservative normalization. Complete final holdout coverage and training-pool admission are explicitly not claimed; the guide records current evidence and limitations.
