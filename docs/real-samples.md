# Real pretraining sample review

`tools/real_samples.py` reviews local, pinned Parquet containers without executing their contents. Raw data, access tokens, source notices and run artifacts stay ignored. Corpus access does not imply blanket downstream rights or authorization for a full build.

## Selection and provenance

The named sources are FineWeb-Edu `sample-10BT` and StarCoderData Python. An ordered whole-document prefix reaches at least **500,000,000 UTF-8 web-text bytes** and **1,000,000,000 UTF-8 Python bytes**, before filters. The boundary document is not clipped; overshoot is reported. These prefixes are not representative random samples. Upstream Parquet containers can require more transfer bytes than the selected text: report both, never describe selected text bytes as network transfer.

Before download, the owner-only manifest records source/revision/file, upstream LFS size/SHA-256, license/attribution, terms date and review purpose. After transfer, verify the complete container size/hash before parsing. The offline tool checks these receipts again and uses PyArrow only for file-format parsing, not training logic. No upstream loader script is run.

FineWeb-Edu carries ODC-By 1.0 plus Common Crawl terms; credit Hugging Face and Lozhkov et al. (2024). StarCoderData/The Stack carries its accepted source terms and the original code licenses; credit BigCode and the original repository authors. The Python schema supplies repository/path provenance but no per-file license column. Therefore only original-file declarations recognized as an allowlisted SPDX identifier, complete MIT notice, or Apache-2.0 notice are retained; missing, compound, conflicting and unsupported declarations are excluded, not inferred from the dataset label. Full source text including notices and repository/path provenance is preserved locally. This narrow declaration check is not a warranty of rights and does not cover every permissive license spelling.

## Filtering and contamination walkthrough

License admission comes first. The existing `filter_documents` then applies length, Python line/alphanumeric/generated-marker, exact-duplicate and contamination checks, assigning one primary rejection reason per document. Reports include zero-valued reasons and separate license drops; do not add overlapping observations as disjoint failures. Content hashes determine disjoint training/validation allocations, with separate stage-1/stage-2/validation web splits.

The contamination index includes pinned HumanEval+ and MBPP+ prompts, solutions and hidden scripts, plus ARC-Easy test questions. Source bytes are parsed only for exclusion, never added to training or tokenizer inputs. The normalized 13-word index also includes shorter complete fields. A final verified task-pool holdout does not yet exist; its coverage is explicitly **0**, not silently claimed complete. Repeat admission against that holdout before any full-corpus training. Verbatim-span filtering does not detect every paraphrase.

Step-3 token counts use the byte alphabet before BPE merges: one UTF-8 byte per token, no BOS. They are labeled as such, not presented as production-BPE counts. After the separately authorized tokenizer freeze, recount the retained samples with that exact merge-table identity and report held-out compression. No tokenizer is trained or silently changed by the review tool.

## Verification

`uv run --locked pytest tests/test_real_samples.py tests/test_filters.py` uses authored fixtures for UTF-8 whole-document selection, insufficient input, file-license decisions, primary filter accounting, provenance retention, contamination and deterministic splits. Real run values belong in the evidence section after collection; fixture tests are not sample measurements.

## Evidence

The first full-size local review is in progress. No filtered-size, production token-count or tokenizer-freeze claim is made before measured reports exist. The full-build estimate remains a separate plan-only owner review; GPU pilot work is not authorized.
