# Real pretraining sample review

`tools/real_samples.py` reviews local, pinned Parquet containers without executing their contents. Raw data, access tokens, source notices and run artifacts stay ignored. Corpus access does not imply blanket downstream rights or authorization for a full build.

## Selection and provenance

The named sources are FineWeb-Edu `sample-10BT` and StarCoderData Python. An ordered whole-document prefix reaches at least **500,000,000 UTF-8 web-text bytes** and **1,000,000,000 UTF-8 Python bytes**, before filters. The boundary document is not clipped; overshoot is reported. These prefixes are not representative random samples. Upstream Parquet containers can require more transfer bytes than the selected text: report both, never describe selected text bytes as network transfer.

Before download, the owner-only manifest records source/revision/file, upstream LFS size/SHA-256, license/attribution, terms date and review purpose. After transfer, verify the complete container size/hash before parsing. The offline tool checks these receipts again and uses PyArrow only for file-format parsing, not training logic. No upstream loader script is run.

FineWeb-Edu carries ODC-By 1.0 plus Common Crawl terms; credit Hugging Face and Lozhkov et al. (2024). StarCoderData/The Stack carries its accepted source terms and the original code licenses; credit BigCode and the original repository authors. The Python schema supplies repository/path provenance but no per-file license column. Therefore only original-file declarations recognized as an allowlisted SPDX identifier, complete MIT notice, or Apache-2.0 notice are retained; missing, compound, conflicting and unsupported declarations are excluded, not inferred from the dataset label. Full source text including notices and repository/path provenance is preserved locally. This narrow declaration check is not a warranty of rights and does not cover every permissive license spelling.

## Filtering and contamination walkthrough

License admission comes first. The existing `filter_documents` then applies length, Python line/alphanumeric/generated-marker, exact-duplicate and contamination checks, assigning one primary rejection reason per document. Reports include zero-valued reasons and separate license drops; do not add overlapping observations as disjoint failures. Content hashes determine disjoint training/validation allocations, with separate stage-1/stage-2/validation web splits.

The contamination index includes separately hash-verified original HumanEval prompts, canonical completions and hidden scripts, pinned HumanEval+ and MBPP+ prompts, solutions and hidden scripts, plus ARC-Easy test questions. Source bytes are parsed only for exclusion, never added to training or tokenizer inputs. The normalized 13-word index also includes shorter complete fields. A final verified task-pool holdout does not yet exist; its coverage is explicitly **0**, not silently claimed complete. Repeat admission against that holdout before any full-corpus training. Verbatim-span filtering does not detect every paraphrase.

Step-3 token counts use the byte alphabet before BPE merges: one UTF-8 byte per token, no BOS. They are labeled as such, not presented as production-BPE counts. After the separately authorized tokenizer freeze, recount the retained samples with that exact merge-table identity and report held-out compression. No tokenizer is trained or silently changed by the review tool.

## Verification

`uv run --locked pytest tests/test_real_samples.py tests/test_filters.py` uses authored fixtures for UTF-8 whole-document selection, insufficient input, file-license decisions, primary filter accounting, provenance retention, contamination and deterministic splits. Real run values belong in the evidence section after collection; fixture tests are not sample measurements.

## Evidence

The completed 2026-10-19 laptop review at source `933820988f1959b8531d911bae993f040c7c45c6` selected the following prefixes. Counts are documents; bytes are UTF-8 text, separate from raw-container transfer. The tokenizer has not yet been frozen at this step.

| Measurement | Web | Python |
| --- | ---: | ---: |
| Sample documents | 103,985 | 209,914 |
| Sample text bytes | 500,038,096 | 1,000,000,226 |
| Missing/unapproved file-license drops | 0 | 198,883 |
| Documents entering shape/contamination filters | 103,985 | 11,031 |
| Too short / too large / long line / low alphanumeric / unknown domain | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| Generated-marker drops | 0 | 93 |
| Exact-duplicate drops | 13 | 0 |
| Contamination drops | 109 | 152 |
| Kept documents | 103,863 | 10,786 |
| Kept text bytes; pre-merge byte tokens, no BOS | 498,123,158 | 96,638,864 |
| Validation documents / bytes | 524 / 2,589,956 | 45 / 415,822 |

Retained Python declarations: Apache-2.0 **8,756**, MIT **1,832**, BSD-3-Clause **144**, BSD-2-Clause **51**, CC-BY-4.0 **2**, CC0-1.0 **1**. Web stage-1 allocation is **92,975 documents / 444,776,561 bytes**; disjoint stage-2 English is **10,364 / 50,756,641**. Python training allocation is **10,741 / 96,223,042**. Content hashes drive these allocations.

Raw container bytes downloaded: **2,152,819,114 web + 399,133,539 Python = 2,551,952,653**. Revisions and complete-container SHA-256:

- FineWeb-Edu `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `sample/10BT/000_00000.parquet`: `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`.
- StarCoderData `9fc30b578cedaec69e47302df72cf00feed7c8c4`, `python/train-00000-of-00059.parquet`: `407cf6628d01fe2da72ac301d669fc99a75b223a1defe71a48ab583dcb047c70`.

Derived retained-document JSONL hashes: web `8096430180bb736f9a977c973bb30756290674de3096efeb95351d980312509b`, Python `2d2a1accb9ed982a8a5526c20492f65b13b59f2fe172b4e466ee6c575fb3f0f6`. These files and private run receipts are not published. Exclusion inputs covered **164 original HumanEval + 164 HumanEval+ + 378 MBPP+ tasks and 2,376 ARC-Easy questions**; final-pool holdout coverage remains **0**. An additional **44,877-byte** hash-verified original HumanEval gzip was fetched for exclusion only, not training. Source notices/API/transport overhead are not included in corpus payload totals. Filtering completed in **144.4532 seconds** on the laptop; this is not a full-build throughput guarantee.

The tool's command is `uv run --locked python -m tools.real_samples --receipts <verified-receipts.json> --evalplus-directory <pinned-releases> --arc-directory <pinned-arc> --humaneval-path <pinned-original.gz> --output <new-directory>`. The complete local report is under ignored `runs/real-samples-20261019/filtered-complete/`. No paid resources, model training, full data build or GPU pilot ran. The full-build estimate remains a separate plan-only owner review.
