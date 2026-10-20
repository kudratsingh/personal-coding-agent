# Real pretraining sample review

`tools/real_samples.py` reviews local, pinned Parquet containers without executing their contents. Raw data, access tokens, source notices and run artifacts stay ignored. Corpus access does not imply blanket downstream rights or authorization for a full build.

## Selection and provenance

The named sources are FineWeb-Edu `sample-10BT` and StarCoderData Python. An ordered whole-document prefix reaches at least **500,000,000 UTF-8 web-text bytes** and **1,000,000,000 UTF-8 Python bytes**, before filters. The boundary document is not clipped; overshoot is reported. These prefixes are not representative random samples. Upstream Parquet containers can require more transfer bytes than the selected text: report both, never describe selected text bytes as network transfer.

Before download, the owner-only manifest records source/revision/file, upstream LFS size/SHA-256, license/attribution, terms date and review purpose. After transfer, verify the complete container size/hash before parsing. The offline tool checks these receipts again and uses PyArrow only for file-format parsing, not training logic. No upstream loader script is run.

FineWeb-Edu carries ODC-By 1.0 plus Common Crawl terms; credit Hugging Face and Lozhkov et al. (2024). StarCoderData/The Stack carries its accepted source terms and the original code licenses; credit BigCode and the original repository authors. The Python schema supplies repository/path provenance but no per-file license column. The current header classifier keeps recognized Apache-2.0, MIT, BSD-2/3-Clause, ISC, PSF-2.0, Unlicense and CC0 grants, with per-file IDs. Copyleft, restricted and unrecognized headers are separate rejection classes. Full source text including notices and repository/path provenance is preserved locally. This heuristic is not a warranty of rights and does not cover every permissive spelling. The earlier quality-only bypass was diagnostic, not admission, and has been removed.

## Filtering and contamination walkthrough

License admission comes first. The existing `filter_documents` then applies length, Python line/alphanumeric/generated-marker, exact-duplicate and contamination checks, assigning one primary rejection reason per document. Reports include zero-valued reasons and separate license drops; do not add overlapping observations as disjoint failures. Content hashes determine disjoint training/validation allocations, with separate stage-1/stage-2/validation web splits.

The contamination index includes separately hash-verified original HumanEval prompts, canonical completions and hidden scripts, pinned HumanEval+ and MBPP+ prompts, solutions and hidden scripts, plus ARC-Easy test questions. Source bytes are parsed only for exclusion, never added to training or tokenizer inputs. The normalized 13-word index also includes shorter complete fields. A final verified task-pool holdout does not yet exist; its coverage is explicitly **0**, not silently claimed complete. Repeat admission against that holdout before any full-corpus training. Verbatim-span filtering does not detect every paraphrase.

Step-3 token counts use the byte alphabet before BPE merges: one UTF-8 byte per token, no BOS. They are labeled as such, not presented as production-BPE counts. After the separately authorized tokenizer freeze, recount the retained samples with that exact merge-table identity and report held-out compression. No tokenizer is trained or silently changed by the review tool.

The subsequent [frozen-tokenizer recount](tokenizer-freeze.md) completed: **182,749,140 web** and **27,195,275 Python tokens**, excluding BOS. These are the production-BPE totals, distinct from the historical step-3 byte counts below.

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

## 2026-10-20 retention audit

The original **9.66388% byte retention is not a quality-filter retention rate**.
An offline rerun reproduced both historical output hashes exactly and attributed
the Python removals below. The inline-license gate accounts for **99.18245% of
removed bytes**. Requiring a license header in every file is not equivalent to
checking a repository license; absence of a recognized header is not evidence of
bad code or necessarily an unlicensed file.

| Primary rule, in order | Threshold or logic | Python files removed | Python UTF-8 bytes removed |
| --- | --- | ---: | ---: |
| Missing/unapproved inline license | First 16,384 characters; one allowlisted SPDX identifier, MIT notice markers, or Apache-2.0 notice; otherwise reject | 198,883 | 895,975,942 |
| Unknown domain | Domain must be web or Python | 0 | 0 |
| Too short | Fewer than 32 UTF-8 bytes | 0 | 0 |
| Too large | More than 1,048,576 UTF-8 bytes | 0 | 0 |
| Long line | Any Python line longer than 1,000 characters | 0 | 0 |
| Low alphanumeric share | Python alphanumeric characters / all characters less than 0.1 | 0 | 0 |
| Generated marker | Case-insensitive `auto-generated`, `autogenerated`, or `do not edit` anywhere | 93 | 4,058,112 |
| Exact duplicate | SHA-256 of whole text already retained in this source | 0 | 0 |
| Contamination | Any normalized 13-word match, or shorter complete indexed field | 152 | 3,327,308 |
| Kept (not a rejection) | Passed the preceding rules | 10,786 | 96,638,864 |

The input totals **209,914 files / 1,000,000,226 bytes**. Primary counts are
sequential, not independent all-rule matches. In particular, zero quality drops
after the old license gate says nothing about the 198,883 files it never examined.
License observations in a quality-only review overlap with quality decisions and
must not be summed as additional removals.

The SPDX allowlist is MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0,
CC-BY-4.0, ODC-By-1.0 and CDLA-Permissive-2.0. Conflicting/compound identifiers
are unresolved. Without SPDX, GNU/CC-BY-SA/CC-BY-NC/non-commercial header markers
block recognition; MIT requires the permission, copyright-retention and warranty
markers, while Apache requires its version-2.0 licensing phrase. Normalized
contamination words are Unicode `\w+` matches after case folding, not BPE tokens.

The first diagnostic separated quality accounting from rights admission, without
relaxing quality thresholds. Its historical bypass retained unresolved evidence
as null for **audit only**. That bypass is removed from the current tool under
the subsequent owner-requested header-classifier policy. Repository/component
rights and notices remain a separate full-build requirement. See [ADR 0011](adr/0011-quality-retention-and-rights-admission.md).
No original artifact or frozen tokenizer is overwritten, and no corpus is fetched.

### Historical quality-only rerun on the same input — not current admission

The single rerun retained **204,915 Python files / 934,016,511 bytes**:
**97.61855% of files / 93.40163% of bytes**. This exceeds the requested 60–85%
expectation; no threshold was changed to force that range. The input is already
cleaned upstream, and these are the actual outcomes under the stated rules.

| Primary quality rule | Files removed | Bytes removed | Bytes remaining | Cumulative byte retention |
| --- | ---: | ---: | ---: | ---: |
| Unknown domain | 0 | 0 | 1,000,000,226 | 100% |
| Too short (<32 bytes) | 668 | 16,076 | 999,984,150 | 99.99839% |
| Too large (>1 MiB) | 0 | 0 | 999,984,150 | 99.99839% |
| Long line (>1,000 characters) | 0 | 0 | 999,984,150 | 99.99839% |
| Alphanumeric share (<0.1) | 0 | 0 | 999,984,150 | 99.99839% |
| Generated marker | 1,673 | 27,989,051 | 971,995,099 | 97.19949% |
| Exact duplicate | 1 | 85 | 971,995,014 | 97.19948% |
| Contamination (same 13-word/short-field rule) | 2,657 | 37,978,503 | 934,016,511 | 93.40163% |

**194,129 retained files still have unresolved inline license evidence.** They
are not newly cleared for training. The retained recognized-declaration counts
remain unchanged at 10,786. A downstream training build must resolve original
repository/component licensing and attribution, not mistake this quality-only
output for an admitted corpus. Default admission remains unchanged.

Quality-review Python JSONL SHA-256:
`d57313145d3a3af62c0d586b2c5a4bb376b20366a2c2c89af36c2525e856c2a3`.
Web remains **103,863 files / 498,123,158 bytes / 99.61704% byte retention**,
with the original JSONL hash. The new local reports are under
`runs/retention-audit-20261020/{baseline,quality-review}/`. They preserve primary
byte/file accounting and explicit non-admission flags. New corpus download bytes
and tokenizer fits during this audit: **0**.

### Historical header-classifier-v1 result

The owner's subsequent policy replaces the original blanket check with a
classifier, **not** the historical bypass. It reads leading comments/module
strings in the first **16,384 characters**, ignoring known leading metadata
lines only for classification. Stored text and quality thresholds are unchanged.
Recognized single SPDX IDs, named declarations and grant patterns are retained;
unsupported/compound grants are unrecognized. Restrictive markers take precedence.
"All rights reserved" is not itself a rejection. See [ADR 0011](adr/0011-quality-retention-and-rights-admission.md).

The corrected run used the same input containers and selected prefixes; it took
**182.8918 seconds**. The Python results below are disjoint final outcomes:

| Class | Files | UTF-8 bytes | Input bytes |
| --- | ---: | ---: | ---: |
| Kept permissive after quality | 13,833 | 122,701,374 | 12.27013% |
| Dropped copyleft (GPL/LGPL/AGPL/MPL markers) | 1,220 | 13,290,562 | 1.32906% |
| Dropped non-commercial/proprietary restrictions | 125 | 928,113 | 0.09281% |
| Dropped unrecognized/no grant | 194,417 | 856,648,292 | 85.66481% |
| Dropped by quality/contamination | 319 | 6,431,885 | 0.64319% |
| Total | 209,914 | 1,000,000,226 | 100% |

Before quality filtering, recognized permissive headers account for
**14,152 files / 129,133,259 bytes**. Quality rejects **116 generated-marker
files / 2,204,697 bytes** and **203 contamination files / 4,227,188 bytes**;
all other quality rules remove **0 files / 0 bytes**. Contamination remains
**13 normalized words plus shorter complete indexed fields**. Do not add this
pre-quality permissive count to the final-kept row.

Kept license IDs: Apache-2.0 **8,776**, MIT **3,635**, BSD-2-Clause **280**,
BSD-3-Clause **1,009**, ISC **82**, PSF-2.0 **0**, Unlicense **30**, CC0-1.0 **21**.
Each retained row records its ID/evidence, repository/path, content hash and full
original text/notices. The first five raw lines of ten unrecognized examples
are saved in ignored `runs/header-classifier-fixed-20261020/python/unrecognized-examples.json`,
not published in Git. Most illustrated files begin with imports or metadata,
not a grant. No repository-level license is inferred from missing inline evidence.

The **12.27013%** byte retention (**6.58984%** of files) remains below 60–85%
because the specified unrecognized-header drop is still substantial. The 93.4%
diagnostic rate is not the rate used for the revised admitted-header forecast.
No threshold or rights relaxation is used to manufacture a target percentage.

Current JSONL hashes: Python
`220f6b1feaae0235a35a3e79ea6c54e8d3c1945cabb1cedee63dd5414c530b27`;
web `88b25b977364bd1f271ac1500760f555a0fb8b04f0f6b82b3dcf9e64694b5ccc`.
Web text/counts/splits are unchanged; its JSONL hash changes because per-file
classification evidence is now included. The current run directory is
`runs/header-classifier-fixed-20261020/`. Its first classifier predecessor
missed grants behind upstream metadata; a reproduced failing synthetic regression
led to the correction. That predecessor and its interrupted recount are
superseded, not accepted inputs. The frozen tokenizer was never changed.

### Repository fallback correction (2026-10-20, current)

Absent inline licensing now uses an explicit `repository_license` metadata
field under the [D-05 fallback policy](adr/0011-quality-retention-and-rights-admission.md).
Off-allowlist repository IDs have separate counters; missing metadata is not
an unrecognized header. An unreadable/restrictive inline header still takes
precedence over a repository-level grant. Ordinary comments/docstrings without
license signals are not inline license declarations.

The pinned Parquet has **zero repository-license fields or mappings**, so the
fallback admits **0 additional files**. A verified mapping is required; the
earlier prefix-parser fix provided no licenses. The actual rerun retained the
same text and output hashes, with the corrected attribution below:

| Final primary class | Python files | UTF-8 bytes |
| --- | ---: | ---: |
| Kept permissive, after quality | 13,833 | 122,701,374 |
| Dropped copyleft header | 1,220 | 13,290,562 |
| Dropped restricted header | 125 | 928,113 |
| Dropped unrecognized inline header | 8,737 | 69,830,333 |
| Unresolved: absent header and missing repository license | 185,680 | 786,817,959 |
| Dropped off-allowlist repository license | 0 | 0 |
| Dropped by quality/contamination | 319 | 6,431,885 |
| Total | 209,914 | 1,000,000,226 |

Byte retention remains **12.2701346%**, not an assumed 60–85%. All retained
files still carry their inline license IDs. Repository-ID rejections are an
empty map because no such metadata was supplied, **not** because all repositories
are permissible. Quality removals remain **116 generated / 203 contamination**;
the contamination index remains 13 normalized words plus shorter complete fields.
Ten actual unrecognized-header examples, first five lines each, are saved in
ignored `runs/repository-license-review-v2-20261020/python/unrecognized-examples.json`.
The revised [capacity estimate](full-build-plan.md) must use only verified retained
bytes, not the unresolved 786,817,959 bytes. New corpus downloads: **0 bytes**.

Authored tests cover allowed repository fallback, rejected IDs, absent metadata,
ordinary module comments, per-file credit and explicit-header precedence.
The local full check passed **699 tests, 7 explicit NumPy-runtime skips** with
the pinned stdlib isolation tests enabled. This does not approve a full build.

Verification: **40 targeted tests**, and **692 passed / 7 explicit NumPy-runtime
skips** in the full local check with stdlib Docker containment enabled. The
classifier reads text/token syntax only and never executes corpus programs.
