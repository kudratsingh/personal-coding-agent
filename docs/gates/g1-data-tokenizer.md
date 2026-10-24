# G1 — data and tokenizer evidence

## Current amendment — 2026-10-20

This block supersedes freeze/compression values in the historical inventory
below. The owner authorized exactly one replacement fit, not a full build.

| Condition | Current proof / measured value | Verdict |
| --- | --- | --- |
| Frozen tokenizer and dated decision | [Replacement freeze](../tokenizer-freeze.md), D-24: **65,536 entries / 65,264 merges**, merge SHA `b0297f478dd5172a690752ceac8f997fccbd95e36e5787ca56a2c555083c7b7b` | Met; replaces October 19 D-19 |
| Same balanced fit selection, held-out compression | **96,223,042 bytes/domain**, original corpus hashes unchanged; English **518,340 tokens / 2,589,956 bytes = 4.9966354130 BPT**; Python **95,959 / 415,822 = 4.3333298596**; 50/50 bytes **4.6414040782** | Met at measured scope; optional 32,768 fit **0** |
| Reference, compatibility and shard boundary | **64 targeted tests**, full **702 passed / 7 explicit NumPy skips**; maximum ID **65,535**, all **16 specials** included; legacy identities preserved | Met at CPU/fixture scope |
| Original versus current license-review selections | Original fit-review **10,786 Python files / 96,638,864 bytes**; 2026-10-20 header-only review **13,833 / 122,701,374**; current upstream-license review **189,369 / 824,650,726** after an exact repository/path join matched all **209,914** files; **8,909 unrecognized or unreadable headers** and **5,695 off-allowlist repository files** stay dropped | Repository IDs from The Stack dedup v1.2 metadata only; no inference; data-card credits retained |
| 100 MB real slow/fast identity and loader/GPU throughput | Actual real probe **512 characters**; GPU runs **0** | Still not met |
| Current review recount and capacity | **197,448,213 Python tokens / 824,650,726 bytes**, **4.1765418561 BPT**; conditional 64GB forecast **12.63668B tokens**, **12.57350B after validation reserve** (2026-10-20 header-only: 26,328,493 tokens, 1.68502B forecast) | Encoding-only frozen-tokenizer recount; ordered single-shard prefix, not a full-source count |
| Full-build decision | [P1-6](../full-build-plan.md), owner approvals/deferrals **0** | Still not met; overall G1 remains open |

The following inventory preserves the original measurement scopes and historical
October 19 freeze. It is not permission to use that superseded tokenizer for
the new pilot or to reinterpret its old checkpoint identities.

Report date: 2026-10-19; retention/estimate evidence revised 2026-10-20. **Tokenizer-freeze condition met; overall roadmap gate awaits owner approval/deferral of the revised full-build estimate.** This page inventories the cited measurements; it does not itself launch a run or authorize a build.

| Roadmap condition | Existing command / record | Value and verdict |
| --- | --- | --- |
| G1: tokenizer frozen, hash under Decided | `python -m tools.freeze_tokenizer …`; [freeze record](../tokenizer-freeze.md), owner decisions D-19 | **32,768 vocabulary / 32,496 merges**; merge SHA `435729be21650397918abf42c56fd2aee742b12dbc3644e7095ee7c58fc0c1e7`. **Met.** One production training call; no retrain without approval. |
| G1: data card lists sources/licenses/credits | [Data card](../data-card.md) and [sample provenance](../real-samples.md) | Real retained text **498,123,158 web / 96,638,864 Python bytes**, with **6 retained Python license identifiers**, source notices/provenance local; benchmark data evaluation-only. **Met for sources actually used**, not a claim of full planned-source admission. |
| G1: full-build estimate approved or deferred by owner | P1-6 plan-only experiment record / owner decision | **0 recorded full-build approvals or deferrals. Not met.** Preparing an estimate does not accept it. |
| P1-1: all named source samples, manifests, credits/terms, identical downloads twice | `tools/real_samples.py` and pinned download receipts; prior evaluation-source audits | Raw web **2,152,819,114 bytes**, Python **399,133,539**, both exact upstream SHA matches; original HumanEval exclusion gzip **44,877 bytes** separately. **1 corpus download per source, not 2. Partial.** Public full-build manifest has **0 approved entries**; sample manifests are ignored. Full AceCode, TACO and language-SFT sample acceptance is absent. |
| P1-2: planted contamination removed; per-filter real-sample counts | `pytest tests/test_filters.py tests/test_real_samples.py`; [full table](../real-samples.md) | **3 + 4 tests passed**. Web drops **13 duplicates / 109 contamination**; Python drops **198,883 license / 93 generated / 152 contamination**; all other primary filter drops **0**. Exclusion coverage **164 original HE + 164 HE+ + 378 MBPP+ + 2,376 ARC questions**. Final-pool holdout coverage **0: not met for that future pool**. |
| P1-3: half web/code, Unicode/code round-trip, deterministic merges, trusted reference, held-out compression | `pytest tests/test_tokenizer.py tests/test_tokenizer_external.py tests/test_freeze_tokenizer.py`; freeze report | **96,223,042 training bytes per domain**, no validation; **15** repeat/slow-reference corpora and **100 random Unicode examples** in passing fixtures; **2 external-reference + 2 freeze tests** passed. Held-out English **2,589,956 bytes / 944,567 tokens = 2.7419505445 BPT**; Python **415,822 / 123,384 = 3.3701452376 BPT**. **Met at stated fixture/real-measurement scopes.** No second production fit. |
| P1-4: 100 MB reference/fast identity and throughput per core | Existing bulk/fast tests; frozen real probes and full recount | **512 real probe characters** have identical reference/fast IDs, not **100 MB**. Full retained recount: **498,123,158 bytes / 53.8215 s web**, **96,638,864 / 31.9963 s Python**, each with **4 workers**, including I/O/probes. **100 MB identity not met.** These are pipeline wall times, not pure per-core encoder benchmarks. |
| P1-5: disjoint splits, deterministic/resumable batches, share within 1 point; loader faster than one GPU | `pytest tests/test_filters.py tests/test_shards.py tests/test_distributed.py`; source split report | Split/hash fixture checks passed; loader compares **5 resumed batches after 7**, share assertion uses **1,000 batches × 16 sequences**, error **<0.01**; **2 ranks** partition **10 documents** with **0 overlap**. Real validation **524 web / 45 Python docs**, disjoint content-hash allocation. **Partial:** no measured loader-versus-GPU throughput; GPU consumption **not measured**. |
| P1-6: full-build estimate in experiment record, owner accepts/changes/defers | [Revised plan status](../full-build-plan.md); owner-only P1-6 Markdown/JSON record, 2026-10-20 | Owner requested revision; revised estimate **prepared, not approved**. Classifier byte retention **12.27013%**; **35,009,073 Python sample tokens**; **2.24058B-token forecast over 64GB**, not a full-source count. **No full build permitted.** |

Production sample BPE totals are **182,749,140 web / 27,195,275 Python tokens**, excluding BOS. These replace neither the explicitly labeled historical byte-alphabet counts nor final corpus admission. Source prefixes are ordered, not representative samples; final task-pool decontamination and full-build approval remain separate.

The other table rows above retain their original 2026-10-19 proof scope. The
2026-10-20 [classifier revision](../real-samples.md) is a separate review output:
**13,833 Python files / 122,701,374 bytes / 35,009,073 tokens**, **7 actually
retained license IDs**, validation **52 files / 475,430 bytes / 140,646 tokens**.
Its per-class/quality counters do not relabel the original tokenizer-fit corpus.
Current recount times: **65.55425s web / 50.73031s Python**, four workers.
The **100MB real reference/fast identity condition remains unmet**; no second
production fit or full-pool exclusion is claimed. Full-build approvals/deferrals
remain **0**, despite the owner's request to revise the draft.
