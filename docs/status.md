# Public project status

**Updated:** 2026-10-21
**Real-source sample evidence updated:** 2026-10-20
**Stage:** Local research pipeline; phase acceptance incomplete

Local data/tokenization, model/trainer, generation, isolated code evaluation, and multiple-choice components are implemented and reference-tested. No trained coding capability, hosted service, or released weight is claimed.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Read-only publication checks reject tracked paths protected by repository ignore rules; public documentation checks exclude ignored local notes and reject symlink documents. These are path checks, not secret detection.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.
- Offline workflow runner checks flag declarations outside the reviewed standard-label subset. This is a limited static check, not billing enforcement or permission to incur costs.
- Pure compute estimates validate integer counts/finite inputs and reject unrepresentable stage or combined results rather than emitting NaN/Inf JSON; they do not authorize paid runs.
- Validated YAML configuration and atomic, schema-tested run records with provenance and parent lineage.
- Explicit schema-2 exact-resume checkpoint identity and failed-resume status/debug records, distinct from stage-parent lineage; legacy schema-1 record serialization and numerical checkpoint format 1 remain supported. See [ADR 0009](adr/0009-resume-provenance.md).
- Local JSONL metrics, an opt-in tracker adapter, and structured debug logs with tracebacks.
- Manifest validation, deterministic filtering/splits, and a byte-level BPE implementation with reference tests.
- Tokenizer merge tables are defensively owned and immutable, preventing caller mutation from changing hashes or diverging from cached encoders; valid format-1 identities remain unchanged.
- Strict manifest field types and canonical acceptance dates are checked before download I/O; malformed review flags cannot become truthy approvals. Source/license review remains a separate human responsibility.
- Download redirects retain the initial HTTPS/no-inline-credentials policy, including cross-host CDN hops; offline transport tests verify downgrade rejection and preservation of prior destination files.
- Shards, exact-resume training, CPU distributed checks, cached sampling, and a synthetic two-stage smoke path.
- Sampling rejects malformed controls/logits before RNG use and handles extreme finite scales with a shifted wide-precision fallback; ordinary seeded draws remain reference-tested against the prior path.
- Fail-closed pinned development sandbox with disposable Linux containment and end-to-end evaluation checks.
- Normalized tasks, conservative provenance-gated source adapters, reference-verified pool admission, metrics, code evaluation reports, and local scorecards.
- Held-out multiple-choice checkpoint scoring, with an explicit label-likelihood protocol and synthetic end-to-end tests.
- Periodic per-domain training validation, raw-byte-accurate bits per byte, and validation identity checks on exact resume.
- Standalone held-out checkpoint loss/bits-per-byte reports, restricted TACO I/O conversion, per-source pool drop counters, and split authorship summaries.
- Independently trained third-party BPE merge/encoding comparison, restricted to reference tooling and a dedicated CI job.
- Parallel tokenization bounds producer prefetch to 16 documents per worker while preserving order, with explicit early-close/error cleanup tests; this is not a byte/RSS limit.
- Hash-bound evaluation-only EvalPlus OriginFmt intake, a locked NumPy runtime, full canonical preflight and optional project-checkpoint workflow. A failing preflight blocks scoring. The reviewed task-32 correction remains explicit opt-in; unmodified tests remain the default.
- Real MBPP+ canonical validation passed 378/378; a synthetic-only nano checkpoint completed the suite with 0/378, including 156 explicit context limits. Current HumanEval+ scoring explicitly excludes task 32 as a known numerical-tolerance failure: canonical **163/163**, synthetic-only nano **0/163**, comprising **115 context limits and 48 candidate failures**. Every subset report discloses the exclusion; this is not the full 164-task suite. The task-32 thread is closed with no further diagnostics or oracle changes. See [evidence](evalplus.md).
- Complete evaluation-only [ARC-Easy intake and clean CPU run](arc-easy.md): 2,376 held-out test questions, valid answer keys and zero cross-split ID overlap. Synthetic-only nano scored 403/2,376 (16.96%), including 664 context-limit failures, matching a local run. This is integration evidence, not useful reasoning capability.
- Shard header/vocabulary/BOS integrity guards, with explicit tokenizer expectations in training/validation and unchanged legacy shard identities. See the [walkthrough](walkthroughs/shards.md).
- Loader resume-state validation rejects malformed source maps, counters, buffers and RNG state before changing the live loader; native/JSON exact continuation remains reference-tested.
- Loader source/share mappings are defensively copied so caller-map edits cannot silently change batches under an unchanged signature; corpus contents must remain immutable.
- Existing CPU bfloat16 autocast has fixed-seed float32 numerical comparisons, exact checkpoint/resume tests with gradient accumulation, and one-batch learning coverage. GPU validation remains separate.
- Two-process CPU Gloo restart tests compare float32 and CPU bfloat16 continuation bit-for-bit, including rank-specific loader/RNG state and checkpoint provenance; GPU and distributed fault-recovery acceptance remain separate.
- A bounded, CPU-only [checkpoint reference tool](cpu-resume-check.md) records exact uninterrupted/resumed comparisons on authored token IDs; CI exercises the tiny preset without network or paid resources. A local 11,536,640-parameter nano check completed all six seed/precision cases exactly at full 256-token context; this is not real-corpus, GPU or capability acceptance.
- Optimizer checkpoint loads validate moments, counters, masks and hyperparameters before applying state; malformed loads and copy failures leave the optimizer unchanged. Saved masks no longer alias live state.
- Optimizer construction uses the same hyperparameter rules and owns its mask/beta inputs, preventing caller edits or constructor-only malformed settings from breaking subsequent updates/checkpoint loads.
- Complete evaluation-only [original MBPP source audit](mbpp-audit.md): all 974 rows accounted for; 519 retained references passed both test partitions after benchmark-ID/schema/contamination checks. No training admission, full-pool approval or phase closure is implied.
- Bounded evaluation-only [OpenCodeInstruct prefix pilot](opencode-pilot.md) completed on a clean isolated worker: all 200 rows accounted for, 60 references passed both partitions after 35 schema, 13 contamination/duplicate and 92 reference-failure rejections. This is not a representative/full-source estimate, training admission or a model benchmark.

**Verification (2026-10-17):** Source revision `45866276154b9da50cc59eeb4b0fb11a2def0235` passed the owner-checkout suite with reference/evaluation extras: 632 passed, 18 explicitly skipped disposable-runtime tests; strict types, lint, workflow/publication checks and 47 public Markdown checks pass. A clean public clone at that same revision passed offline setup/check/smoke without private files: 608 passed, 41 explicit runtime/parser/reference skips. Both synthetic smoke stages also passed in the owner checkout. Dedicated CI exercises optional parsing/reference and actual isolation separately. Local skips are not evidence of isolation. See [EvalPlus validation](evalplus.md) for real-suite outcomes and blockers, rather than treating loader/unit tests as benchmark acceptance.

## Phase acceptance

**Latest local verification (2026-10-21):** corrected tokenizer source `e680745`,
combined with merged classifier PR #77,
passed **702 tests / 7 explicit NumPy-runtime skips** with the actual stdlib
sandbox enabled, strict types and lint; the final public documentation check
passes **60 Markdown files**. The dedicated local containment/
evaluation command previously passed **13/13** cases on October 20. The October 21
full suite includes those enabled stdlib containment/integration cases; an initial
sandboxed attempt failed three Gloo cases because loopback access was unavailable,
then the complete suite passed with local loopback/Docker access. No additional NumPy image,
reference-benchmark variant, task-32 diagnostic or paid run was launched. The
earlier verification paragraph retains its original revision/scope.

Condition-by-condition evidence reports: [G0](gates/g0-foundation.md), [G1](gates/g1-data-tokenizer.md), [G2](gates/g2-model-training.md), [G3](gates/g3-evaluation.md). They inventory existing runs/tests only and mark missing conditions. Reference G3 closure does not imply the original production-host condition or the entire phase is accepted.

| Phase | Implemented | Remaining acceptance |
| --- | --- | --- |
| 0. Foundation | Tooling, records, logs, estimator, CI | Final milestone review; live tracker delivery not exercised |
| 1. Data/tokenizer | Manifest controls, real-source sample reports, frozen BPE, fast encoder, shards/loader | Full-build estimate owner decision, remaining source/pool coverage and real-scale acceptance |
| 2. Model/training | CPU-reference model/optimizer/trainer, generation, sanity ladder | Approved GPU pilot, hardware precision/throughput and durable-run validation |
| 3. Evaluation/safety | Development sandbox, task/pool/metric tooling, code/choice CLI; reference G3 closed; disclosed 163-task HumanEval+ scoring complete | Complete real-source suite/holdout coverage, host signoff |

These are implementation milestones, not closed phase gates. Synthetic fixtures cannot substitute for real-source or hardware acceptance. No paid compute is authorized without explicit owner permission for the specific run.

**G3 (reference benchmark): CLOSED — owner accepted 14/164 (8.54%) on 2026-10-17**, within two problems of the card's 7.84%. There were **0 empty completions**, **157 budget hits**, and **164/164 canonical passes**. The protocol of record is trailing-whitespace-stripped prompts, greedy decoding, a 256-token budget and the original stop strings, recorded as `prompt_stripped: true`. See [protocol and evidence](reference-evaluation.md). This closes the reference gate, not remaining suite coverage, host approval or the entire evaluation phase.

## Next public milestone

Original intake downloaded **2,551,952,653 raw-container bytes**, selecting **500,038,096 web / 1,000,000,226 Python text bytes**. The current [repository-license review](real-samples.md) on those same files retains **103,863 web / 13,833 Python documents**, **498,123,158 / 122,701,374 text bytes**. Python retention remains **12.27013% by bytes**. **185,680 files / 786,817,959 bytes** lack repository-license metadata; only **8,737 files / 69,830,333 bytes** are unreadable inline headers. Copyleft, restrictions and quality drops are separate. The Parquet has no repository-license field; the fallback cannot admit extra files without a verified mapping. **New corpus download bytes: 0.** Final pool-holdout coverage is still unavailable; this is not full-build approval.

The owner-authorized [replacement tokenizer freeze](tokenizer-freeze.md) is
**65,536 entries**, single-leading-space format 2, on the identical **96,223,042
bytes/domain**. Merge SHA `b0297f478dd5172a690752ceac8f997fccbd95e36e5787ca56a2c555083c7b7b`.
Same-text original held-out BPT: project **4.99664 English / 4.33333 Python**,
GPT-2 **4.66758 / 2.19915**; equal-byte mix **4.64140 / 2.98969**. Fox sentence
**18 → 9 tokens**, GPT-2 **9**. One replacement fit, **1,013.82 CPU seconds**;
the October 19 artifact is preserved with legacy behavior. **No further
retraining without approval.** All specials fit inside IDs **0–65,535**;
[model counts](model.md) now include the **135,285,504-parameter S shape**.

The [four gate reports](gates/g0-foundation.md) and [12 Phase 4 work orders](rehearsal-plan.md) remain prepared, not blanket phase acceptance. HumanEval+ scoring still uses **163 tasks with task 32 explicitly excluded**; reference G3 stays closed. No further reference variants or task-32 investigations were run. The [GPU pilot](gpu-pilot-plan.md) has conditional owner approval, but launch still awaits the landed tokenizer, owner funding confirmation, durable checkpoint storage and operational readiness. **0 GPU-hours / 0 paid resources**.

The [revised P1-6 estimate](full-build-plan.md) remains **not approved**. The replacement-tokenizer recount gives **26,328,493 Python tokens / 122,701,374 bytes**, predicting **1.68502B Python tokens over 64GB**, or **1.67660B after validation reserve**, conditionally—not a measured full-source count. The old 2.24058B forecast used the superseded tokenizer. Cached-source timing covers **80 isolated partition calls**, with unmeasured sources/proxies explicit. The small rehearsal remains selected, but full builds and Phase 4 model runs are **0**. Owner review and the pilot's separate launch prerequisites remain outstanding.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
