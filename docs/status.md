# Public project status

**Updated:** 2026-10-16
**Stage:** Local research pipeline; phase acceptance incomplete

Local data/tokenization, model/trainer, generation, isolated code evaluation, and multiple-choice components are implemented and reference-tested. No trained coding capability, hosted service, or released weight is claimed.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Read-only publication checks reject tracked paths protected by repository ignore rules; public documentation checks exclude ignored local notes and reject symlink documents. These are path checks, not secret detection.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.
- Validated YAML configuration and atomic, schema-tested run records with provenance and parent lineage.
- Local JSONL metrics, an opt-in tracker adapter, and structured debug logs with tracebacks.
- Manifest validation, deterministic filtering/splits, and a byte-level BPE implementation with reference tests.
- Strict manifest field types and canonical acceptance dates are checked before download I/O; malformed review flags cannot become truthy approvals. Source/license review remains a separate human responsibility.
- Shards, exact-resume training, CPU distributed checks, cached sampling, and a synthetic two-stage smoke path.
- Fail-closed pinned development sandbox with disposable Linux containment and end-to-end evaluation checks.
- Normalized tasks, conservative provenance-gated source adapters, reference-verified pool admission, metrics, code evaluation reports, and local scorecards.
- Held-out multiple-choice checkpoint scoring, with an explicit label-likelihood protocol and synthetic end-to-end tests.
- Periodic per-domain training validation, raw-byte-accurate bits per byte, and validation identity checks on exact resume.
- Standalone held-out checkpoint loss/bits-per-byte reports, restricted TACO I/O conversion, per-source pool drop counters, and split authorship summaries.
- Independently trained third-party BPE merge/encoding comparison, restricted to reference tooling and a dedicated CI job.
- Hash-bound evaluation-only EvalPlus OriginFmt intake, a locked NumPy runtime, full canonical preflight and optional project-checkpoint workflow. A failing preflight blocks scoring. The reviewed task-32 correction remains explicit opt-in; unmodified tests remain the default.
- Real MBPP+ canonical validation passed 378/378; a synthetic-only nano checkpoint completed the suite with 0/378, including 156 explicit context limits. HumanEval+ remains blocked: the corrected run still passed 163/164, with task 32 now reporting `AssertionError`. Separate sandbox-only numerical diagnostics do not change scoring or authorize further oracle changes. See [evidence](evalplus.md).
- Complete evaluation-only [ARC-Easy intake and clean CPU run](arc-easy.md): 2,376 held-out test questions, valid answer keys and zero cross-split ID overlap. Synthetic-only nano scored 403/2,376 (16.96%), including 664 context-limit failures, matching a local run. This is integration evidence, not useful reasoning capability.
- Shard header/vocabulary/BOS integrity guards, with explicit tokenizer expectations in training/validation and unchanged legacy shard identities. See the [walkthrough](walkthroughs/shards.md).
- Loader resume-state validation rejects malformed source maps, counters, buffers and RNG state before changing the live loader; native/JSON exact continuation remains reference-tested.
- Existing CPU bfloat16 autocast has fixed-seed float32 numerical comparisons, exact checkpoint/resume tests with gradient accumulation, and one-batch learning coverage. GPU validation remains separate.
- Complete evaluation-only [original MBPP source audit](mbpp-audit.md): all 974 rows accounted for; 519 retained references passed both test partitions after benchmark-ID/schema/contamination checks. No training admission, full-pool approval or phase closure is implied.

**Verification:** Latest owner-checkout suite with reference/evaluation extras: 285 passed, 18 explicitly skipped disposable-runtime tests; strict types, lint, publication-path checks and 44 public Markdown checks pass. A fresh public clone at `fdd3755` passed setup/check/smoke without private files: 226 passed, 31 explicit runtime/parser/reference skips. Dedicated CI exercises optional parsing/reference and actual isolation separately. Local skips are not evidence of isolation. See [EvalPlus validation](evalplus.md) for real-suite outcomes and blockers, rather than treating loader/unit tests as benchmark acceptance.

## Phase acceptance

| Phase | Implemented | Remaining acceptance |
| --- | --- | --- |
| 0. Foundation | Tooling, records, logs, estimator, CI | Final milestone review; live tracker delivery not exercised |
| 1. Data/tokenizer | Manifest controls, filters, BPE, fast encoder, shards/loader | Reviewed real-source samples/reports, production tokenizer freeze, measured full-build review |
| 2. Model/training | CPU-reference model/optimizer/trainer, generation, sanity ladder | Approved GPU pilot, hardware precision/throughput and durable-run validation |
| 3. Evaluation/safety | Development sandbox, task/pool/metric tooling, code/choice CLI | Complete real-source suite/holdout coverage, host signoff, reference-score discrepancy |

These are implementation milestones, not closed phase gates. Synthetic fixtures cannot substitute for real-source or hardware acceptance. No paid compute is authorized without explicit owner permission for the specific run.

## Next public milestone

Complete reviewed source/suite integration and reconcile the reference benchmark before making capability claims. The [model comparison and diagnostic rerun](reference-evaluation.md) each passed 1/164 HumanEval tasks (0.61%), not the card's 7.84%; canonical checks passed 164/164. The gap is repeatable but unexplained. GPU work and gated-data acceptance remain separately approval-gated.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
