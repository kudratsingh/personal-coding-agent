# Public project status

**Updated:** 2026-10-16
**Stage:** Local research pipeline; phase acceptance incomplete

Local data/tokenization, model/trainer, generation, isolated code evaluation, and multiple-choice components are implemented and reference-tested. No trained coding capability, hosted service, or released weight is claimed.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.
- Validated YAML configuration and atomic, schema-tested run records with provenance and parent lineage.
- Local JSONL metrics, an opt-in tracker adapter, and structured debug logs with tracebacks.
- Manifest validation, deterministic filtering/splits, and a byte-level BPE implementation with reference tests.
- Shards, exact-resume training, CPU distributed checks, cached sampling, and a synthetic two-stage smoke path.
- Fail-closed pinned development sandbox with disposable Linux containment and end-to-end evaluation checks.
- Normalized tasks, conservative provenance-gated source adapters, reference-verified pool admission, metrics, code evaluation reports, and local scorecards.
- Held-out multiple-choice checkpoint scoring, with an explicit label-likelihood protocol and synthetic end-to-end tests.
- Periodic per-domain training validation, raw-byte-accurate bits per byte, and validation identity checks on exact resume.
- Standalone held-out checkpoint loss/bits-per-byte reports, restricted TACO I/O conversion, per-source pool drop counters, and split authorship summaries.
- Independently trained third-party BPE merge/encoding comparison, restricted to reference tooling and a dedicated CI job.
- Hash-bound evaluation-only EvalPlus OriginFmt intake, a locked NumPy runtime, full canonical preflight and optional project-checkpoint workflow. A failing preflight blocks scoring. A reviewed task-32 correction is explicit opt-in; unmodified tests remain the default, and corrected full-suite validation is pending.
- Real MBPP+ canonical validation passed 378/378; a synthetic-only nano checkpoint completed the suite with 0/378, including 156 explicit context limits. HumanEval+ remains blocked by the unchanged standalone task-32 oracle (163/164 canonical passes). See [evidence](evalplus.md).
- Complete evaluation-only [ARC-Easy intake](arc-easy.md): pinned test/train/validation bytes, 2,376 held-out test questions, valid answer keys and zero cross-split ID overlap. Dedicated parser CI and a synthetic-only nano validation workflow are provided; full checkpoint scoring is pending.

**Verification:** Latest local suite with the optional reference and evaluation extras installed: 148 passed, 17 explicitly skipped disposable-runtime tests; strict types, lint and public links pass. Dedicated CI exercises optional parsing/reference and actual isolation checks separately. A previous fresh public clone passed setup/check/smoke without private files; detailed evidence is recorded in the relevant PRs. Local skips are not evidence of isolation. See [EvalPlus validation](evalplus.md) for real-suite outcomes and blockers, rather than treating loader/unit tests as benchmark acceptance.

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
