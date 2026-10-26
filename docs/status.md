# Public project status

**Updated:** 2026-10-26
**Real-source sample evidence updated:** 2026-10-24
**Stage:** Local research pipeline; phase acceptance incomplete

Local data/tokenization, model/trainer, generation, isolated code evaluation, and multiple-choice components are implemented and reference-tested. No trained coding capability, hosted service, or released weight is claimed.

## Latest result: rehearsal pretraining (2026-10-26)

Two rehearsal pretraining runs of the 135M-parameter S model completed on 2026-10-26, each on one A40. Both started from one saved initialization and read the same 2B-token multiset.
- **Run A (two-stage):** 1B English tokens, then 0.9B Python and 0.1B unseen English, with a fresh optimizer and schedule.
- **Run B (mixed):** the same tokens shuffled together, on one schedule.

Full-validation bits per byte:

| Checkpoint | English (web) | Python |
| --- | ---: | ---: |
| Run A, end of stage 1 | 1.0538 | 2.3156 |
| Run A, two-stage final | 1.1328 | **0.7747** |
| Run B, mixed final | **1.0637** | 0.7827 |

**Finding (owner-accepted):**
- **Python:** a tie within noise, in the owner's reading. Two-stage is lower by 0.0079 bits per byte, about 1.0%, just over the pre-registered 1% threshold, with one seed per arm.
- **English:** mixed is better by 6.1% of two-stage's value.
- **Forgetting:** stage 2 cost the two-stage run 0.079 bits per byte of English.
- **What it compares:** two recipes, not order alone, since the two-stage run also reset its optimizer and schedule, at half the peak learning rate.
- **Not measured:**
  - the pre-registered dev pass rate after matched SFT, because post-training continues on run B only;
  - ARC-Easy accuracy for these checkpoints, which is still to come.

**Decision:** the main model will pretrain on one mixed stage with one schedule. It keeps its planned token streams, 9B web English plus 9B of Python with 10% English, shuffled together. The two-stage code path stays as an experiment.

**Next:** the rest of the rehearsal continues from run B's checkpoint: SFT on code and language, a short RL run with a "more SFT" control, and full evaluation. One proposal covers all three, priced at the measured A40 speed; nothing is spent before owner approval.

**Cost and teardown:** run A was billed $6.40 and run B $6.44, each under its $10 cap. Both Pods were deleted after their final checkpoints were verified in R2.

Details: [measured results](rehearsal-runs.md#measured-results-2026-10-26). These are base-model validation bits per byte; they are not coding-capability scores.

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
- File training accepts explicit CPU/CUDA device selection with fail-before-run checks and records the resolved device. Step metrics include tokens and elapsed seconds, with CUDA synchronization at both timing boundaries; CPU/mocked-CUDA checks do not establish actual GPU correctness or throughput. See the [training walkthrough](walkthroughs/training-loop.md).
- Two-process CPU Gloo restart tests compare float32 and CPU bfloat16 continuation bit-for-bit, including rank-specific loader/RNG state and checkpoint provenance; GPU and distributed fault-recovery acceptance remain separate.
- A bounded, CPU-only [checkpoint reference tool](cpu-resume-check.md) records exact uninterrupted/resumed comparisons on authored token IDs; CI exercises the tiny preset without network or paid resources. A local 11,536,640-parameter nano check completed all six seed/precision cases exactly at full 256-token context; this is not real-corpus, GPU or capability acceptance.
- Optimizer checkpoint loads validate moments, counters, masks and hyperparameters before applying state; malformed loads and copy failures leave the optimizer unchanged. Saved masks no longer alias live state.
- Optimizer construction uses the same hyperparameter rules and owns its mask/beta inputs, preventing caller edits or constructor-only malformed settings from breaking subsequent updates/checkpoint loads.
- Complete evaluation-only [original MBPP source audit](mbpp-audit.md): all 974 rows accounted for; 519 retained references passed both test partitions after benchmark-ID/schema/contamination checks. No training admission, full-pool approval or phase closure is implied.
- GPU pilot readiness audit (2026-10-22): checked-in A40 specification with a test that pins the 135,285,504-parameter S shape, 131,072-token updates and the 700-update / 91,750,400-token ceiling; probe window counted from the first probe with `incomplete`/`skipped` statuses; probes run with two accumulated microbatches plus a memory-headroom rule; a finalization reserve bounds the training deadline; checkpoint SHA-256 sidecars verified before every resume; the continuation keeps the winner's screen checkpoint; a fresh-trainer resume test preserves provenance; a memory-lean cross-entropy and an opt-in fused attention kernel, both with equivalence tests (CPU activation proxy per 1,024-token sequence: 2,919 MiB before, 1,760 MiB after); GPT-2 depth-scaled residual initialization, single-sync gradient clipping, and opt-in list-kernel AdamW, z-loss, QK-normalization, warmup-stable-decay and `torch.compile`, each reference-tested and summarized with sources in the [training recipe](training-recipe.md); a token-level validation leakage audit; the watchdog no longer reads a first-poll 404 as termination, backs off on rate limits and allows a bounded Mac-observed transfer window; an in-Pod self-destruct timer, Pod-identity capture, receipt assembler and feasibility calculator. The untrained S checkpoint measured 1,623,502,465 bytes (1.512 GiB) on CPU. No GPU, Pod or paid resource was used; termination remains layered best-effort because Runpod has no provider-enforced Pod lifetime. See [PILOT_LAUNCH_READINESS.md](../PILOT_LAUNCH_READINESS.md).
- Upstream repository-license recovery (2026-10-24): `tools/stack_licenses.py` joins the Python sample to The Stack dedup v1.2 `max_stars_repo_licenses` on the exact repository and path, reading those three columns and the Parquet footers over HTTPS range requests and refusing any request that would touch a content column chunk (footer min/max statistics, which can hold short files, are disclosed and measured); every listed license ID must be on the D-05 allowlist and explicit headers keep precedence. Authored fixtures cover the byte-range guard, join classes, mapping identity, token handling and review integration. See [sample evidence](real-samples.md).
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
| 2. Model/training | CPU-reference model/optimizer/trainer, generation, sanity ladder; one completed A40 pilot (2026-10-24) with measured precision, memory, throughput, checkpoint transfer and verified shutdown | Phase gate review of the pilot evidence; multi-run and distributed fault-recovery acceptance |
| 3. Evaluation/safety | Development sandbox, task/pool/metric tooling, code/choice CLI; reference G3 closed; disclosed 163-task HumanEval+ scoring complete | Complete real-source suite/holdout coverage, host signoff |
| 4. Rehearsal | S build, task pool and holdouts; rehearsal pretraining runs A and B completed on two A40s (2026-10-26) and the order comparison recorded | SFT, short RL with its more-SFT control, full evaluation and the local demo, each with its own approval; G4 review |

These are implementation milestones, not closed phase gates. Synthetic fixtures cannot substitute for real-source or hardware acceptance. No paid compute is authorized without explicit owner permission for the specific run.

**G3 (reference benchmark): CLOSED — owner accepted 14/164 (8.54%) on 2026-10-17**, within two problems of the card's 7.84%. There were **0 empty completions**, **157 budget hits**, and **164/164 canonical passes**. The protocol of record is trailing-whitespace-stripped prompts, greedy decoding, a 256-token budget and the original stop strings, recorded as `prompt_stripped: true`. See [protocol and evidence](reference-evaluation.md). This closes the reference gate, not remaining suite coverage, host approval or the entire evaluation phase.

## Next public milestone

Original intake downloaded **2,551,952,653 raw-container bytes**, selecting **500,038,096 web / 1,000,000,226 Python text bytes**. The current [repository-license review](real-samples.md) on those same files retains **103,863 web / 190,222 Python documents**, **498,123,158 / 828,088,405 text bytes**. Python retention is **82.80882% by bytes**, up from 12.27013%, after the owner-directed recovery of upstream repository licenses: an exact repository/path join to The Stack dedup v1.2 `max_stars_repo_licenses` matched all **209,914** sample files, reading the key and license columns plus Parquet footers (**822,321,592 bytes**; no content column chunks, but the footers' content statistics embed an estimated 24 MB of short-file text, held in memory only). **8,909 files / 71,345,855 bytes** remain unrecognized or unreadable inline headers and **4,828 files / 29,270,798 bytes** carry off-allowlist repository licenses awaiting an owner decision. Copyleft, restrictions and quality drops are separate. The owner corrected D-05 to list ISC, Unlicense, 0BSD and WTFPL, which were always allowed; other off-allowlist IDs stay out, and the owner accepted the footer-statistics exposure. **New corpus content bytes: 0.** Final pool-holdout coverage is still unavailable; this is not full-build approval.

The owner-authorized [replacement tokenizer freeze](tokenizer-freeze.md) is
**65,536 entries**, single-leading-space format 2, on the identical **96,223,042
bytes/domain**. Merge SHA `b0297f478dd5172a690752ceac8f997fccbd95e36e5787ca56a2c555083c7b7b`.
Same-text original held-out BPT: project **4.99664 English / 4.33333 Python**,
GPT-2 **4.66758 / 2.19915**; equal-byte mix **4.64140 / 2.98969**. Fox sentence
**18 → 9 tokens**, GPT-2 **9**. One replacement fit, **1,013.82 CPU seconds**;
the October 19 artifact is preserved with legacy behavior. **No further
retraining without approval.** All specials fit inside IDs **0–65,535**;
[model counts](model.md) now include the **135,285,504-parameter S shape**.

The [four gate reports](gates/g0-foundation.md) and [12 Phase 4 work orders](rehearsal-plan.md) remain prepared, not blanket phase acceptance. HumanEval+ scoring still uses **163 tasks with task 32 explicitly excluded**; reference G3 stays closed. No further reference variants or task-32 investigations were run. The [GPU pilot](gpu-pilot-plan.md) has conditional owner approval and a completed readiness audit ([verdict and blockers](../PILOT_LAUNCH_READINESS.md), [runbook](../PILOT_RUNBOOK.md)); the owner selected the agreed specification on 2026-10-24, with `torch.compile` off for training and one separate 20-update compiled probe that measures the speedup. The owner authorized launch on 2026-10-24 and the pilot completed: the A40 preflight passed after a UUID/memory-report fix, the 6e-4 arm won the screen and reached step 500 (**91,750,400 tokens at 35,974 tok/s**, final held-out loss **5.5789**), the compile probe measured **1.57x**, the checkpoint was verified on the Mac and the Pod was deleted (204, then 404). **One Pod for 3,679 seconds, $0.60 spent.** See the [measured results](gpu-pilot-plan.md#measured-a40-pilot-2026-10-24).

The [revised P1-6 estimate](full-build-plan.md) is **approved for the S rehearsal build only, effective after the GPU pilot report** (owner, 2026-10-24); M, L and other full builds are not approved. The [S build ran locally on 2026-10-25](rehearsal-data.md): exactly **1,000,000,000 / 100,000,000 / 900,000,000** stage-1 English, stage-2 English and stage-2 Python shard tokens, Python byte retention **83.22674%** over all 1,090,395 files of five containers, verification passed, $0. After the task-pool holdout exclusion (2,940 documents dropped, quotas still exact) the shards were [admitted for training on 2026-10-25](rehearsal-data.md#holdout-exclusion-and-admission-2026-10-25). The [rehearsal task pool](task-pool.md) holds 8,459 training, 2,000 dev and 2,000 test tasks plus a 4,660-task RL-only AceCode pool. [Rehearsal runs A and B](rehearsal-runs.md) (two-stage and mixed, 2B tokens each on one A40 apiece) ran under a $10 cap per run, which is also each run's binding limit (a 60,000-second Pod wall, $10.00 at the quoted rate), and a 15-minute speed check that stops a run below 90% of the reference speed. Their pre-launch code is written and reviewed: the one-pass loader, quarter-stage validation, the R2 checkpoint uploader, the run binding, the worker, the supervisor and the runbook. The binding, the shards and the shared seed-0 initialization are staged in R2. Both runs completed on 2026-10-26, billed $6.40 and $6.44 ([results](rehearsal-runs.md#measured-results-2026-10-26)). The frozen-tokenizer recount of the current review gives **198,273,119 Python tokens / 828,088,405 bytes**, predicting **12.68948B Python tokens over 64GB**, or **12.62603B after validation reserve**, conditionally—not a measured full-source count. Every model's Python target now fits inside the 59 available containers under that forecast. The 2026-10-20 forecast of 1.68502B covered only header-licensed files. Cached-source timing covers **80 isolated partition calls**, with unmeasured sources/proxies explicit. The small rehearsal remains selected. Builds beyond S are **0**; Phase 4 model runs are **2** (A and B). Post-training and full evaluation for the rehearsal each need their own owner approval.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
