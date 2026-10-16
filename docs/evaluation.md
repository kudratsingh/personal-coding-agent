# Evaluation policy

The initial external reference-model result is available but does not reproduce its published comparison score; see [reference evaluation](reference-evaluation.md). No result from a real-corpus-trained project checkpoint is available. Reports must identify checkpoint, source/dataset revision, split, decoding settings, seed, execution limits, and metric definition.

Report uncertainty across tasks or seeds as appropriate, and include negative results. Compare checkpoints on the same task set and settings; distinguish development tuning from held-out reporting.

Validate the harness against simple known answers and a reference baseline before trusting model scores. Generated code must run through the isolated runner, with hidden tests kept out of model-visible feedback.

The planned scorecard compares pretraining stages, supervised tuning, test-based optimization, and the repair loop. Scorecard rows will be added when supported by reproducible evidence.

The metric implementation includes unbiased pass@k, equal-task-weighted mean pass rate, aligned-answer accuracy, bits per byte, and paired task bootstrap intervals. Exhaustive small-subset tests and hand-computed fixtures validate the formulas. These correctness checks are not benchmark scores.

`scratchcoder eval --help` documents local checkpoint evaluation against normalized held-out task JSONL. The harness records input/checkpoint/tokenizer hashes, source revision, dirty state, seeds, decoding settings, per-task outcome hashes, equal-task metrics, and source/authorship breakdowns. Completed reports and compact scorecard JSONL stay in ignored local outputs. Missing isolation aborts reporting; context overflow is recorded as a model limit, not silently truncated.

Function tests and I/O cases use the same isolated runner. Public prompts exclude hidden tests and reference solutions. A dedicated disposable CI integration test verifies a known correct and incorrect synthetic candidate through the real runner, pool admission, metrics, and scorecard. This is not a reproduction of an external model's benchmark score.

`scratchcoder eval-choice` evaluates explicit local held-out multiple-choice JSONL with checkpoint label likelihoods, accuracy, and task-bootstrap intervals. See the [multiple-choice walkthrough](walkthroughs/multiple-choice.md) for the exact prompt, scoring policy, reference tests, and data schema. The [ARC-Easy guide](arc-easy.md) records verified complete held-out intake and the synthetic-only nano run, including every context-limit failure in its denominator.

`scratchcoder eval-validation` records per-domain held-out loss/bits per byte from an explicit local checkpoint, tokenizer, and domain-to-shard JSON map. Supply `--documents` as the maximum first-document-prefix windows per domain; optional `--sequence-length` defaults to model context. Reports identify this bounded, nonrandom diagnostic protocol, actual/available document counts, source/checkpoint/tokenizer hashes, and metrics. It is not a full-corpus estimate. Byte counts use raw token bytes, not replacement-character decoding. See [training validation](walkthroughs/training-loop.md) for the metric reference and denominator convention. No dataset is downloaded, no training occurs, and no code is executed.

The [EvalPlus workflow](evalplus.md) preserves pinned official standalone test scripts and requires a complete canonical preflight before optional checkpoint scoring. Reported runtime/limits are explicit, not claimed equivalent to official per-input resource calibration. Safe failure categories do not include raw stdout, test source or tracebacks. Benchmark defects or inadequate resource limits must be investigated rather than silently weakening tests or dropping tasks.
