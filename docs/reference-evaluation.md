# Reference-model evaluation

The optional `baseline` dependency extra is used only by `tools/baselines/`; it is not a core training dependency. The HumanEval comparison pins the benchmark revision/content hash and the model revision, loads safetensor weights without remote model code, and executes every candidate through the isolated runner.

The tool uses greedy completion with an explicit token budget and stop strings, records all settings/hashes/outcomes, and compares the resulting rate with the model card's stated value. The card does not fully specify decoding settings, so a mismatch must be investigated rather than described as a reproduced score.

The manually dispatched Reference benchmark workflow uses a standard public-repository CPU runner. It does not provision rented GPUs or upload model caches, weights, or generated programs. Public benchmark provenance and outcome-status records are retained in its console log. Ordinary CI tests use synthetic rows and do not download the reference model.

## Observed comparison

The [initial recorded run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38780757269) completed all 164 tasks with greedy decoding and a 256-token completion budget: **1/164 passed (0.61%)**, versus the card's stated 7.84%. This is not a reproduced score. The card marks its score unverified and does not specify the full decoding protocol. That uncertainty is a limitation, not an established explanation for the gap.

Subsequent runs first require every canonical solution to pass the same hidden tests through isolation. `--canonical-only` runs this check without loading model weights. A failed canonical check stops model benchmarking. Model reports additionally retain allowlisted failure categories, never raw exception text, generated programs, or hidden tests. Token-budget comparisons must be recorded separately; do not tune on the benchmark and present the best result as an independent validation.

The [canonical-only validation run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38870605573) passed **164/164**. This establishes correct handling of these known answers under this runner, not published-score reproduction or general sandbox security.

The [completed diagnostic rerun](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38870858985) preserved the original greedy/256-token settings and again passed **1/164**, after all 164 canonical answers passed its preflight. Reported model-output categories were 147 assertion failures, 6 syntax errors, 6 type errors, 2 name errors, 2 other failures, and 1 pass; no infrastructure failure was reported. These categories establish repeatability and help narrow investigation, but do not prove the cause of the gap to the card's score. The reference reproduction gate remains open.

## Observational generation diagnostics

The [completed observational run](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38887684383) again passed all 164 canonical checks and scored 1/164. All 164 candidate hashes matched the preceding diagnostic run. It additionally records generated token counts, whether the budget was reached, EOS presence, the first matched existing stop-string index, retained/decoded character counts, and syntactic completeness of the unchanged candidate. AST parsing inspects text only; generated programs still execute exclusively through the sandbox. No source text, token IDs, hidden tests or raw syntax-error messages enter these diagnostic records. Both per-task observations and aggregate counts are retained. Fixture tests verify existing truncation behavior, EOS-on-the-last-budget-token accounting, syntax checks and source-text exclusion.

Of 164 completions, 128 reached the budget without EOS, and 36 contained EOS. Existing stop strings matched 126 outputs. There were 87 empty retained completions: 57 stopped at the existing definition marker, while 30 had no stop-string match and contained EOS. Only eight outputs both exhausted the budget without EOS and had no stop-string match. These overlapping observations must not be summed as disjoint failure categories; parsing or a stop match does not establish semantic correctness.

These observations did not alter prompts, model/tokenizer revisions, greedy decoding, budget, stop strings or benchmark tests. A complete canonical preflight is still required. Truncation/EOS counts by themselves cannot establish the cause of the score discrepancy; the reference reproduction gate remains open.

The [pinned model card](https://huggingface.co/bigcode/tiny_starcoder_py/blob/8547527bef0bc927268c1653cce6948c5c242dd1/README.md) does not supply a complete evaluation protocol. Arithmetically, 7.84% cannot be a single deterministic integer pass count over 164 tasks rounded to two decimals: 12/164 is 7.32%, and 13/164 is 7.93%. This suggests an averaged or otherwise different protocol, but is an inference—not an explanation or reproduction of the reported score. No benchmark-driven search for a better-looking decoding setting is authorized by this diagnostic work.
