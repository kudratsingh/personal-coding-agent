# EvalPlus release intake

The evaluation-only loader pins the official **OriginFmt** releases, not a guessed conversion of legacy differential-testing inputs. HumanEval+ supplies its own `check(candidate)` function; MBPP+ supplies inline executable tests. We preserve those test strings and append only the HumanEval invocation. We do not reimplement or replace the release's numeric, set, or task-specific comparisons with ordinary equality.

| Suite | Release revision | Gzip SHA-256 | Tasks |
| --- | --- | --- | --- |
| HumanEval+ | `200defce9e3429d28ca215b6dd061c0f7f31c18b` | `daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5` | 164 |
| MBPP+ | `dadf43da556a00f7bacd71cb154f2932757d9144` | `7a5aec547b134fa297ee6c45f829298aa3f58de02547b8d8a5c3d85d9cb10efc` | 378 |

Sources: [HumanEval+ release](https://github.com/evalplus/humanevalplus_release/tree/200defce9e3429d28ca215b6dd061c0f7f31c18b), [MBPP+ release](https://github.com/evalplus/mbppplus_release/tree/dadf43da556a00f7bacd71cb154f2932757d9144). Their pinned upload scripts identify OriginFmt as the format published to the dataset hub. Release tests are benchmark input, not a ready-made model, optimizer, trainer, or metric implementation.

## Boundaries

Intake verifies compressed size/hash, bounds gzip expansion, requires exact reviewed schemas and unique original IDs/counts, and retains source revision and attribution. MBPP+ IDs retain their original integer numbering so the same MBPP examples can be excluded from training pools. Plus tests and canonical solutions never enter model-visible prompts. Benchmark tasks remain evaluation-only.

The loader only parses source text; it never executes it. Legacy `base_input`/`plus_input` schemas are rejected rather than approximating their deserialization and special oracles. All actual canonical/candidate execution must use the explicit dependency sandbox. The stdlib-only default does not supply NumPy. The [dependency runtime](adr/0006-evaluation-runtime.md) passed its separate CI checks; complete real-suite canonical validation remains required. Parsing alone does not establish suite equivalence or a benchmark result.

## Attribution and terms

The release repositories declare Apache-2.0. HumanEval+ also retains OpenAI HumanEval's MIT conditions, explicitly noted in its release license. MBPP derives from Google Research's dataset, whose [repository policy](https://github.com/google-research/google-research#readme) declares CC-BY-4.0 for datasets, separately from source-code licensing. Keep both upstream and release attribution; an aggregate dataset label does not replace these component conditions. Our manifest records the combined conditions and originators. No benchmark files, generated code, or weights are committed here, and this does not select the project's own code/weight license.

Tests in `tests/test_evalplus.py` use synthetic rows to check oracle-byte preservation, public/hidden separation, original IDs, purpose guards, bounded decoding, hash/count failures and duplicates. No external dataset is downloaded by ordinary tests.

## Explicit validation command

`tools/evalplus_check.py` takes a suite, a local release path, an immutable locally built NumPy image ID, and a new output directory. Only `--download` opts into fetching that suite's pinned bytes. It requires every canonical solution to pass before scoring an optional own-project `--checkpoint` with its `--tokenizer`. A failed canonical run preserves its report but blocks model scoring. Reports retain source, runtime ID, limits, prompt policy and hashes; raw programs/tests/weights are not printed or uploaded.

The manual **EvalPlus canonical validation** workflow builds the dependency runtime on a standard disposable public CPU runner and performs this preflight. Optional nano mode trains only on repository-authored synthetic web/code fixtures before evaluating; benchmark text is never training/tokenizer input. These nano numbers test integration, not useful coding capability.

The benchmark command declares one global 60-second budget per task, 4 GiB memory by default (explicitly overridable within the bound), 16 processes, bounded output, and a 2 MiB code/test input cap. The general sandbox default remains 128 MiB. Initial 1 GiB preflight evidence is retained separately; MBPP/255's large materialized candidate/reference outputs require a larger bound. This is not EvalPlus's original per-input time calibration/resource policy, and reports explicitly mark that difference. Do not call these runs a reproduction of official leaderboard settings. Missing isolation aborts; oversized model prompts count as context limits rather than being silently truncated.

## Observed validation runs

| Run | Suite | Canonical result | Project scoring |
| --- | --- | --- | --- |
| [Initial HumanEval+](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38879134729) | 164 tasks, 1 GiB cap | 163/164; task 32 failed | Blocked |
| [Initial MBPP+](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38879138709) | 378 tasks, 1 GiB cap | 377/378; task 255 failed | Blocked |
| [Diagnostic HumanEval+](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38880458614) | 164 tasks, 4 GiB cap, unchanged tests | 163/164; task 32 reports `TypeError` | Blocked |
| [MBPP+ rerun](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38880455583) | 378 tasks, 4 GiB cap, unchanged tests | 378/378 | Synthetic-trained nano: 0/378 |
| [Opt-in task-32 correction](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38885124795) | 164 tasks, 4 GiB cap, `find-zero-residual-v1` | 163/164; task 32 reports `AssertionError` | Blocked |

HumanEval/32's standalone script attempts to unpack the scalar root returned by its candidate into the polynomial helper. The [pinned official framework](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py) instead uses an absolute polynomial-residual check. An explicit reviewed correction is available as described below; the original failed runs remain unchanged.

This standalone-release defect is separate from the [original HumanEval reference-model discrepancy](reference-evaluation.md), which used OpenAI's original tests, not this plus release. It does not explain that earlier low model score.

MBPP/255 contains an input with 1,663,740 length-77 combinations materialized for both the candidate and inline reference. The initial 1 GiB record is retained; a separately recorded 4 GiB rerun uses the same tests on the same standard free CPU runner. These are canonical validation results, not model capability scores. Complete safe reports are preserved locally; the raw workflow log archive retains large report lines that some CLI log views omit.

The completed MBPP+ rerun passed all 378 canonical solutions and then evaluated the synthetic-only nano checkpoint. Its result was **0/378**: 156 complete prompts exceeded the 256-token model context and counted as failures; the other 222 generated candidates failed the tests. This demonstrates the recorded integration path, not useful coding capability or real-corpus training. No prompts were shortened, tasks removed, or benchmark material used to fit the tokenizer/model.

## Opt-in task-32 correction walkthrough

[ADR 0007](adr/0007-find-zero-oracle-correction.md) defines `find-zero-residual-v1`. The default loader and command still use unmodified OriginFmt tests. Only the explicit `--correct-find-zero-oracle` flag (or matching manual-workflow boolean, default false) enables the correction, and only for the exact pinned HumanEval+ release.

The correction hashes task 32's normalized hidden script, including the appended `check(find_zero)` invocation. It requires original SHA-256 `41ab5c51c15d24d07317ba9153761d568c3c249234f3442223c3f4511b88fd01` and exactly one reviewed expression. It changes scalar argument handling and checks the absolute residual, preserving every input and the original `0.0001` threshold. A changed release, script, task provenance, absent/duplicate task, or wrong suite is rejected before execution. No canonical output is substituted as an expected root.

Both canonical and checkpoint reports identify the original release hash, policy/version, patched task ID, original/effective hidden-script hashes, effective individual task hash, and reference evaluator revision. The aggregate report task hash changes too. No source text is included in those correction records. The workflow still requires a complete canonical pass before scoring the optional synthetic-only nano checkpoint.

The first full corrected run still passed only **163/164**: task 32 reports `AssertionError`, replacing its original `TypeError`. Its effective hidden-script SHA-256 is `e060eeb751a1acbd6ca57565aac40c566fc8df64949ca81918cc9b4fd975a2f7`. The canonical report was preserved and model scoring remained blocked. No additional oracle change, tolerance relaxation, canonical replacement, or passing-suite claim follows from this result.

## Separate numerical diagnosis

`tools/find_zero_diagnose.py` and the manual **Find-zero canonical diagnosis** workflow are diagnostic-only, not benchmark scoring. They require the same pinned original script/release guards, retain all original literal inputs, and execute the canonical program only inside the dependency sandbox. No host fallback exists. Reports identify canonical/script/input hashes, static domain counts, sandbox profile, aggregate residual/exception counts and up to eight bounded numeric examples; raw inputs, source, tests and stdout are not copied into the report. Missing/duplicate markers, invalid diagnostic fields or incomplete case accounting are rejected. A successful diagnostic process does not mean that the canonical solution passed its tests.

Static parsing found 886 input/result pairs, with even coefficient counts but 62 zero highest coefficients and 46 zero linear coefficients. The canonical program's docstring promises a nonzero highest coefficient. These observations alone do not explain the residual failure.

The [completed isolated diagnostic](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38887075752) observed all **886** cases: **879 met the residual criterion, seven did not, and no exceptions occurred**. Failing zero-based indices were 217, 275, 283, 497, 714, 748 and 792. All seven had nonzero highest and linear coefficients, so the static domain violations do not explain these failures. At least four of those recorded cases also fail when the release's supplied expected root is evaluated by the same residual helper.

The first failure, case 217, returned a root of `17.3124550475086` with residual `-0.0023150274059275944`; the supplied expected root produced that same residual, beyond the unchanged `0.0001` threshold. This identifies numerical/reference inconsistency under the recorded runtime, not a justification to relax a tolerance, substitute a reference, or filter inputs. Full HumanEval+ scoring remains blocked; further remedial choices require separate review. The complete safe diagnostic report is retained locally and is explicitly marked `benchmark_acceptance=false`.
