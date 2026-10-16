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
