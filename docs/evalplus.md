# EvalPlus release intake

The evaluation-only loader pins the official **OriginFmt** releases, not a guessed conversion of legacy differential-testing inputs. HumanEval+ supplies its own `check(candidate)` function; MBPP+ supplies inline executable tests. We preserve those test strings and append only the HumanEval invocation. We do not reimplement or replace the release's numeric, set, or task-specific comparisons with ordinary equality.

| Suite | Release revision | Gzip SHA-256 | Tasks |
| --- | --- | --- | --- |
| HumanEval+ | `200defce9e3429d28ca215b6dd061c0f7f31c18b` | `daa7661c8189924068069b0872a440b491edb60f8bdf431d5957adc88d18bae5` | 164 |
| MBPP+ | `dadf43da556a00f7bacd71cb154f2932757d9144` | `7a5aec547b134fa297ee6c45f829298aa3f58de02547b8d8a5c3d85d9cb10efc` | 378 |

Sources: [HumanEval+ release](https://github.com/evalplus/humanevalplus_release/tree/200defce9e3429d28ca215b6dd061c0f7f31c18b), [MBPP+ release](https://github.com/evalplus/mbppplus_release/tree/dadf43da556a00f7bacd71cb154f2932757d9144). Their pinned upload scripts identify OriginFmt as the format published to the dataset hub. Release tests are benchmark input, not a ready-made model, optimizer, trainer, or metric implementation.

## Boundaries

Intake verifies compressed size/hash, bounds gzip expansion, requires exact reviewed schemas and unique original IDs/counts, and retains source revision and attribution. MBPP+ IDs retain their original integer numbering so the same MBPP examples can be excluded from training pools. Plus tests and canonical solutions never enter model-visible prompts. Benchmark tasks remain evaluation-only.

The loader only parses source text; it never executes it. Legacy `base_input`/`plus_input` schemas are rejected rather than approximating their deserialization and special oracles. All actual canonical/candidate execution must use an explicitly provisioned development sandbox. The existing stdlib-only sandbox does not supply NumPy, which the release tests require. A compatible runtime and canonical validation are separate acceptance work; parsing alone does not establish suite equivalence or a benchmark result.

## Attribution and terms

The release repositories declare Apache-2.0. HumanEval+ also retains OpenAI HumanEval's MIT conditions, explicitly noted in its release license. MBPP derives from Google Research's dataset, whose [repository policy](https://github.com/google-research/google-research#readme) declares CC-BY-4.0 for datasets, separately from source-code licensing. Keep both upstream and release attribution; an aggregate dataset label does not replace these component conditions. Our manifest records the combined conditions and originators. No benchmark files, generated code, or weights are committed here, and this does not select the project's own code/weight license.

Tests in `tests/test_evalplus.py` use synthetic rows to check oracle-byte preservation, public/hidden separation, original IDs, purpose guards, bounded decoding, hash/count failures and duplicates. No external dataset is downloaded by ordinary tests.
