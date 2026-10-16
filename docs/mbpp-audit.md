# MBPP source reference audit

This is an **evaluation-only audit of one planned task source**, not a model benchmark or an approved training-pool build. It returns safe counts, identities and retained audit IDs, never a training-task export. All retained tasks remain in the `eval` partition. Final TACO/source coverage, complete held-out pools and full-build approval are separate gates.

## Provenance and terms

Source: [Google Research MBPP](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp), revision `f82046ba5aabbbb427dbfd38a254d26bff08b533`, file `mbpp/mbpp.jsonl`. Expected size: **563,743 bytes**, SHA-256 `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`, **974 unique original IDs**.

The [pinned repository policy](https://github.com/google-research/google-research/blob/f82046ba5aabbbb427dbfd38a254d26bff08b533/README.md) explicitly licenses datasets under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), separately from source files under Apache 2.0. This is primary-source review, not inference from a hub tag. Attribution: Google Research; Austin et al. (2021), *Program Synthesis with Large Language Models*. Preserve source credit/license links and indicate adaptations if redistributing material; raw/converted tasks are not redistributed here. This does not choose a project code/weight license.

The source README identifies IDs 11–510 as its original evaluation partition, 1–10 as prompting examples, and 511–1000 as training/prompting material. The audit records those original ID-range counts but places no task into a new training partition. It makes no official MBPP score claim.

## Walkthrough and boundaries

1. Explicit `--download` fetches only pinned size/hash-verified files. The MBPP reader bounds file/count sizes and requires the reviewed six-field schema, unique integer IDs, primary assertion arrays, optional challenge contents, and source text. No dataset or reference program executes during parsing.
2. Every actual MBPP+ ID is excluded **before normalization or sandbox execution**, regardless of its original MBPP partition. The protected IDs must exist in the complete original source.
3. Remaining rows pass through the existing conservative assertion/import normalizer. Both primary and challenge arrays are supplied; exact duplicate assertions collapse. One distinct assertion is public and the rest hidden. Unsupported setup statements or test forms are rejected and counted, not silently run on the host or stripped to manufacture passing references. MBPP remains weakly tested.
4. ARC-Easy questions and public option text receive the same normalized n-gram protection. Standalone answer labels are not indexed. Existing pool checks protect complete HumanEval+/MBPP+ prompt/solution/public/hidden text, canonical IDs and near-duplicate prompts. Missing final TACO/pool holdouts prevent any claim of complete training admission.
5. References must pass both partitions in separate invocations of the existing development sandbox. The stdlib-only runtime uses its pinned immutable image, network/mount/capability restrictions, 2-second/128-MiB defaults and bounded input/output. Failed, resource-limited or dependency-incompatible references are not retained under this profile; that is not necessarily proof of mathematical incorrectness. Missing/unknown isolation aborts the audit.
6. Source-stage and pool-stage accounting must conserve every row. Reports retain manifests, benchmark identities, original partition counts, retained audit IDs, safe execution categories and runtime limits. Driver peak RSS excludes container memory. Raw programs, tests, stdout and weights are excluded from console reports; local artifacts stay ignored.

Run explicitly on a disposable Docker worker:

```sh
uv run --locked --extra evaluation python tools/mbpp_audit.py \
  --data-directory runs/mbpp-audit-data --download \
  --output runs/mbpp-audit/report.json
```

The manual **MBPP source reference audit** workflow uses a standard public CPU runner. No paid compute, real-corpus training, source-term acceptance or full production build is authorized by it. Fixture tests cover pins/schema/hash/count failures, challenge retention, pre-execution ID/choice exclusion, safe accounting and unavailable-isolation failure.

## Current evidence

The [completed clean CPU audit](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38891611877), at revision `4c535a2b7f862bc88d0af2983140e35f07a55f39`, accounted for all **974 source rows**: 378 MBPP+ IDs excluded before execution, three unsupported assertion/setup schemas, and 593 candidates entering pool checks. Those checks rejected 74 as contamination or duplicates and retained **519 evaluation-only tasks**. All 1,038 public/hidden reference executions passed under the unchanged stdlib sandbox profile. The source contains 2,922 primary and 16 challenge assertions; exact duplicate assertions are normalized as described above.

Protection included all 542 pinned HumanEval+/MBPP+ tasks and 2,376 ARC-Easy test questions/public options. Retained-task hash: `7ea7e6ce22b8b993525215c54bb59391a2e060518921b6f3f394f3275fbba846`. Driver peak RSS was 547,176,448 bytes, excluding sandbox container memory. This establishes reference execution under this profile, not strong tests, model capability, complete holdout coverage or permission to train. No task was admitted to training or exported as a training pool.
