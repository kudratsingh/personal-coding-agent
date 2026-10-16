# OpenCodeInstruct prefix reference pilot

This is a bounded **evaluation-only source-component audit**, not training admission, a full task-pool build or a model benchmark. It inspects the first 200 rows in file order from one pinned shard. This ordered prefix is not random or representative; do not extrapolate retention rates to the five-million-row source. All retained audit tasks remain `eval`; no raw task export is produced.

## Provenance

Source: NVIDIA [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct/blob/8f3ba5bafe4d6e8db46082cf7ae6741bc370604d/README.md), revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`. The governing-terms body explicitly declares CC BY 4.0 and names NVIDIA; this is not inference from a license tag. Attribution: NVIDIA; Ahmad et al. (2025), *OpenCodeInstruct*. Source/seed-component rights and training suitability still require separate review before training admission. No project code/weight license is selected here.

Artifact: `data/train-00000-of-00050.parquet`, **133,476,478 bytes**, SHA-256 `342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891`. The complete shard is downloaded/hash-verified, but only its first 200 rows are normalized or executed. Its metadata declares 100,000 rows in one row group, with 362,891,070 expanded bytes. Raw data remains ignored and is not redistributed.

## Walkthrough

1. Validate the evaluation-only manifest and bounded prefix specification before downloading. Verify the complete artifact's byte count and SHA-256 before parsing it. This is source-content identity, not an authenticated legal-rights guarantee.
2. Use the locked optional `evaluation` parser with Arrow extensions disabled and bounded metadata. Require the observed nine-column schema: eight strings plus float64 `average_test_score`. The card's feature metadata lists that score as a string, but its table and actual artifact use a float; do not guess or silently coerce. Bounds are 160 MiB compressed, 512 MiB declared expanded size, eight row groups, a 4 MiB output batch, 128,000 characters per string and at most 200 sampled rows. Decoder workspace may exceed output-batch size; this is not a process memory sandbox.
3. Preserve complete supported assertion lists through the existing conservative adapter: one distinct assertion public, the rest hidden, exact duplicates collapsed. Reject unsupported setup/test forms rather than stripping them. Upstream scores/status strings never count as local reference-pass evidence.
4. Apply ARC question/public-option contamination checks, then existing code-benchmark IDs/full prompt/solution/public/hidden text and near-duplicate prompt checks. The full pinned HumanEval+/MBPP+ and ARC-Easy test suites are supplied. Missing final TACO/pool coverage is explicit; this is not final training decontamination.
5. Execute references against both partitions only inside the pinned stdlib development sandbox, under unchanged 2-second/128-MiB limits. Dependency/resource failures are profile outcomes, not necessarily mathematical incorrectness. Unavailable/unknown isolation aborts the pilot. Source programs never execute on the host.
6. Conserve all sample rows across intake and pool outcomes. Reports contain manifests, counts, retained IDs, hashes, safe execution categories and runtime limits—not code, test text, stdout or weights. Driver peak RSS excludes sandbox-container memory. Passing weak tests does not prove general correctness or capability.

Run explicitly on a disposable Docker worker:

```sh
uv run --locked --extra evaluation python tools/opencode_audit.py \
  --data-directory runs/opencode-pilot-data --download \
  --output runs/opencode-pilot/report.json
```

The manual **OpenCodeInstruct prefix reference pilot** workflow uses the reviewed standard public CPU runner. It does not authorize paid compute, source-term acceptance, training or a full build. Ordinary tests use authored fixtures only; optional parser coverage runs in dedicated CI.

## Current evidence

The [completed clean standard-CPU worker](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38898993575), revision `6a695ffc324c1a2f33441e3cc9a576e5db85b668`, accounted for all **200 prefix rows**: 35 unsupported assertion schemas; 165 entering pool checks, of which 13 were contamination/duplicates, 92 failed at least one reference partition and **60 passed both**. All retained tasks remain evaluation-only. The source prefix contains 2,000 JSON test statements; supported lists were preserved through normalization rather than stripped to manufacture passes.

There were 304 partition executions: 170 passed, 75 assertion failures, six name errors, 52 other failures and one output-limit result. Partition passes are not task passes. Failures reflect the fixed stdlib/2-second/128-MiB profile and test/schema limitations, not necessarily mathematical incorrectness. Missing final TACO/pool coverage still blocks training admission; this nonrepresentative prefix is not a full-source quality estimate or model capability result.

Protection included 542 pinned code tasks and 2,376 ARC-Easy test questions/public options. Retained-task hash: `23cb766986a1a5070d698ef08628ec1f583be1a521d5f4d9ec1627bb77d2f9cf`. Driver peak RSS was 700,588,032 bytes, excluding sandbox containers. The run did not train a model, export raw tasks, alter tests or authorize a full build.

The [unchanged-profile diagnostic repeat](https://github.com/kudratsingh/personal-coding-agent/actions/runs/38900425418), clean revision `7e2f0c43b18d165cba0b3dfb83c20a5d6ca5f131`, retained exactly the same 60 IDs, task hash and pool accounting. Expanded safe exception labels split the previous 52 `other_failure` partition results into 24 `ModuleNotFoundError`, four `IndexError`, four `ValueError` and 20 still unclassified. All other execution counts were unchanged. These are heuristic debugging categories from untrusted output, not authenticated causes or proof of mathematical incorrectness. No dependency, input, test, runtime or resource limit changed to improve retention.
