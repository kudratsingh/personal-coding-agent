# Documentation

Start with the [project README](../README.md), [current status](status.md), and [public roadmap](roadmap.md).

| Guide | Purpose |
| --- | --- |
| [Getting started](getting-started.md) | Navigate the starter and choose a first task. |
| [Architecture](architecture.md) | Understand the intended component boundaries. |
| [Model and vocabulary](model.md) | Exact tied-embedding parameter counts, special-token IDs and uint16 vocabulary limits. |
| [Training recipe](training-recipe.md) | What comparable small-model runs do, which practices are implemented, and which specification enables each option. |
| [Development](development.md) | Run the starter checks and understand the implementation milestone. |
| [Logging](logging.md) | Record local events, metrics, and diagnostic tracebacks. |
| [Compute](compute.md) | Estimate two-stage compute from supplied assumptions. |
| [Artifact storage](storage.md) | Upload and download R2 artifacts under a cost cap; background checkpoint uploads and retention. |
| [GPU pilot preparation](gpu-pilot-plan.md) | Understand the proposal's approval boundary and missing execution readiness; no provision/launch authorization. |
| [CPU checkpoint reference](cpu-resume-check.md) | Compare bounded authored-fixture training and exact resume without paid compute. |
| [Bounded pilot walkthrough](walkthroughs/bounded-pilot.md) | Prepare hash-bound data, understand probe/training behavior, monitor deadlines, and verify transferred artifacts. |
| [Data policy](data.md) | Record provenance, licensing, splits, and contamination controls. |
| [Real samples](real-samples.md) | Inspect actual sizes, hashes, per-filter counts and exclusion coverage. |
| [Frozen tokenizer](tokenizer-freeze.md) | Identify the immutable table and held-out compression measurements. |
| [S rehearsal data build](rehearsal-data.md) | How the S pretraining shards were built, verified and credited, with the 2026-10-25 results. |
| [Rehearsal task pool](task-pool.md) | How the capped SFT/RL pool and its dev/test holdouts were built and verified, with the 2026-10-25 counts. |
| [Rehearsal SFT data](sft-data.md) | How the hash-bound chat-v1 SFT rows, RL-candidate holdout and dev sets are built; the OASST2 language intake under its terms of record, and the 30% language share cap. |
| [Full-build plan status](full-build-plan.md) | Understand the estimate's scope, source-capacity limits and outstanding owner decision; no build authorization. |
| [Rehearsal run operations](rehearsal-runs.md) | Supervise the two concurrent D-35 rehearsal Pods: limits, R2 layout, credentials, the teardown rule and the projected fit; commands are in the [rehearsal runbook](../REHEARSAL_RUNBOOK.md). |
| [Rehearsal work-order outline](rehearsal-plan.md) | Review the plan-only Phase 4 sequence and unresolved execution prerequisites. |
| [G0 evidence](gates/g0-foundation.md) / [G1 evidence](gates/g1-data-tokenizer.md) / [G2 evidence](gates/g2-model-training.md) / [G3 evidence](gates/g3-evaluation.md) | Match every original gate/work-order pass condition to existing proof, numeric values and explicit gaps. |
| [Evaluation policy](evaluation.md) | Run local evaluators and interpret results and limitations. |
| [Reference evaluation](reference-evaluation.md) | Understand the optional pinned comparison track. |
| [EvalPlus intake](evalplus.md) | Preserve official standalone test releases and source attribution. |
| [ARC-Easy intake](arc-easy.md) | Validate the complete held-out choice suite and fixed scoring protocol. |
| [MBPP source audit](mbpp-audit.md) | Audit original task references without admitting a training pool. |
| [OpenCodeInstruct prefix pilot](opencode-pilot.md) | Audit a bounded source prefix without extrapolating or admitting training data. |
| [Sandbox](sandbox.md) | Understand requirements for executing generated programs. |
| [Walkthroughs](walkthroughs/README.md) | Document core components as they land. |
| [Decision records](adr/README.md) | Publish technical decisions safe for contributors to see. |
| [Release checklist](release.md) | Prepare reproducible artifacts and public claims. |
| [Private material](private-material.md) | Keep local operating material out of the public repository. |

The detailed owner plan is local and ignored. Public clones have all documentation needed to understand the project direction; accepted issues and pull requests define contributor work.
