# Phase 4 rehearsal — plan only

The rehearsal outline has been expanded into **12 owner-only work orders**, each with inputs/dependencies, a narrow scope, explicit “done when” checks and stop conditions. **None is authorized to execute.** There is no new implementation, data build, model training, machine, GPU pilot or paid resource from this planning work.

| Work order | Planned outcome |
| --- | --- |
| P4-0 | Resolve readiness/owner decisions and preregister the comparison before results. |
| P4-1 | Specify missing S-model, SFT, RL, chat-evaluation, control and demo interfaces with required future reference tests/walkthroughs. |
| P4-2 | Approve and materialize hash-bound rehearsal data, true holdouts and identical token tickets. |
| P4-3 | Pretrain the small model on English; save and evaluate its milestone. |
| P4-4 | Continue on Python with unseen English, preserving parent weights but resetting optimizer/schedule. |
| P4-5 | Train the mixed comparator from the identical initialization and exact same token multiset. |
| P4-6 | Apply matched short code-and-language SFT to both pretrained arms. |
| P4-7 | Run short code-only RL on the rehearsal arm under separately approved limits. |
| P4-8 | Run a compute-matched more-SFT control before claiming RL helps beyond supervision. |
| P4-9 | Fill the fixed full scorecard and paired order-comparison analysis, including nulls/regressions. |
| P4-10 | Demonstrate and record the repair loop locally; no hosted execution service. |
| P4-11 | Audit G4 evidence, actual costs and durable artifacts, then obtain the owner's next decision. |

The two-stage/mixed comparison controls initialization, model/tokenizer, exact token multiplicities, total training budget and matched SFT. It is explicitly a **two-stage recipe versus mixed recipe** comparison: the optimizer/schedule reset differs too, so it does not establish order-only causality. The planned interpretation retains paired task confidence intervals and a predeclared practical BPB threshold; one training seed is not a multi-seed conclusion. No extra control or seed is implicitly authorized.

The [frozen tokenizer](tokenizer-freeze.md) is reused without retraining. Base scoring keeps the [protocol of record](reference-evaluation.md); chat-trained scoring needs its own tested/versioned format. HumanEval+ remains **163 tasks with task 32 explicitly excluded**, not the full 164-task suite.

Current blockers are in [G0](gates/g0-foundation.md), [G1](gates/g1-data-tokenizer.md), [G2](gates/g2-model-training.md) and [G3](gates/g3-evaluation.md), and the [full-build estimate status](full-build-plan.md). Only nano/smoke configs exist today; SFT/RL/agent/demo work is not represented as already implemented. Scope/full-build decisions, source capacity/component rights, final holdouts, measured hardware and production isolation must be resolved. Every future build/run still needs its specific approval, including local full builds and any paid service. The repository is already public; this plan does not select a code/weight license or authorize main training.

G4 can close only after an actual documented command sequence runs the full pipeline, per-stage costs are measured/re-priced, complete scorecards and the comparison record exist, the local demo is recorded, and required owner decisions are recorded. Work-order checkboxes are currently unchecked. GPU pilot preparation remains untouched; no work continues automatically after this planning packet.
