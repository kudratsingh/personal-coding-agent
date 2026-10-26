# Phase 4 rehearsal plan

**Progress (2026-10-26):** P4-2 is done, and so is the pretraining in P4-3 to P4-5. Their ARC-Easy benchmark has not been measured yet.
- **Data:** the S build, the task pool and the holdouts are admitted.
- **Pretraining:** runs A and B [ran on two A40s](rehearsal-runs.md#measured-results-2026-10-26) under their own approval.
- **Comparison:** the owner accepted it. On Python, two-stage was lower by about 1.0%, just over the pre-registered 1% threshold, which the owner reads as a tie within noise. Mixed was 6.1% better on English, and two-stage lost 0.079 bits per byte of English in stage 2. The main model therefore pretrains mixed, and the two-stage path is kept as an experiment.
- **What changes for the remaining work orders:** P4-6 to P4-9 now continue from run B's mixed checkpoint only. P4-6's matched SFT on both arms is dropped, and so is the comparison's second co-primary endpoint, dev pass rate after SFT.
- **Approval:** each remaining work order still needs its own approval before any spend.

The rehearsal outline has been expanded into **12 owner-only work orders**, each with inputs/dependencies, a narrow scope, explicit “done when” checks and stop conditions. P4-2 to P4-5 ran under their own approvals; each remaining work order still needs its own. There is no new implementation, data build, model training, machine, GPU pilot or paid resource from this planning work.

| Work order | Planned outcome |
| --- | --- |
| P4-0 | Resolve readiness/owner decisions and preregister the comparison before results. |
| P4-1 | Specify missing S-model, SFT, RL, chat-evaluation, control and demo interfaces with required future reference tests/walkthroughs. |
| P4-2 | Approve and materialize hash-bound rehearsal data, true holdouts and identical token tickets. |
| P4-3 | Pretrain the small model on English; save and evaluate its milestone. |
| P4-4 | Continue on Python with unseen English, preserving parent weights but resetting optimizer/schedule. |
| P4-5 | Train the mixed comparator from the identical initialization and exact same token multiset. |
| P4-6 | Apply short code-and-language SFT to run B's mixed checkpoint (matched SFT on both arms dropped, 2026-10-26). |
| P4-7 | Run short code-only RL on the rehearsal arm under separately approved limits. |
| P4-8 | Run a compute-matched more-SFT control before claiming RL helps beyond supervision. |
| P4-9 | Fill the fixed full scorecard, including nulls/regressions; the order comparison is recorded on pretraining validation only. |
| P4-10 | Demonstrate and record the repair loop locally; no hosted execution service. |
| P4-11 | Audit G4 evidence, actual costs and durable artifacts, then obtain the owner's next decision. |

The two-stage/mixed comparison controlled initialization, model/tokenizer, exact token multiplicities and total training budget. It was explicitly a **two-stage recipe versus mixed recipe** comparison: the optimizer/schedule reset differed too, so it does not establish order-only causality. It was read against the predeclared practical BPB threshold; one training seed is not a multi-seed conclusion. The matched-SFT arm and its paired task intervals were dropped when post-training moved to run B only. No extra control or seed is implicitly authorized.

The [frozen tokenizer](tokenizer-freeze.md) is reused without retraining. Base scoring keeps the [protocol of record](reference-evaluation.md); chat-trained scoring needs its own tested/versioned format. HumanEval+ remains **163 tasks with task 32 explicitly excluded**, not the full 164-task suite.

Current blockers are in [G0](gates/g0-foundation.md), [G1](gates/g1-data-tokenizer.md), [G2](gates/g2-model-training.md) and [G3](gates/g3-evaluation.md), and the [full-build estimate status](full-build-plan.md). Runnable S pilot and rehearsal configurations exist; SFT/RL/agent/demo work is not implemented yet. Scope/full-build decisions, source capacity/component rights, final holdouts, measured hardware and production isolation must be resolved. Every future build/run still needs its specific approval, including local full builds and any paid service. The repository is already public; this plan does not select a code/weight license or authorize main training.

G4 can close only after an actual documented command sequence runs the full pipeline, per-stage costs are measured/re-priced, complete scorecards and the comparison record exist, the local demo is recorded, and required owner decisions are recorded. Work-order checkboxes are currently unchecked. The GPU pilot completed on 2026-10-24. No work continues automatically after this plan.
