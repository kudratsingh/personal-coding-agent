# Pilot launch readiness: single A40 bounded training pilot

Audit date: 2026-10-22. Audited revision: branch `pilot-readiness-audit` on top of `940530d` (PR #80). Environment used for verification: a CPU-only Linux container (4 vCPU, 15 GB RAM, Python 3.12.13, torch 2.14.1 CPU build). No Runpod Pod, GPU, volume or other paid resource was created, started or quoted by this audit; Runpod spending by this work is $0.

## 1. Final readiness verdict

**READY WITH BLOCKERS.** The engineering, tests, safeguards and runbook are complete and verified on CPU. Launch is blocked until the owner resolves the conditions in section 6, the most important being that Runpod has no provider-enforced Pod lifetime: termination within 90 minutes is layered best-effort (in-Pod timer, Mac watchdog, worker deadline), not a hard guarantee, and the owner must accept that residual risk explicitly in the readiness receipt. All A40 numbers remain unmeasured.

## 2. Verified implementation components

Verified by reading the code and running the tests or tools on CPU (section 5 has the counts):

- Model: `GPT(ModelConfig(65536, 768, 12, 12, 1024))` has exactly 135,285,504 parameters, equal to `expected_parameters`; weights are initialised with standard deviation 0.02, the embedding is tied to the output projection, attention is causal (perturbing a later token leaves earlier logits unchanged), rotary positions are pairwise rotations, RMSNorm and softmax run in fp32 under bf16 autocast, master weights stay fp32.
- Optimizer: hand-written AdamW matches `torch.optim.AdamW` within float rounding; decay mask excludes the embedding and norm gains; warmup plus cosine schedule resumes at step 100 on the identical curve; gradient clipping and nonfinite detection work.
- Data: the loader is deterministic per seed, packs documents with BOS tokens inside sequences (no padding), and repeats the same seeded permutation only if a second pass starts; shards are uint16 with BOS 256 and maximum ID 65,535.
- Training and checkpoints: exact interrupted/resumed equivalence (tests, `tools/cpu_resume_check.py` 4 cases, `tools/pilot_preflight.py` on CPU); atomic checkpoint writes; weights-only loading of the saved structures under torch 2.14.
- Orchestration: probes, three fair screens, winner selection with the incomplete-screen refusal, continuation, final save and resume test all run end to end in the CPU smoke (`tools/gpu_pilot.py smoke`, 3.8 s).
- Watchdog, supervisor and transfer: offline tests cover identity matching, delete confirmation (204 then 404), cost and wall limits, heartbeat validation, staged and hash-verified pulls, private-repository archive with read-back.
- Publication hygiene: `tools/check_public_tree.py` finds no tracked ignored paths; a grep for `rpa_`, `AKIA`, `ghp_`, private keys and bearer tokens over the tracked tree finds nothing; `RUNPOD_API_KEY` is read only from the process environment and never printed.

Prior claims checked against repository evidence: the "826 passing tests" figure is not reproducible here (this environment collects 832 and runs 791 passed / 41 skipped at the base revision; the skips need Docker, the NumPy image or the evaluation extra, which the owner's Mac may have). "Green CI" cannot be observed from this container; the workflows are declared on standard runners only. "Dataset hashes bound" is implemented (`tools/pilot_data.py` binding with SHA-256 of source, tokenizer and shards) but the real binding file is private and not in this clone, so the 66.03M-token figure is taken from the owner's statement. "Untrained checkpoint 1.512 GiB" is confirmed: a checkpoint of the untrained S model written by `Trainer.checkpoint` on CPU measures 1,623,502,465 bytes = 1.5120 GiB (fp32 weights 541,142,016 bytes plus two fp32 moment sets). "Zero GPU spending" is consistent with every record in the repository. "Best-effort watchdog" and "unresolved strict termination" are confirmed by the code and by Runpod's documentation.

## 3. Issues discovered

Confirmed by the parallel audit and reproduced in code or tests before fixing:

1. The batch-probe window was measured from the billing origin, so more than five minutes of image pull, bootstrap and preflight aborted the whole pilot before any training.
2. Probes ran without gradient accumulation, so their peak memory understated the accumulated training phases, where fp32 gradients stay resident during forward passes.
3. The watchdog treated a 404 on its very first poll (wrong Pod ID, wrong key scope, wrong base URL) as a completed shutdown and exited 0 while the Pod kept billing.
4. The 600-second idle limit applied during artifact transfer, so a 1.5 GiB pull over a home connection would get the Pod deleted mid-transfer; the supervisor used a 300-second transfer timeout.
5. With no heartbeat yet, the idle clock started at Pod creation, so a slow bootstrap would get the Pod deleted before the worker started.
6. The watchdog and supervisor polled every 2 to 5 seconds against a per-minute rate-limited API.
7. Checkpoint bytes were hashed but never compared on resume; a flipped byte inside a tensor loaded silently.
8. The final checkpoint was exported twice, doubling the transfer inside the idle window.
9. The continuation saved over the winner's step-100 checkpoint and deleted its export, destroying the fair-comparison artifact; the resume test rewrote the continuation's run record so its provenance pointed at itself; screen-arm records stayed `running`.
10. Only `torch.cuda.OutOfMemoryError` counted as a probe OOM; cuBLAS allocation failures raised as `RuntimeError` would have failed the pilot.
11. The previous export bundle was deleted the moment a new one appeared, racing an in-progress pull; exports required hard-link support.
12. The self-destruct deadline recorded in the receipt was not folded into the worker's wall limit.
13. After the continuation resumed, the periodic checkpoint cadence was measured from the screen save, forcing an immediate extra 1.5 GiB save.
14. Train/validation separation was ID-level only; verbatim duplicate text under another ID passed.
15. Cross-entropy kept three fp32 copies of the `[B, T, 65536]` logits for backward (12 bytes per logit).
16. No checked-in A40 specification existed; the defaults would have allowed fp32, 2 validation documents and a 1-step warmup; nothing verified the validation document count against the binding.
17. No in-Pod termination existed, no tool captured the Pod identity in the shape the watchdog requires, and the readiness receipt had to be typed by hand.
18. The sanitizer did not recognise Runpod key prefixes; the readiness receipt was embedded in records unsanitised.

Risks noted but not changed (documented limitations): attention materialises `[B, H, T, T]` scores in bf16 with an fp32 softmax (about 6 bytes per score saved per layer); gradient clipping performs one device sync per parameter tensor; data loading is synchronous Python; the schedule applies the peak learning rate on two consecutive steps; the rising-loss guard counts consecutive increases (minimum now 5, specification uses 25); the loader never reshuffles across passes (not reached in this pilot).

## 4. Fixes implemented

- `src/scratchcoder/train/pilot.py`: `finalization_reserve_seconds` in `Limits` with a `training_deadline` rule; `choose_microbatch` accepts `skipped`/`incomplete` rows, applies a reserved-memory headroom budget and a rising-steps floor; `PilotStop` derives from the new `TrainingStop`.
- `tools/gpu_pilot.py`: probe window counted from the first probe with `incomplete`/`skipped` statuses; probes use two accumulated microbatches; headroom at 90% of device memory; SHA-256 sidecars written at every save and verified before every resume; continuation saves under `<arm>-continued` and keeps the step-100 checkpoint; resume test in a fresh trainer; single final export; previous bundle retained; copy fallback when hard links fail; broader CUDA OOM detection; token accounting, validation identity, data exposure and deadline fields in the report; readiness gates renamed to `termination_guarantee` plus owner acceptance, `pod_self_destruct_armed`, `self_destruct_deadline_unix`; run path requires bf16, warmup inside the screen, validation on every bound document, a 600-second reserve, and folds the armed self-destruct deadline into the wall limit.
- `src/scratchcoder/train/trainer.py`: `resume(expected_sha256=...)`, `finish()`, directory fsync after the checkpoint rename, `TrainingStop` logged as a stop rather than a failure.
- `src/scratchcoder/optim/adamw.py`: cross-entropy formed from the log-softmax primitive (one fp32 saved copy, 4 bytes per logit) with an equivalence test against the explicit formulation and a saved-memory bound.
- `tools/runpod_watchdog.py`: a 404 before the identity was ever observed alive is `unknown`, observation persisted across restarts, exit 3 if never observed; 429 back-off; 15-second default poll; `max_transfer_seconds` window driven only by Mac-observed progress; `idle_baseline_unix`; generic `request_path`.
- `tools/pilot_supervisor.py`: `--transfer-timeout` (default 1,800 s), `--poll-seconds`, transfer-progress stamps from growing staged bytes, observation state, `-continued` bundle names.
- `tools/pilot_data.py` and new `tools/pilot_data_audit.py`: token-level audit (exact duplicate documents, validation window containment anywhere in the training stream) integrated into `verify_binding`.
- New `tools/pod_self_destruct.py` (standard-library in-Pod timer), `tools/pilot_pod_bootstrap.sh`, `tools/runpod_pod_identity.py` (capture, status, billing), `tools/pilot_readiness_receipt.py`, `tools/pilot_feasibility.py`, `configs/pilot-a40-s.json`, `configs/runpod-pod-a40-s.json`.
- `src/scratchcoder/common/logging.py`: `rpa_` keys redacted.
- Documentation: `docs/walkthroughs/bounded-pilot.md`, `docs/walkthroughs/optimizer.md`, `docs/gpu-pilot-plan.md`, `docs/status.md`, `.env.example`, this file and `PILOT_RUNBOOK.md`.

## 5. Test counts and results

Measured in this container with the CPU torch build (the locked `uv run --locked` path cannot download the CPU wheel here, so the identical commands ran against the same locked dependency set with torch 2.14.1 from PyPI):

| Check | Result |
| --- | --- |
| Base revision `940530d` full suite | 791 passed, 41 skipped (Docker, NumPy image, evaluation extra), 9 subtests |
| Audited branch full suite | 872 passed, 41 skipped, 9 subtests (871 in the worktree run with the console-script test deselected; that test passes under `uv run`) |
| `ruff check`, `ruff format --check` | clean |
| `mypy --strict` (src, tests, tools) | clean |
| `tools.check_workflows`, `tools.check_public_tree`, `tools.check_docs` | pass |
| `tools/gpu_pilot.py smoke` | completed, 3 probes, 3 arms, winner continued to step 5, resume test passed |
| `tools/pilot_preflight.py --device cpu` | completed (precision and exact resume) |
| `tools/cpu_resume_check.py --size tiny --seeds 0 7 --steps 4` | 4 exact cases |
| Untrained S checkpoint on CPU | 1,623,502,465 bytes = 1.5120 GiB |

One existing test, `tests/test_cli.py::test_smoke_runs_complete_synthetic_pipeline[entrypoint1]`, needs the `scratchcoder` console script on `PATH`; it passes under `uv run` and was deselected only in the worktree run where the script was not on `PATH`.

New regression tests added: probe window semantics (two tests), finalization reserve, tampered checkpoint rejection, token accounting, single final export and bundle retention, CUDA library OOM classification, rising-guard floor, readiness gates, `Limits.training_deadline`, skipped/incomplete/headroom selection, specification pinning (two tests), resume hash rejection, controlled stop logging, sanitizer, supervisor transfer-progress stamping and bundle names, watchdog observation gating, state file, transfer window (six tests), idle baseline (two tests), self-destruct timer (twelve tests), Pod identity and billing capture (five tests), readiness receipt (three tests), leakage audit (five tests), feasibility calculator (29 tests), cross-entropy equivalence (two parametrised cases).

## 6. Remaining blockers

1. **Termination is not provider-enforced.** Runpod documents no Pod lifetime or TTL setting; the only account-level backstops are the $0-balance stop and the default $80/hour spend limit. The layered safeguards (in-Pod timer using the injected Pod-scoped key, Mac watchdog, worker deadlines) all depend on the Runpod control plane being reachable. The owner must set `termination_guarantee` to `redundant_best_effort` and `owner_accepts_best_effort_termination` to true in the receipt, or the run command refuses to start. Classified BLOCKING per the task definition.
2. **The Pod-scoped key's permission to delete its own Pod is undocumented.** It must be verified on the real Pod before relying on the timer; the bootstrap arms it anyway, and the Mac watchdog stands independently.
3. **A40 preflight, probe table, throughput and VRAM are unmeasured** (section 7). The specification fixes the plan; the probes decide the microbatch.
4. **Owner authorization and quote confirmation** are required for step 5 of the runbook; the specification's $0.59/hour must match the console quote on the day.
5. **GitHub push from this environment is refused** (the session has no GitHub App access for this organization), so the owner must push the branch and open the pull request; the final report states the exact commands.

## 7. A40-specific unverified checks

Not verifiable without the GPU, recorded as UNVERIFIED: bf16 correctness and the exact-resume preflight on CUDA (deterministic kernels are assumed; a mismatch blocks the run command); peak allocated and reserved memory for microbatches 16, 32, 8 and 4; tokens per second; the image pull and bootstrap time; the container disk write speed; the SSH transfer rate from the Pod to the Mac; whether `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` hosts in the chosen data center expose a public IP; the exact console price.

## 8. Dataset and model checks

- Model shape and count are pinned by `tests/test_pilot_specification.py` (135,285,504 parameters; 12 x 768 x 12; context 1,024; vocabulary 65,536).
- Effective batch 131,072 tokens: accumulation 8, 4, 16 and 32 for microbatches 16, 32, 8 and 4.
- Updates: 3 x 100 screening plus 400 continuation = 700; maximum aggregate 91,750,400 tokens (test-pinned). Probes add 20 updates x 2 microbatches per size (discarded) on top.
- Token semantics (from the loader code): each screening arm consumes the same first 13,107,200 tokens in the same seeded document order; the winner continuation resumes at that position and reaches 65,536,000 tokens at step 500, which is 0.9925 passes over the stated 66.03M-token shard, so no document repeats within the winner's run. The 91.75M aggregate counts the two losing arms' tokens, which were consumed from the same prefix. The run command records this as `data_exposure` using the real binding's token count.
- Leakage: ID-level disjointness plus the new token-level audit (exact duplicate documents, 1,025-token validation windows searched across the whole training stream). It does not detect near-duplicates or tokenisation-boundary variants.
- The tokenizer identity is checked against the binding before model allocation; the shard header, vocabulary and BOS are verified on load.

## 9. Shutdown enforcement status

| Layer | Covers | Does not cover |
| --- | --- | --- |
| Worker deadlines (`training_deadline`, wall, cost, idle) | normal and slow runs | a hung process or a dead container |
| In-Pod self-destruct timer (`tools/pod_self_destruct.py`) | Mac offline, SSH loss, worker crash, GPU hang, watchdog death | Runpod API outage, Pod-scoped key lacking delete permission, host failure (billing usually stops), a killed container |
| Mac watchdog (`tools/runpod_watchdog.py`) | worker crash, GPU hang, delayed termination, identity drift | Mac offline, API outage; never claims success without a confirmed 404 |
| Account backstops | runaway spend | nothing specific to this Pod; $80/hour default limit is far above the pilot |

Net: the 90-minute limit is enforced redundantly but is **not guaranteed**. Network volumes are not used, so no storage survives termination; the only durable artifacts are the Mac copies and the private GitHub records.

## 10. Budget and cost safeguard status

Quoted $0.59/hour GPU plus container disk (20 GB at $0.10/GB-month, about $0.003/hour) gives $0.60/hour in the specification; 90 minutes cost $0.90. The cost stop is `min($5, 1.5 x $0.90) = $1.35` and binds after the wall limit at this price. The watchdog floors the accrued cost at the quoted rate even if the API reports a lower rate. Provider billing is fetched after termination with `tools/runpod_pod_identity.py billing`. Expected spend for a complete run: $0.80 to $0.90; absolute ceiling $5.

## 11. Mac transfer requirements

The Pod must have a public IP with TCP port 22 exposed (the `ssh.runpod.io` proxy cannot carry transfers); the SSH public key must be registered in the Runpod account; the Pod host key is pinned by `ssh-keyscan` into a dedicated `known_hosts` and compared against the web terminal; transfers are Mac-initiated pulls with manifest, size and SHA-256 checks, atomic promotion and a receipt; a dummy 1.5 GiB pull measures the real rate before training; `max_transfer_seconds` (1,500 s) bounds the post-training window; the 900-second finalization reserve plus that window must cover the measured rate. Contingencies: if the Mac goes offline, the pilot still terminates at the wall or idle limit and the checkpoint is lost, while the last pulled records survive; if a pull fails it is retried from the retained export; nothing deletes the Mac copy.

## 12. Exact launch and monitoring commands

See `PILOT_RUNBOOK.md` sections 5 to 7 for the ordered commands. The first billable action is the Pod creation request in section 5 of the runbook:

```sh
export ORIGIN=$(date +%s)
curl -sS -X POST https://api.runpod.io/v2/pods -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" --data @configs/runpod-pod-a40-s.json
```

Monitoring: `tools/runpod_watchdog.py` (API guard, receipt file), `tools/pilot_supervisor.py` (heartbeat mirror, pulls, archive), the Pod's `report.json`, `heartbeat.json` and `updates.jsonl`, and `tools/runpod_pod_identity.py status`.

## 13. Expected outputs and acceptance criteria

Outputs on the Mac under `runs/pilot-controller/`: `pod-identity.json`, `pod-connection.json`, `self-destruct.json`, `preflight-a40-report.json`, `environment.json`, `readiness-receipt.json`, `watchdog-receipt.json`, `supervisor/received/<bundle>/{checkpoint.pt,record.json,training.jsonl,transfer-receipt.json,github-receipt.json}`, `billing.json`. In the private repository: `pilot-runs/<pod-id>/<bundle>/{record.json,training.jsonl}` plus the final record with shutdown and transfer receipts.

Acceptance for a successful pilot: all four probes recorded with status and peaks; a chosen microbatch; three arms each at step 100 with validation losses on identical windows; a winner; a continuation that reached step 500 or stopped for `training_time_limit`/`finalization_reserve`; `last_checkpoint_resume.tested: true`; a Mac bundle whose transfer receipt verifies; a watchdog receipt stating `terminated` with reason ending `delete_204_get_404`; a billing total below $5 and no residual Pod or volume. A stopped or failed run is still a valid pilot result if its reason is recorded; it is not a winner.
