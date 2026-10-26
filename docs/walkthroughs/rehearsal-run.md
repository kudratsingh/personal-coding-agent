# Rehearsal run worker

`tools/rehearsal_run.py` trains one of the two rehearsal runs on a single A40 Pod that already exists. Run A trains `stage1` on English web text, then `stage2` on Python plus unseen English. Run B trains one `mixed` stage on all three streams. Both start from the same saved initialization and read exactly the same documents, each exactly once. The worker never creates, rents or deletes a machine. Checkpoints leave the Pod through R2 while training continues. The Mac-side tools watch the heartbeat and delete the Pod.

The two specifications are [configs/rehearsal-a40-s-a.json](../../configs/rehearsal-a40-s-a.json) and [configs/rehearsal-a40-s-b.json](../../configs/rehearsal-a40-s-b.json). They hold the owner-approved values of decision D-35.

## Worked example

**Tiny smoke.** `python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke` writes authored shards for every bound stream. It writes one seed-0 initialization and trains both runs on CPU against an in-memory bucket. The tiny model packs 8-token windows, 2 per microbatch and 2 microbatches per update, so one update holds 32 targets. Run A's `stage1` has 254 targets: `ceil(254 / 8) = 32` windows and `ceil(32 / 4) = 8` updates. The last update holds `254 - 7 x 32 = 30` real targets; the rest of it is padding. `stage2` adds 216 targets in 7 updates, numbered 9 to 15. Run B's single stage holds all 470 targets in 15 updates. With a validation interval of 4, run A validates its subset at global updates 0, 4 and 8 in `stage1` and at 12 in `stage2`. It validates both full streams at updates 8 and 15. Run B validates at 0, 4, 8 and 12, and fully at 15. The smoke prints both runs' multiset digests; they are equal.

**Real runs.** At the S packing an update holds 16 x 8 = 128 windows of 1,024 tokens (131,072 targets). Each of run A's stages has 10^9 targets, so 976,563 windows and 7,630 updates. Warmup is `round(0.02 x 7,630) = 153` updates. The final update holds 51 windows: 50 full ones and one with 512 targets. Run B has 2 x 10^9 targets in 15,259 updates with a 305-update warmup. Stage 1's first update uses `6e-4 / 153` and its last `6e-5`. Stage 2 restarts at `3e-4 / 153` with a fresh optimizer. Validation runs at global updates 0, 500, ..., 15,000: 31 times in each run. Run A's stage-2 validations start at 8,000, continuing its global count.

**Launch rule.** The plan is the whole run at the reference speed of 56,479 tokens a second: 2 x 10^9 targets take 35,411 s. Validation adds 865 s for A and 743 s for B at the planning rate of 100,000 tokens a second. Startup is 300 s per stage (600 s for A, 300 s for B). Checkpoints are one 15 s save per 1,800 s of training and validation plus one per stage end: 22 saves (330 s) for A and 21 (315 s) for B. The finalization reserve is 900 s. In total A needs 38,107 s and B 37,669 s. The worker refuses to start unless that much Pod lifetime is left, and the receipt tool refuses to write a receipt on the same rule (`launch_fit`, `require_fit`). The lifetime ends at the first of the 45,000 s wall, the armed in-Pod deadline, the watchdog's wall and either cost stop (`lifetime_end`). With the deadline armed 180 s inside the wall, Pod setup may take up to 6,713 s for A and 7,151 s for B. At the 85% floor the same plan would need 44,401 s and 43,963 s; that is the 900 s check's business, not the launch gate's.

## Data flow and shapes

1. **Launch gates (`run`).** The worker accepts exactly the receipt `python -m tools.pilot_readiness_receipt rehearsal` writes (`check_readiness`):
   - `kind` is `rehearsal-readiness` and `run` is this specification's run;
   - the specification, the binding, the preflight report and the initialization are bound by SHA-256, and none of the four is optional;
   - every gate in `REHEARSAL_GATES` (`rehearsal_ops.READINESS_GATES`, the receipt builder's own names) is `true`: the pilot's applicable gates, `pod_self_destruct_armed`, and `r2_round_trip_tested`, `r2_shards_staged_and_hash_checked` and `r2_token_bucket_scoped`.

   The pilot's `ssh_transfer_tested` is not required, because checkpoints leave through R2. The termination guarantee, Pod identity, watchdog policy and armed deadline are checked by the same helpers as the pilot's. The policy must say `mode: rehearsal` and carry this specification's `max_transfer_seconds`. The Pod ID comes from the receipt's captured identity, because the worker's SSH launch does not source `/etc/rp_environment`; `RUNPOD_POD_ID`, when set, must match it, and `r2_run_prefix` must be this Pod's prefix. Then the launch rule above. The A40 preflight must have passed its precision, resume, fused-attention and compile checks (`tools.pilot_preflight --compile-check`). R2 credentials come from `R2_ENDPOINT`, `R2_BUCKET`, `R2_POD_ACCESS_KEY_ID` and `R2_POD_SECRET_ACCESS_KEY`.
2. **Data.** `load_run_data` calls `rehearsal_binding.verify_binding` for the run. That re-hashes every shard, manifest and the tokenizer, and recomputes each stage's document order. The specification's packing and planned updates must equal the binding's. The initialization must hash to `--initialization-sha256` and be in the parent-checkpoint format `{"config": {"model": ...}, "tokenizer_hash", "model", "run_id": "rehearsal-init"}`.
3. **Each stage.**
   - The parent's bytes are hashed against the SHA-256 recorded when it was saved: the initialization for stage 1, stage 1's `final.pt` for stage 2.
   - A `OnePassLoader` over the stage's shards emits microbatches of `[16, 1024]` inputs and targets.
   - A `Trainer` loads the parent's weights only, with a fresh AdamW and a cosine schedule with its own warmup. `torch.compile` is on.
   - After each update the worker appends a row to `updates.jsonl` and writes the heartbeat.
   - At every multiple of 500 global updates it validates the fixed subset: the first 1,000,000 targets of each validation stream, cut into document windows of at most 1,025 tokens. Batches hold 8 windows with bfloat16 autocast; logits are `[8, <=1024, 65536]`.
   - Every 1,800 s it saves a rolling checkpoint.
   - At the stage end it compares the loader's ticket with the binding's: documents, targets, windows, updates, multiset and order digests. It saves the milestone, then validates both full streams.
4. **Checkpoints.** Each save writes `checkpoints/<stage>/last.pt` (rolling) or `final.pt` (milestone). It then publishes `exports/<label>/` with `checkpoint.pt` (a hard link), `checkpoint.pt.sha256`, `record.json` (the report so far), `updates.jsonl` (all rows so far) and, for milestones, `ticket.json`. The directory is submitted to `CheckpointUploader` under `rehearsal-runs/<run>/<pod id>/<label>/`. Labels are `<stage>-step<global update>` and `<stage>-final`. The uploader keeps the two newest rolling labels and every milestone. Older local rolling directories are removed once the uploader no longer needs them.
5. **Finalization.**
   - The heartbeat reads `uploading` from here until the last step, whatever the training outcome. The GPU is idle; the watchdog allows this for the policy's `max_transfer_seconds` (900 s) after the last GPU work, while the Mac sees new objects reach R2.
   - The worker drains the uploader for at most `drain_seconds` (480 s), never later than `FINALIZE_MARGIN_SECONDS` (120 s) before the end of that window, and never past the Pod end.
   - It checks every kept label in R2 by HEAD against the digests it sent.
   - It writes `report.json`, uploads it to `rehearsal-runs/<run>/<pod id>/report.json` and checks that too.
   - Only then does the heartbeat read its final status. A completed or time-limited run whose milestones are verified reads `awaiting_transfer`, with `r2_final`: the report key and SHA-256 and the final checkpoint's key and SHA-256. `r2_final` always names the last stage-final milestone, never a rolling label, and it is published for a stopped run too when that milestone and every earlier one are verified.
   - `finalization.json` records the drain and every check. A wall or cost stop skips all of this and reads `stopped` at once.

## Design choices and common mistakes

- **One stage end is not run completion.** The heartbeat reads `running` through every stage and across run A's boundary, then `uploading` during finalization. Every label's `record.json` also says `running`. A completed or time-limited run reads `awaiting_transfer` only once every milestone and `report.json` are verified in R2. If they cannot be verified, the run ends `failed` with `training_status` set, and the operator decides whether to pull the local copy over SSH.
- **Every planned stage is in the report.** A stage that never started is listed as `pending` with no final checkpoint. The Mac side reads the stages with a final checkpoint as the closed ones, so a run A time-limited between its stages closes at `stage1-final`, but only a run that closed every stage counts as completed.
- **Global numbering.** Run A's stage 2 continues the global update count. Its trainer still numbers its own steps from 1, so its learning-rate schedule and warmup restart.
- **Verify before trusting a parent.** The trainer reads its parent by path. Hashing the file first means a corrupted stage-1 checkpoint stops the run before stage 2 rather than training from it.
- **Post-compile speed, then fit.** At 900 s the speed check measures throughput, leaving out each stage's first two updates, which carry compilation. A run below 85% of 56,479 tokens a second stops (`throughput_below_limit`). So does a run whose projected finish passes the Pod end, the armed deadline when that comes first (`projected_finish_after_deadline`); this rule binds even when the speed passes. The projection adds, at measured rates, the remaining targets, every remaining validation, a recompilation per later stage, the remaining checkpoint saves and the 900 s reserve. No save has happened by 900 s, so each save counts at the planned `checkpoint_save_seconds` (15 s) until one is measured. The report records the rule (`speed_check.rule`) next to the launch rule (`launch_fit.rule`).
- **The reserve permits finishing, not training.** Training stops when only the 900 s reserve is left. The current stage still gets its milestone save and uploads, but no full validation and no next stage, and the run ends `time_limited`. The wall, cost (`$7.55`) and idle (600 s) limits are never extended. A wall or cost stop does not even wait for uploads.
- **Stops as in the pilot.** Twenty-five consecutive rising losses and the speed check end a run `stopped`. A nonfinite loss or metric ends it `failed`.
- **Records stay small and secret-free.** `report.json` must stay under 1,000,000 bytes; per-update rows live in `updates.jsonl`. Everything written goes through `sanitize` and then has every known credential value replaced, the R2 key pair included. An R2 error that echoes the key ID is redacted. Credentials never enter the Runpod create body or any argument.
- **Validation batches pad little.** The subset windows are sorted by length once. The full streams are sorted in runs of 512 windows. Only the batching changes; every window is still scored once.
- **Planning numbers.** `validation_tokens_per_second`, `startup_seconds`, `checkpoint_save_seconds` and `drain_seconds` are planning bounds, not measurements. The speed check replaces the validation rate and the save time with measured ones. `max_transfer_seconds` must equal the watchdog policy's; the receipt tool and the worker both check it.
- **Key order.** The specification's validation streams are read in the binding's order whatever the JSON file's key order, so a file rewritten with sorted keys means the same run (its SHA-256 still changes, and the receipt binds the bytes).
- **Not covered.** There is no resume after a crash: a failed run keeps its R2 labels and stops. A CPU smoke says nothing about A40 speed, memory or bfloat16 behavior.

## Verification

Implementation: [`tools/rehearsal_run.py`](../../tools/rehearsal_run.py); the compile check is in [`tools/pilot_preflight.py`](../../tools/pilot_preflight.py). Reference tests: [`tests/test_rehearsal_run.py`](../../tests/test_rehearsal_run.py), run with:

```sh
uv run --locked pytest tests/test_rehearsal_run.py tests/test_rehearsal_integration.py tests/test_rehearsal_readiness.py
uv run --locked python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke
```

The tests run both runs on authored shards with a fake clock and an in-memory bucket that answers HEAD and GET as R2 does. They check:

- the approved values in both specifications, the launch rule's arithmetic at the reference speed and the strict schema;
- that a receipt built by the receipt tool is accepted by `check_readiness` and the `run` command line, and that a tampered specification, binding, preflight or initialization hash, another run's receipt, a wrong transfer window or R2 prefix is refused;
- the smoke's heartbeats, reports and bucket replayed into the real supervisor, which tears down only after verifying R2, and never on a mismatched SHA-256;
- the final drain's budget inside the transfer window, and the `uploading` heartbeat with no GPU activity;
- the stage transition: the parent hash, a fresh optimizer, a new peak and warmup, and a refused tampered stage-1 checkpoint;
- exact one-pass exhaustion with the masked final update;
- the global validation cadence and full validation at each stage end;
- rolling and milestone labels in the bucket, retention and local pruning;
- that A's two stage multisets combine to B's and both equal the bound value;
- the speed-check abort paths and its post-compile exclusion;
- the cost, wall, idle, rising-loss, nonfinite and reserve stops;
- the heartbeat across the stage boundary, and `awaiting_transfer` only after R2 verification;
- the report size bound, and that no credential value reaches any file or object.

They also check the readiness and preflight gates, the shared initialization, and the glue between a verified binding and the worker.

## Questions and answers

1. *Why can run A's two stages be compared with run B at all?* Both runs start from the same saved initialization, and their tickets prove they read the same token multiset. The comparison still differs in the stage-2 optimizer and schedule reset, so it is a recipe comparison, not an order-only one.
2. *Why does the heartbeat not read `awaiting_transfer` as soon as training ends?* The Mac side deletes the Pod on that status. Until the milestones and the report are verified in R2, the Pod still holds the only copy. Meanwhile it reads `uploading`, which keeps the idle Pod alive for the transfer window only while the Mac sees the uploads arrive.
3. *What happens if training is slower than planned?* The launch rule needs the whole run to fit at the reference speed. The 900 s check then stops a run below the speed floor, or one projected to pass the deadline at its measured speed, while little money is spent. A run that slows later stops at the reserve with a time-limited milestone.
4. *Why hash the stage-1 checkpoint again before stage 2?* The trainer loads its parent by path. A file changed after its save must not become the starting weights.
5. *Why is validation every 500 global updates, not every 500 stage updates?* Global numbering puts run A's and run B's validations at the same token counts, so their curves line up.
