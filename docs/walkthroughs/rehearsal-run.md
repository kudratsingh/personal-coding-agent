# Rehearsal run worker

`tools/rehearsal_run.py` trains one of the two rehearsal runs on a single A40 Pod that already exists. Run A trains `stage1` on English web text, then `stage2` on Python plus unseen English. Run B trains one `mixed` stage on all three streams. Both start from the same saved initialization and read exactly the same documents, each exactly once. The worker never creates, rents or deletes a machine. Checkpoints leave the Pod through R2 while training continues. The Mac-side tools watch the heartbeat and delete the Pod.

The two specifications are [configs/rehearsal-a40-s-a.json](../../configs/rehearsal-a40-s-a.json) and [configs/rehearsal-a40-s-b.json](../../configs/rehearsal-a40-s-b.json). They hold the owner-approved values of decision D-35.

## Worked example

**Tiny smoke.** `python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke` writes authored shards for every bound stream. It writes one seed-0 initialization and trains both runs on CPU against an in-memory bucket. The tiny model packs 8-token windows, 2 per microbatch and 2 microbatches per update, so one update holds 32 targets. Run A's `stage1` has 254 targets: `ceil(254 / 8) = 32` windows and `ceil(32 / 4) = 8` updates. The last update holds `254 - 7 x 32 = 30` real targets; the rest of it is padding. `stage2` adds 216 targets in 7 updates, numbered 9 to 15. Run B's single stage holds all 470 targets in 15 updates. With a validation interval of 4, run A validates its subset at global updates 0, 4 and 8 in `stage1` and at 12 in `stage2`. It validates both full streams at updates 8 and 15. Run B validates at 0, 4, 8 and 12, and fully at 15. The smoke prints both runs' multiset digests; they are equal.

**Real runs.** At the S packing an update holds 16 x 8 = 128 windows of 1,024 tokens (131,072 targets). Each of run A's stages has 10^9 targets, so 976,563 windows and 7,630 updates. Warmup is `round(0.02 x 7,630) = 153` updates. The final update holds 51 windows: 50 full ones and one with 512 targets. Run B has 2 x 10^9 targets in 15,259 updates with a 305-update warmup. Stage 1's first update uses `6e-4 / 153` and its last `6e-5`. Stage 2 restarts at `3e-4 / 153` with a fresh optimizer. Validation runs at global updates 0, 500, ..., 15,000: 31 times in each run. Run A's stage-2 validations start at 8,000, continuing its global count.

**Launch fit.** At the 85% speed floor (`0.85 x 56,479 = 48,007` tokens a second), 2 x 10^9 targets take 41,660 s. Validation adds 865 s for A and 743 s for B at the planning rate of 100,000 tokens a second. Startup is 300 s and the finalization reserve 900 s. In total A needs 43,726 s and B 43,603 s. The worker refuses to start unless that much Pod lifetime is left. The lifetime ends at the 45,000 s wall or the armed in-Pod deadline, whichever comes first.

## Data flow and shapes

1. **Launch gates (`run`).** The readiness receipt must bind the specification, the binding, the preflight report and, when it names one, the initialization by SHA-256. It must set every gate in `REHEARSAL_GATES` to `true`:
   - the pilot's applicable gates;
   - `r2_round_trip_tested`, `r2_shards_staged_and_hash_checked` and `r2_token_bucket_scoped`.

   The pilot's `ssh_transfer_tested` is not required, because checkpoints leave through R2. The termination guarantee, Pod identity, watchdog policy and armed deadline are checked by the same helpers as the pilot's. A 45,000 s policy must say `mode: rehearsal`. The worker then checks that the remaining lifetime fits `planned_seconds`. The A40 preflight must have passed its precision, resume, fused-attention and compile checks (`tools.pilot_preflight --compile-check`). R2 credentials come from `R2_ENDPOINT`, `R2_BUCKET`, `R2_POD_ACCESS_KEY_ID` and `R2_POD_SECRET_ACCESS_KEY`. `RUNPOD_POD_ID`, when set, must match the receipt.
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
   - The worker drains the uploader for at most `drain_seconds` (480 s), and never past the idle window or the Pod end.
   - It checks every kept label in R2 by HEAD against the digests it sent.
   - It writes `report.json`, uploads it to `rehearsal-runs/<run>/<pod id>/report.json` and checks that too.
   - Only then does the heartbeat read `awaiting_transfer`. It carries `r2_final`: the report key and SHA-256 and the final checkpoint's key and SHA-256.
   - `finalization.json` records the drain and every check.

## Design choices and common mistakes

- **One stage end is not run completion.** The heartbeat reads `running` through every stage and across run A's boundary. Every label's `record.json` also says `running`. A completed or time-limited run reads `awaiting_transfer` only once every milestone and `report.json` are verified in R2. If they cannot be verified, the run ends `failed` with `training_status` set, and the operator decides whether to pull the local copy over SSH.
- **Global numbering.** Run A's stage 2 continues the global update count. Its trainer still numbers its own steps from 1, so its learning-rate schedule and warmup restart.
- **Verify before trusting a parent.** The trainer reads its parent by path. Hashing the file first means a corrupted stage-1 checkpoint stops the run before stage 2 rather than training from it.
- **Post-compile speed.** At 900 s the speed check measures throughput, leaving out each stage's first two updates, which carry compilation. A run below 85% of 56,479 tokens a second stops. So does a run whose projected finish passes the Pod end. The projection adds, at measured rates, the remaining targets, every remaining validation, a recompilation per later stage, the measured checkpoint saves and the 900 s reserve. No save has happened by 900 s, so saves are not yet counted.
- **The reserve permits finishing, not training.** Training stops when only the 900 s reserve is left. The current stage still gets its milestone save and uploads, but no full validation and no next stage, and the run ends `time_limited`. The wall, cost (`$7.55`) and idle (600 s) limits are never extended. A wall or cost stop does not even wait for uploads.
- **Stops as in the pilot.** Twenty-five consecutive rising losses and the speed check end a run `stopped`. A nonfinite loss or metric ends it `failed`.
- **Records stay small and secret-free.** `report.json` must stay under 1,000,000 bytes; per-update rows live in `updates.jsonl`. Everything written goes through `sanitize` and then has every known credential value replaced, the R2 key pair included. An R2 error that echoes the key ID is redacted. Credentials never enter the Runpod create body or any argument.
- **Validation batches pad little.** The subset windows are sorted by length once. The full streams are sorted in runs of 512 windows. Only the batching changes; every window is still scored once.
- **Planning numbers.** `validation_tokens_per_second`, `startup_seconds` and `drain_seconds` are planning bounds, not measurements. The speed check replaces the validation rate with the measured one.
- **Not covered.** There is no resume after a crash: a failed run keeps its R2 labels and stops. A CPU smoke says nothing about A40 speed, memory or bfloat16 behavior.

## Verification

Implementation: [`tools/rehearsal_run.py`](../../tools/rehearsal_run.py); the compile check is in [`tools/pilot_preflight.py`](../../tools/pilot_preflight.py). Reference tests: [`tests/test_rehearsal_run.py`](../../tests/test_rehearsal_run.py), run with:

```sh
uv run --locked pytest tests/test_rehearsal_run.py
uv run --locked python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke
```

The tests run both runs on authored shards with a fake clock and an in-memory bucket. They check:

- the approved values in both specifications, the launch-fit arithmetic and the strict schema;
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
2. *Why does the heartbeat not read `awaiting_transfer` as soon as training ends?* The Mac side deletes the Pod on that status. Until the milestones and the report are verified in R2, the Pod still holds the only copy.
3. *What happens if training is slower than planned?* The 900 s check stops a run below the speed floor, or one projected to pass the deadline, while little money is spent. A run that slows later stops at the reserve with a time-limited milestone.
4. *Why hash the stage-1 checkpoint again before stage 2?* The trainer loads its parent by path. A file changed after its save must not become the starting weights.
5. *Why is validation every 500 global updates, not every 500 stage updates?* Global numbering puts run A's and run B's validations at the same token counts, so their curves line up.
