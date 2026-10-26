# Rehearsal run operations (D-35)

The owner approved two rehearsal pretraining runs of the S model (decision D-35), each on its own Runpod A40 Pod at the same time. **Run A** trains `stage1` on English and then `stage2` on Python plus unseen English, with a fresh optimizer and schedule from stage 1's final weights. **Run B** trains one `mixed` stage on the same tokens. Both start from one shared saved initialization. This page describes the Mac-side and Pod-side operations around the training worker; the step-by-step commands are in the [rehearsal runbook](../REHEARSAL_RUNBOOK.md). Nothing here launches a Pod by itself, and no rehearsal run has happened yet.

## Approved limits per run

| Limit | Value |
| --- | --- |
| Pod wall | 45,000 s (12.5 h): watchdog `--mode rehearsal`, self-destruct `--max-lifetime-seconds 45000`, armed 180 s inside the wall |
| Cost | cap $10; estimate $6.2917 x 1.2 = worker stop $7.55; Mac watchdog stop $7.54 |
| Idle | 600 s without GPU activity |
| Launch rule | the Pod lifetime left must hold the whole planned run at the reference 56,479 tok/s (see below) |
| Speed and fit | after 900 s of training: stop below 0.85 x 56,479 tok/s, **or** when the projected finish at the measured rates passes the armed deadline |
| Upload window | `max_transfer_seconds` 900: an `uploading` worker may keep the idle Pod up to 900 s after its last GPU work while the Mac sees new objects reach R2 |
| Quote | $0.59/h GPU, $0.60/h with the 40 GB container disk ([Pod body](../configs/runpod-pod-a40-s-rehearsal.json)); at most $7.50 per Pod, $15.00 for both |

## The two time rules

**At launch.** The receipt tool and the worker apply one rule, written once in `tools/rehearsal_run.py` (`launch_fit` and `require_fit`) and called by both. The lifetime left is measured to the first moment any guard ends the Pod: the 45,000 s wall, the armed in-Pod deadline (180 s earlier), the watchdog's wall, or the worker's or the watchdog's cost stop at its quoted rate (`lifetime_end`). It must hold the whole planned run at the **reference speed**: every update at 56,479 tok/s, every interval and full validation at the planning 100,000 tok/s, 300 s of startup per stage (model build, parent check and compilation), a 15 s checkpoint save every 1,800 s of training and validation and at each stage end, and the 900 s finalization reserve.

| Run | Training | Validation | Startup | Saves | Reserve | Needed | Setup allowance to the armed deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 35,411 s | 865 s | 600 s | 22 x 15 s | 900 s | 38,107 s | 6,713 s (112 min) |
| B | 35,411 s | 743 s | 300 s | 21 x 15 s | 900 s | 37,669 s | 7,151 s (119 min) |

The earlier worker draft planned at the 85% floor instead, which left only about 18 to 21 minutes for Pod setup; at the floor the same plan now needs 44,401 s (A) and 43,963 s (B). The receipt records both plans (`time_fit.reference` and `time_fit.floor`) and reports `fits_at_floor` for information only.

**At 900 s of training.** The worker measures its post-compile throughput, leaving out each stage's first two updates. It stops `throughput_below_limit` below 85% of the reference (48,007 tok/s). It also projects its finish at the measured rates: the remaining updates, every remaining validation, a recompilation per later stage, the remaining checkpoint saves (the planned 15 s each until one save is measured) and the reserve. A projection past the end of the Pod lifetime stops it `projected_finish_after_deadline`. This second rule binds even when the speed passes, so a run that cannot finish stops early, after about $0.15 of training, instead of ending time-limited near the wall. Both rules are recorded in the worker's `report.json` (`launch_fit.rule`, `speed_check.rule`) and in the receipt (`time_fit.rule`).

## What runs where

| Piece | Where | What it does |
| --- | --- | --- |
| `tools/rehearsal_ops.py` | Mac and Pod | The D-35 constants, the receipt's kind and gate names, the worker's statuses and the R2 run layout. `pod-env` writes the Pod's R2 key file to a pipe; `round-trip` tests PUT, HEAD, GET and DELETE under the run's prefix; `fetch-object` downloads one pinned object (the initialization); `verify-final` checks a finished run in R2. |
| `tools/pilot_pod_bootstrap.sh --mode rehearsal` | Pod | Arms the self-destruct timer first, then loads the 0600 R2 key file without printing it, records the environment, installs the project, runs the R2 round trip, fetches and re-verifies the run's pinned data and the initialization, runs the CPU tests (the worker's `tests/test_rehearsal_run.py`, the integration test and every rehearsal test except the Mac-only data build's; none needs Docker or the network), and runs the A40 preflight with `--compile-check`. The pilot path is unchanged. |
| `tools/pilot_readiness_receipt.py rehearsal` | Mac | Builds one receipt per run from the Pod's own receipts and the owner's confirmations, in exactly the form the worker accepts. It reads the worker's specification file with the worker's parser, checks it against D-35 and the binding, requires a passed `compile_equivalence` when the specification compiles, pins the policy's billing origin and 900 s transfer window, and refuses to write if any gate is false or the launch rule fails. |
| `tools/pilot_feasibility.py rehearsal` | Mac | Projects time, cost, container disk and R2 storage for both runs at the reference speed and at the 85% floor, with its own assumptions (Pod setup included). A projection only; the launch rule above is the gate. |
| `tools/runpod_watchdog.py --pidfile` | Mac | One watchdog per run; each writes its own PID file so one run's watchdog can be restarted without touching the other's. Under a `rehearsal` policy it accepts the worker's `uploading` status as a transfer status. |
| `tools/pilot_supervisor.py --mode rehearsal` | Mac | One supervisor per run. It mirrors the heartbeat, guards the Pod through the API, reads `heartbeat.json` and `report.json` over SSH with bounded sizes, never copies a checkpoint over SSH, stamps transfer progress only from new R2 objects it lists with the Mac's key, and approves teardown only after the final state is verified in R2 and archived. |

## Readiness receipt and worker

The worker accepts exactly the receipt `python -m tools.pilot_readiness_receipt rehearsal` writes:

- `kind` is `rehearsal-readiness` and `run` is the specification's run;
- every gate in `rehearsal_ops.READINESS_GATES` is `true`: the owner's seven statements, the two evidence gates derived from the Pod's receipts (`r2_round_trip_tested`, `r2_shards_staged_and_hash_checked`) and `pod_self_destruct_armed`;
- `specification_sha256`, `binding_sha256`, `preflight_sha256` and `initialization_sha256` equal the SHA-256s of the files on the worker's own command line; none is optional;
- `pod_identity` is the captured provider identity. The worker takes its Pod ID from it, because its SSH launch does not source `/etc/rp_environment` and `RUNPOD_POD_ID` is then unset; when that variable is set it must agree. `r2_run_prefix` must be `rehearsal-runs/<run>/<pod id>/`;
- `watchdog_policy` says `"mode": "rehearsal"` and `max_transfer_seconds` 900, carries the create-request billing origin (the policy file's own value must agree with `--billing-origin-unix`), and is no weaker than the specification's limits;
- `self_destruct_deadline_unix` is the armed in-Pod deadline; every worker deadline plans against `lifetime_end`, which is that deadline for the D-35 values.

The worker then applies the launch rule again at its own start, so a receipt written long before the launch cannot carry a stale fit.

## R2 layout

Data sits under the binding's own prefix (see [artifact storage](storage.md)). The shared initialization is stored once at `rehearsal-init/<sha256>/rehearsal-init.pt`. Each run writes under `rehearsal-runs/<run>/<pod id>/`:

- rolling labels `<stage>-step<global update>`, of which the uploader keeps only the newest (two by default);
- milestone labels `<stage>-final` (run A: `stage1-final` and `stage2-final`; run B: `mixed-final`), never deleted;
- each label holds `checkpoint.pt`, `checkpoint.pt.sha256`, `record.json`, `updates.jsonl` (cumulative) and, for a stage end, `ticket.json`;
- `report.json`, the run summary, at most 1,000,000 bytes;
- `preflight/round-trip-<random>.bin` exists only for the moment of the round-trip test.

## Credentials

The Pods use their own Cloudflare token, `R2_POD_ACCESS_KEY_ID` and `R2_POD_SECRET_ACCESS_KEY`, limited to the one bucket. The operator pipes the key file from `rehearsal_ops pod-env` over SSH stdin into `/workspace/rehearsal/r2-pod.env` with mode 0600. `pod-env` refuses to write to a terminal or to hand out the Mac's own key. The bootstrap refuses a file that is not 0600 and refuses to continue if the Mac's variables (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) are set on the Pod. The Runpod create body carries no environment variables at all. The Mac keeps `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` to verify what the Pods wrote, and `ssh` and `gh` child processes never inherit any `R2_` variable.

## When a Pod may be deleted

The worker's heartbeat stays `running` through every stage, including run A's boundary between its two stages. When training ends, for any reason other than a wall or cost stop, it reads `uploading` while the worker drains its uploader (at most 480 s, and never later than 120 s before the end of the 900 s transfer window), checks every kept label in R2 by HEAD, and uploads and checks `report.json`. Only then does it write its final status: `awaiting_transfer` with `r2_final` (the report and checkpoint keys and their SHA-256s) for a verified completed or time-limited run, or `stopped`/`failed`.

While the heartbeat reads `uploading` or `awaiting_transfer`, the supervisor lists the run's prefix with the Mac's key. The first listing is a baseline; any later new object stamps `transfer_progress_unix` in the mirrored heartbeat. The watchdog lets an idle GPU pass its 600 s limit only for these statuses, only while that stamp is younger than 600 s, and never more than 900 s after the last GPU work. The Pod cannot stamp it itself, so a hung worker still ends at the idle limit. The supervisor treats `uploading` exactly like `running`: no verification and no teardown.

On `awaiting_transfer` with `r2_final`, the supervisor, with the Mac's own key:

1. checks that the SSH copy of `report.json` hashes to `r2_final.report_sha256`;
2. HEADs the final checkpoint, `report.json` and the final label's other files, and compares the stored SHA-256 metadata;
3. downloads `report.json` pinned to that hash and size;
4. checks that the report closes the run: the worker lists every planned stage in order, a stage it never trained as `pending` with no final checkpoint; the stages with a final checkpoint come first, the last of them is the heartbeat's, and every earlier one's milestone is in R2 with the recorded hash. A completed run must have closed every stage, so run A's `stage1-final` is never its completion;
5. archives `report.json` and a small `summary.json` to `rehearsal-runs/<run>/<pod id>/final/` in the private artifacts repository, reading each back byte for byte;
6. only then marks `shutdown_ready`, which lets the watchdog and its own guard delete the Pod.

Any mismatch stops at its step and is retried each cycle; it never approves teardown, and the idle limit still ends the Pod 600 seconds after its last GPU activity (up to 900 s while new objects keep reaching R2). The worker publishes `r2_final` only when a verified stage-final milestone closes what it trained, so a stopped run A whose `stage1-final` is in R2 is verified the same way under its own status. A worker that reports `stopped` or `failed` without `r2_final` gets a 60-second grace (its own drain is already over), then its own report and a list of the labels that reached R2 are archived under `diagnostics/`, and teardown is approved. `updates.jsonl` and every checkpoint stay in R2; GitHub only holds the small records.

## Projected fit

`uv run --locked python -m tools.pilot_feasibility rehearsal` with the S build's counts and its default assumptions (startup 20 minutes, a 120-second compile per stage, forward-only validation at twice the training rate, 15 seconds per checkpoint save):

| Run | Updates | Reference 56,479 tok/s | Floor 48,007 tok/s |
| --- | ---: | --- | --- |
| A (stage1, stage2) | 15,260 | 10.8 h total; fits with 100 min to spare | 12.6 h total; **does not fit** by 7 min; fits only if startup takes at most 12.7 min |
| B (mixed) | 15,259 | 10.7 h total; fits with 104 min to spare | 12.5 h total; **does not fit** by 3 min; fits only if startup takes at most 17.1 min |

These projections use their own assumptions and are not the launch gate; the launch rule above is. At the quoted $0.60/h each Pod costs at most $7.50 at its wall, inside the $7.55 stop; a quote of $0.61/h GPU ($0.62/h total) would put the wall above it. Each Pod needs 16 to 18 GB of its 40 GB container disk with headroom, and the bucket holds about 16 GB ($0.09 a month). These are projections from one measured pilot, not rehearsal measurements: a run near the floor speed will be stopped by the 900-second projected-finish rule.

## Verification

`uv run --locked pytest tests/test_rehearsal_run.py tests/test_rehearsal_integration.py tests/test_rehearsal_runbook.py tests/test_rehearsal_supervisor.py tests/test_rehearsal_ops.py tests/test_rehearsal_bootstrap.py tests/test_rehearsal_readiness.py tests/test_rehearsal_feasibility.py tests/test_rehearsal_watchdog.py` runs offline. The tests cover:

- every tool command in the runbook, parsed by the real parsers, `write-initialization` with both specifications, and the policy file against the watchdog command line;
- a receipt built by `build_rehearsal_receipt` from fixture evidence and accepted by the worker's `check_readiness` and its `run` command line, and refused for a tampered specification, binding, preflight or initialization hash;
- the worker's CPU smoke for both runs replayed heartbeat by heartbeat, with its `report.json` and in-memory bucket, into the real supervisor over fake SSH and GitHub: no teardown at run A's stage boundary or while uploading, `shutdown_ready` only after the R2 verification and archive, and never after a mismatched R2 SHA-256;
- the supervisor with fake SSH, R2 and GitHub, including the R2-observed progress stamps, and the watchdog's transfer window under a rehearsal policy;
- the bootstrap run end to end against stub programs: the timer first, the 45,000-second ceiling, the 0600 key file, the Pod key never printed, the compile check, the on-Pod test list, the pilot path unchanged;
- receipts refused for each missing gate, weak limit, missing compile check, wrong transfer window or billing origin, and a specification that differs from D-35 or the binding; the feasibility arithmetic and the watchdog PID file.

## Limitations

- No rehearsal run has been measured; every time and cost figure above is a projection.
- Termination is layered best-effort (in-Pod timer, Mac watchdog, supervisor guard, worker deadlines); Runpod offers no provider-enforced Pod lifetime.
- The Mac must stay awake and online for the whole run: a sleeping Mac stops the heartbeat mirror, and on waking the watchdog would read a stale heartbeat and delete the Pod.
- R2 progress is seen only when an object completes: an upload of one 1.6 GB checkpoint slower than about 2.7 MB/s shows no progress for 600 s and the idle limit ends the Pod.
