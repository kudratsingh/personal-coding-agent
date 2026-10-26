# Rehearsal run operations (D-35)

The owner approved two rehearsal pretraining runs of the S model (decision D-35), each on its own Runpod A40 Pod at the same time. **Run A** trains `stage1` on English and then `stage2` on Python plus unseen English, with a fresh optimizer and schedule from stage 1's final weights. **Run B** trains one `mixed` stage on the same tokens. Both start from one shared saved initialization. This page describes the Mac-side and Pod-side operations around the training worker; the step-by-step commands are in the [rehearsal runbook](../REHEARSAL_RUNBOOK.md). Nothing here launches a Pod by itself, and no rehearsal run has happened yet.

## Approved limits per run

| Limit | Value |
| --- | --- |
| Pod wall | 60,000 s (16.7 h; owner, 2026-10-25, so the $10 cap is the binding limit): watchdog `--mode rehearsal`, self-destruct `--max-lifetime-seconds 60000`, armed 180 s inside the wall |
| Cost | cap $10, also the hard stop for runs A and B (owner, 2026-10-25): worker stop $10.00 (estimate $6.2917 x 1.6 = $10.07, capped); Mac watchdog stop $9.99. At $0.60/h the 60,000 s wall falls at $10.00; the watchdog's $9.99 stop (59,940 s) and the in-Pod timer (59,820 s, about $9.97) end a Pod just before it |
| Idle | 600 s without GPU activity |
| Launch rule | the Pod lifetime left must hold the whole planned run at the reference 56,479 tok/s (see below) |
| Speed and fit | after 900 s of training: stop below 0.9 x 56,479 tok/s (50,831; owner, 2026-10-25), **or** when the projected finish at the measured rates passes the armed deadline |
| Upload window | `max_transfer_seconds` 900: the idle Pod may stay up to 900 s after its last GPU work while the Mac stamps progress (new R2 objects, or SSH-pull bytes). The worker's drain ends by 360 s, its bounded checks and final heartbeat by 480 s, and the last 420 s are the Mac's SSH-pull reserve |
| Upload rate | the Pod's measured R2 upload rate must reach `REQUIRED_UPLOAD_BYTES_PER_SECOND`, 1 x 1.65 GB x 2 / 360 s = 9.17 MB/s: one label in the 360 s drain with a factor-2 margin |
| Quote | $0.59/h GPU, $0.60/h with the 40 GB container disk ([Pod body](../configs/runpod-pod-a40-s-rehearsal.json)); at most $10.00 per Pod, $20.00 for both; the captured Pod's `cost_usd_per_hour` must not exceed $0.60 |

## The two time rules

**At launch.** The receipt tool and the worker apply one rule, written once in `tools/rehearsal_run.py` (`launch_fit` and `require_fit`) and called by both. The lifetime left is measured to the first moment any guard ends the Pod: the 60,000 s wall, the armed in-Pod deadline (180 s earlier), the watchdog's wall, or the worker's or the watchdog's cost stop at its quoted rate (`lifetime_end`). It must hold the whole planned run at the **reference speed**: every update at 56,479 tok/s, every interval validation and every full validation (both full streams at each quarter and end of every stage: 8 passes for A, 4 for B) at the planning 100,000 tok/s, 300 s of startup per stage (model build, parent check and compilation), a 15 s checkpoint save every 1,800 s of training and validation and at each stage end, and the 900 s finalization reserve.

| Run | Training | Validation | Startup | Saves | Reserve | Needed | Setup allowance to the armed deadline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 35,411 s | 1,598 s | 600 s | 22 x 15 s | 900 s | 38,839 s | 20,981 s (350 min) |
| B | 35,411 s | 1,109 s | 300 s | 21 x 15 s | 900 s | 38,035 s | 21,785 s (363 min) |

An earlier worker draft planned at the speed floor instead, which left only about 18 to 21 minutes for Pod setup. At the 90% floor the same plan now needs 42,804 s (A) and 42,000 s (B), which fit before the 59,820 s armed deadline when Pod setup takes at most 17,016 s (A) or 17,820 s (B). These figures come from the reviewed binding: full passes of 12,220,020 validation targets, about 122 s each. The receipt records both plans (`time_fit.reference` and `time_fit.floor`) and reports `fits_at_floor` for information only.

**At 900 s of training.** The worker measures its post-compile throughput, leaving out each stage's first two updates. It stops `throughput_below_limit` below 90% of the reference (50,831 tok/s). It also projects its finish at the measured rates: the remaining updates, every remaining validation, a recompilation per later stage, the remaining checkpoint saves (the planned 15 s each until one save is measured) and the reserve. A projection past the end of the Pod lifetime stops it `projected_finish_after_deadline`. This second rule binds even when the speed passes, so a run that cannot finish stops early, after about $0.15 of training, instead of ending time-limited near the wall. Both rules are recorded in the worker's `report.json` (`launch_fit.rule`, `speed_check.rule`) and in the receipt (`time_fit.rule`).

## What runs where

| Piece | Where | What it does |
| --- | --- | --- |
| `tools/rehearsal_ops.py` | Mac and Pod | The D-35 constants (the required upload rate among them), the receipt's kind and gate names, the worker's statuses, the code-revision record and the R2 run layout. `pod-env` writes the Pod's R2 key file to a pipe; `round-trip` tests PUT, HEAD, GET and DELETE of a 128 MiB probe under the run's prefix and records `upload_bytes_per_second` and `download_bytes_per_second`; `fetch-object` downloads one pinned object (the initialization); `verify-final` checks a finished run in R2. |
| `tools/pilot_pod_bootstrap.sh --mode rehearsal` | Pod | Arms the self-destruct timer first, then loads the 0600 R2 key file without printing it, records the environment, installs the project, runs the R2 round trip, fetches and re-verifies the run's pinned data and the initialization, runs the CPU tests (the worker's `tests/test_rehearsal_run.py`, the integration test and every rehearsal test except the Mac-only data build's; none needs Docker or the network), and runs the A40 preflight with `--compile-check`. The pilot path is unchanged. |
| `tools/pilot_readiness_receipt.py rehearsal` | Mac | Builds one receipt per run from the Pod's own receipts and the owner's confirmations, in exactly the form the worker accepts. It reads the worker's specification file with the worker's parser, checks it against D-35 and the binding, requires a passed `compile_equivalence` when the specification compiles, a round trip fast enough for the upload window, a self-destruct receipt the live timer rewrote within 60 s, a captured Pod price within the quote and a clean code revision (`--revision`), pins the policy's billing origin and 900 s transfer window, and refuses to write if any gate is false or the launch rule fails. |
| `tools/pilot_feasibility.py rehearsal` | Mac | Projects time, cost, container disk, R2 storage and the required R2 upload rate for both runs at the reference speed and at the 90% floor, with its own assumptions (Pod setup included). A projection only; the launch rule above is the gate. |
| `tools/runpod_watchdog.py --pidfile` | Mac | One watchdog per run; each writes its own PID file so one run's watchdog can be restarted without touching the other's. Under a `rehearsal` policy it accepts the worker's `uploading` status as a transfer status, and it ignores a mirrored heartbeat that names another Pod or predates this Pod's creation. |
| `tools/pilot_supervisor.py --mode rehearsal` | Mac | One supervisor per run. It mirrors the heartbeat, guards the Pod through the API, reads `heartbeat.json`, `report.json` and `finalization.json` over SSH with bounded sizes, stamps transfer progress only from new R2 objects it lists with the Mac's key (or bytes of its own SSH pull), and approves teardown only after the final state is verified in R2 and archived, or, for a milestone the worker could not verify in R2, after the milestone is pulled over SSH and its SHA-256 checked. |

## Readiness receipt and worker

The worker accepts exactly the receipt `python -m tools.pilot_readiness_receipt rehearsal` writes:

- `kind` is `rehearsal-readiness` and `run` is the specification's run;
- every gate in `rehearsal_ops.READINESS_GATES` is `true`: the owner's seven statements, the two evidence gates derived from the Pod's receipts (`r2_round_trip_tested`, `r2_shards_staged_and_hash_checked`) and `pod_self_destruct_armed`;
- `specification_sha256`, `binding_sha256`, `preflight_sha256` and `initialization_sha256` equal the SHA-256s of the files on the worker's own command line; none is optional;
- `pod_identity` is the captured provider identity. The worker takes its Pod ID from it, because its SSH launch does not source `/etc/rp_environment` and `RUNPOD_POD_ID` is then unset; when that variable is set it must agree. `r2_run_prefix` must be `rehearsal-runs/<run>/<pod id>/`;
- `watchdog_policy` says `"mode": "rehearsal"` and `max_transfer_seconds` 900, carries the create-request billing origin (the policy file's own value must agree with `--billing-origin-unix`), and is no weaker than the specification's limits;
- `self_destruct_deadline_unix` is the armed in-Pod deadline; every worker deadline plans against `lifetime_end`, which is that deadline for the D-35 values.

The receipt tool also refuses, before writing:

- a round trip (`r2_round_trip_tested`) whose probe is smaller than 128 MiB, whose `upload_bytes_per_second` or `download_bytes_per_second` is missing or disagrees with its timed step, or whose upload rate is below `REQUIRED_UPLOAD_BYTES_PER_SECOND` (9.17 MB/s);
- a self-destruct receipt that is not `armed` for this Pod, names no positive `pid`, or whose `checked_at_unix` is more than 60 s older than the receipt (the live timer rewrites it every 5 s, so copy it from the Pod just before running the tool, as the first line of the runbook's receipt block does);
- a captured connection record (`--connection`, default `pod-connection.json` next to `--identity`) for another Pod, without `cost_usd_per_hour`, or above the specification's quoted $0.60/h total: the watchdog stops at the higher of the API cost and the quote, so a higher live price would end the Pod before the worker's deadlines;
- a code revision (`--revision`, default `revision.json` next to `--identity`) that is not exactly `{"git_commit": <40 hex>, "clean": true, "archive_sha256": <64 hex>}`. The runbook writes it on the Mac from the commit and the shipped tarball (refusing a dirty tree) and ships it to the Pod as `/workspace/rehearsal/revision.json`. The receipt records it as `revision`, with `pod_cost_usd_per_hour` and the SHA-256 of both files; the worker refuses to start unless its copy equals the receipt's `revision`, and records it in `report.json` as `code_revision`.

The worker then applies the launch rule again at its own start, so a receipt written long before the launch cannot carry a stale fit.

## R2 layout

Data sits under the binding's own prefix (see [artifact storage](storage.md)). The shared initialization is stored once at `rehearsal-init/<sha256>/rehearsal-init.pt`. Each run writes under `rehearsal-runs/<run>/<pod id>/`:

- rolling labels `<stage>-step<global update>`, of which the uploader keeps only the newest (two by default);
- milestone labels `<stage>-final` (run A: `stage1-final` and `stage2-final`; run B: `mixed-final`), never deleted;
- each label holds `checkpoint.pt`, `checkpoint.pt.sha256`, `record.json`, `updates.jsonl` (cumulative) and, for a stage end, `ticket.json`;
- `report.json`, the run summary, at most 1,000,000 bytes;
- `preflight/round-trip-<random>.bin`, the 128 MiB probe, exists only for the moment of the round-trip test.

## Credentials

The Pods use their own Cloudflare token, `R2_POD_ACCESS_KEY_ID` and `R2_POD_SECRET_ACCESS_KEY`, limited to the one bucket. The operator pipes the key file from `rehearsal_ops pod-env` over SSH stdin into `/workspace/rehearsal/r2-pod.env` with mode 0600. `pod-env` refuses to write to a terminal or to hand out the Mac's own key. The bootstrap refuses a file that is not 0600 and refuses to continue if the Mac's variables (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`) are set on the Pod. The Runpod create body carries no environment variables at all. The Mac keeps `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` to verify what the Pods wrote, and `ssh` and `gh` child processes never inherit any `R2_` variable.

## When a Pod may be deleted

The worker's heartbeat stays `running` through every stage, including run A's boundary between its two stages. When training ends, for any reason other than a wall or cost stop, it reads `uploading` while any milestone is pending or in flight: the worker resubmits every unverified milestone (a failed milestone upload is retried for the rest of the run) and drains its uploader until 360 s after its last GPU work (`DRAIN_SECONDS`, counted from the heartbeat's `last_gpu_activity_unix`, the end of the final save or validation). It then checks, by HEAD, every milestone (the newest first), uploads `report.json` and checks it, and last checks the rolling labels for the record, also when the report upload failed; `finalization.json` lists a rolling label it had no time to check as `labels_unchecked`, so a resumable one can still be found in R2. Each of these R2 calls retries transient failures while time lasts, with a socket timeout of at most 20 s and a quarter of the time left; none starts within 45 s of the handover, except the HEAD that verifies a `report.json` PUT that succeeded (`TRANSFER_MARGIN_SECONDS`, 120 s in all). By 480 s after its last GPU work it writes its final status: `awaiting_transfer` with `r2_final` (the report and checkpoint keys and their SHA-256s) for a verified completed or time-limited run, or `stopped`/`failed`. If a milestone still cannot be verified in R2, the run ends `failed` with `failure: r2_milestones_unverified`, and `finalization.json` lists each unverified milestone as `unverified_milestones` (also in `report.json` as `r2.unverified_milestones`), one row per milestone: `label`, `export` (`exports/<label>`), `export_path`, `checkpoint_sha256`, `checkpoint_size_bytes` and `files`, each file with its `name`, `path`, `sha256` and `size_bytes`. The rest of the 900 s window, 420 s (`PULL_RESERVE_SECONDS`), belongs to the supervisor's SSH pull of those milestones.

While the heartbeat reads `uploading` or `awaiting_transfer`, the supervisor lists the run's prefix with the Mac's key. The first listing is a baseline; any later new object stamps `transfer_progress_unix` in the mirrored heartbeat, and so do the growing bytes of its own SSH pull. The watchdog lets an idle GPU pass its 600 s limit only for these statuses (and the final `stopped`/`failed`), only while that stamp is younger than 600 s, and never more than 900 s after the last GPU work. The Pod cannot stamp it itself, so a hung worker still ends at the idle limit. The supervisor treats `uploading` exactly like `running`: no verification and no teardown.

Every heartbeat carries the worker's `pod_id`. The watchdog and the supervisor treat a heartbeat that names another Pod as absent, never as fresh; under a rehearsal policy so is one whose `updated_at_unix` predates this Pod's `createdAt`. A controller directory reused for a new Pod still holds the previous Pod's mirrored heartbeat (often `shutdown_ready`, or an hour stale); it is ignored, the new watchdog's idle baseline applies, and the supervisor never mirrors or acts on such a heartbeat. The watchdog receipt's `detail` records the reason.

On `awaiting_transfer` with `r2_final`, the supervisor, with the Mac's own key:

1. checks that the SSH copy of `report.json` hashes to `r2_final.report_sha256`;
2. HEADs the final checkpoint, `report.json` and the final label's other files, and compares the stored SHA-256 metadata;
3. downloads `report.json` pinned to that hash and size;
4. checks that the report closes the run: the worker lists every planned stage in order, a stage it never trained as `pending` with no final checkpoint; the stages with a final checkpoint come first, the last of them is the heartbeat's, and every earlier one's milestone is in R2 with the recorded hash. A completed run must have closed every stage, so run A's `stage1-final` is never its completion;
5. archives `report.json` and a small `summary.json` to `rehearsal-runs/<run>/<pod id>/final/` in the private artifacts repository, reading each back byte for byte;
6. only then marks `shutdown_ready`, which lets the watchdog and its own guard delete the Pod.

Any mismatch stops at its step and is retried each cycle; it never approves teardown, and the idle limit still ends the Pod 600 seconds after its last GPU activity (up to 900 s while new objects keep reaching R2 or the SSH pull keeps receiving bytes). The worker publishes `r2_final` only when a verified stage-final milestone closes what it trained, so a stopped run A whose `stage1-final` is in R2 is verified the same way under its own status.

A worker that reports `stopped` or `failed` without `r2_final` is first checked for milestones at risk, those that may exist only on the Pod:

- after `failure: r2_milestones_unverified` or `r2_finalization`: every milestone `finalization.json` lists that the Mac cannot find whole in R2, plus every stage-final milestone the report records whose checkpoint the Mac's own HEAD cannot find with its recorded SHA-256. `finalization.json` must be readable; until it is, the supervisor retries and never approves;
- after any other stop or failure: the listed milestones the Mac cannot find whole in R2, and those the report flags in `r2.milestones_verified` as unverified.

The worker may list a milestone only because it ran out of time to confirm it, so the Mac checks every listed file in R2 first, by HEAD with its own key, stopping at the first that is missing, differs or times out; only a milestone missing a file is pulled. The Mac's listings and HEAD checks use a 5 s socket timeout (`MAC_CHECK_TIMEOUT_SECONDS`), and each HEAD is a single request with no retry. An unreachable R2 still delays the pull: the connection is tried on every address the endpoint resolves to, each with the whole timeout, so each milestone checked costs up to 5 s per address, about 40 s for run A's two milestones with four addresses, and name resolution has no timeout at all. Each milestone is checked once: one already pulled, or one whose check failed, goes straight to the pull when a dropped pull is retried in the next 5 s cycle.

With none at risk, the run gets the 60-second grace (its own drain is already over); then its report and a list of the labels that reached R2 are archived under `diagnostics/`, and teardown is approved. With any at risk there is no grace timer. The supervisor pulls each one, the last stage's first, from `exports/<label>/` (`checkpoint.pt` and `record.json`) to `$CTRL/supervisor/recovered/<pod id>/<label>/` with the pilot's verified pull (`pilot_transfer.pull_artifacts`: the Pod's own manifest, then size and SHA-256 of every streamed file). Each pull must end by the end of the transfer window, 900 s after the heartbeat's last GPU work. The worker's final heartbeat arrives by 480 s, so the pull has the 420 s reserve; while its bytes grow, the supervisor's mirror stamps `transfer_progress_unix` every 5 s, so the watchdog waits that long and no longer. The supervisor then checks the pulled checkpoint against the SHA-256 the report and `finalization.json` record, archives the report and a summary (with `milestone_recovery`: each pulled file's path, SHA-256 and size, and each milestone the Mac verified in R2) under `diagnostics/`, and only then approves teardown. A failed or mismatched pull is retried while the window lasts; it writes `MILESTONES-NOT-SECURED.json` and `controller-error.json` naming each file on the Pod and its SHA-256, prints a banner, and never approves teardown. The watchdog's idle, transfer, wall and cost limits then end the Pod. `updates.jsonl` and every checkpoint that reached R2 stay there; GitHub only holds the small records.

## Projected fit

`uv run --locked python -m tools.pilot_feasibility rehearsal` with the S build's counts and its default assumptions (startup 20 minutes, a 120-second compile per stage, forward-only validation at twice the training rate, 15 seconds per checkpoint save):

| Run | Updates | Reference 56,479 tok/s | Floor 50,831 tok/s |
| --- | ---: | --- | --- |
| A (stage1, stage2) | 15,260 | 11.0 h total; fits with 339 min to spare | 12.1 h total; fits with 270 min to spare (startup up to 290.2 min) |
| B (mixed) | 15,259 | 10.8 h total; fits with 348 min to spare | 11.9 h total; fits with 281 min to spare (startup up to 300.6 min) |

These projections use their own assumptions and are not the launch gate; the launch rule above is. At the quoted $0.60/h the 60,000 s wall falls at the $10.00 stop, so the cap is the binding limit; at a higher rate the cost stop ends the Pod first, and the receipt refuses any Pod costing more than the $0.60/h quote. Each Pod needs 16 to 18 GB of its 40 GB container disk with headroom, and the bucket holds about 16 GB ($0.09 a month). These are projections from one measured pilot, not rehearsal measurements: a run near the floor speed whose setup took long will be stopped by the 900-second projected-finish rule.

## Verification

`uv run --locked pytest tests/test_rehearsal_run.py tests/test_rehearsal_integration.py tests/test_rehearsal_runbook.py tests/test_rehearsal_supervisor.py tests/test_rehearsal_ops.py tests/test_rehearsal_bootstrap.py tests/test_rehearsal_readiness.py tests/test_rehearsal_feasibility.py tests/test_rehearsal_watchdog.py` runs offline. The tests cover:

- every tool command in the runbook, parsed by the real parsers, `write-initialization` with both specifications, and the policy file against the watchdog command line;
- the runbook's own shell run in bash and zsh against stubs: the Pod ID taken from the create response and the refusals that name that Pod, the bounded wait for RUNNING, the step 7 wait that never launches a second bootstrap and stops when none runs, the fresh timer receipt before the readiness receipt, and the step 13 DELETE that is sent only on `shutdown_ready` or `TERMINATE=yes`;
- a receipt built by `build_rehearsal_receipt` from fixture evidence and accepted by the worker's `check_readiness` and its `run` command line, and refused for a tampered specification, binding, preflight or initialization hash;
- the worker's CPU smoke for both runs replayed heartbeat by heartbeat, with its `report.json` and in-memory bucket, into the real supervisor over fake SSH and GitHub: no teardown at run A's stage boundary or while uploading, `shutdown_ready` only after the R2 verification and archive, and never after a mismatched R2 SHA-256;
- the supervisor with fake SSH, R2 and GitHub, including the R2-observed progress stamps, and the watchdog's transfer window under a rehearsal policy;
- an unverified milestone pulled over a fake SSH pull, checked and archived before teardown, with no grace timer; a failed, tampered, late or unconfigured pull that names the files and never approves teardown while the watchdog still ends the Pod at the window; an R2 that never answers the Mac, each HEAD taking the socket timeout on four addresses, with the first pull's bytes still arriving before the idle limit and a dropped pull retried in the next cycle without checking R2 again; `r2_finalization` with every milestone checked in R2 by the Mac; strict parsing of `finalization.json`;
- a reused controller directory's old mirrored heartbeat (another Pod's ID, or older than this Pod) that never deletes the new Pod and is never mirrored;
- the bootstrap run end to end against stub programs: the timer first, the 60,000-second ceiling, the 0600 key file, the Pod key never printed, the compile check, the on-Pod test list, the pilot path unchanged;
- receipts refused for each missing gate, weak limit, missing compile check, wrong transfer window or billing origin, a specification that differs from D-35 or the binding, a small or slow round trip, a stale self-destruct receipt or one without a pid, a captured cost above the quote, and a dirty or malformed revision; the round trip's measured rates, the feasibility arithmetic and the watchdog PID file.

## Limitations

- No rehearsal run has been measured; every time and cost figure above is a projection.
- Termination is layered best-effort (in-Pod timer, Mac watchdog, supervisor guard, worker deadlines); Runpod offers no provider-enforced Pod lifetime.
- The Mac must stay awake and online for the whole run: a sleeping Mac stops the heartbeat mirror, and on waking the watchdog would read a stale heartbeat and delete the Pod.
- R2 progress is seen only when an object completes, so a slow final upload shows the Mac no R2 progress until it ends; one still unfinished at 360 s is left to the SSH pull. The receipt's 9.17 MB/s requirement is measured once, before training; R2 may be slower at the end.
- If nothing reaches R2 during the final drain (an R2 outage), the worker still ends its drain by 360 s and writes its final heartbeat by 480 s after its last GPU work, before the 600 s idle limit. From then on only the supervisor's SSH pull, whose growing bytes stamp progress, keeps the Pod, and never past 900 s.
- The SSH fallback has the 420 s reserve, from 480 s to 900 s after the last GPU work: one 1.65 GB label needs about 4.5 MB/s from Pod to Mac in about 370 s of it, two labels about 9 MB/s. The pilot's receipt times bound one 1.62 GB pull at 150 s or less, about 10.7 MB/s or faster. A slower pull fails, is reported in `MILESTONES-NOT-SECURED.json`, and the Pod is deleted with that copy.
