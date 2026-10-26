# Rehearsal run operations (D-35)

The owner approved two rehearsal pretraining runs of the S model (decision D-35), each on its own Runpod A40 Pod at the same time. **Run A** trains `stage1` on English and then `stage2` on Python plus unseen English, with a fresh optimizer and schedule from stage 1's final weights. **Run B** trains one `mixed` stage on the same tokens. Both start from one shared saved initialization. This page describes the Mac-side and Pod-side operations around the training worker; the step-by-step commands are in the [rehearsal runbook](../REHEARSAL_RUNBOOK.md). Nothing here launches a Pod by itself, and no rehearsal run has happened yet.

## Approved limits per run

| Limit | Value |
| --- | --- |
| Pod wall | 45,000 s (12.5 h): watchdog `--mode rehearsal`, self-destruct `--max-lifetime-seconds 45000`, armed 180 s inside the wall |
| Cost | cap $10; estimate $6.2917 x 1.2 = worker stop $7.55; Mac watchdog stop $7.54 |
| Idle | 600 s without GPU activity |
| Speed and fit | after 900 s: stop below 0.85 x 56,479 tok/s, or when the projected finish passes the armed deadline |
| Quote | $0.59/h GPU, $0.60/h with the 40 GB container disk ([Pod body](../configs/runpod-pod-a40-s-rehearsal.json)); at most $7.50 per Pod, $15.00 for both |

## What runs where

| Piece | Where | What it does |
| --- | --- | --- |
| `tools/rehearsal_ops.py` | Mac and Pod | The D-35 constants and the R2 run layout. `pod-env` writes the Pod's R2 key file to a pipe; `round-trip` tests PUT, HEAD, GET and DELETE under the run's prefix; `fetch-object` downloads one pinned object (the initialization); `verify-final` checks a finished run in R2. |
| `tools/pilot_pod_bootstrap.sh --mode rehearsal` | Pod | Arms the self-destruct timer first, then loads the 0600 R2 key file without printing it, records the environment, installs the project, runs the R2 round trip, fetches and re-verifies the run's pinned data and the initialization, runs the CPU tests including every `tests/test_rehearsal_*.py`, and runs the A40 preflight. The pilot path is unchanged. |
| `tools/pilot_readiness_receipt.py rehearsal` | Mac | Builds one receipt per run from the Pod's own receipts and the owner's confirmations; refuses to write if any gate is false or the time left cannot hold the run at the reference speed. |
| `tools/pilot_feasibility.py rehearsal` | Mac | Projects time, cost, container disk and R2 storage for both runs at the reference speed and at the 85% floor. |
| `tools/runpod_watchdog.py --pidfile` | Mac | One watchdog per run; each writes its own PID file so one run's watchdog can be restarted without touching the other's. |
| `tools/pilot_supervisor.py --mode rehearsal` | Mac | One supervisor per run. It mirrors the heartbeat, guards the Pod through the API, reads `heartbeat.json` and `report.json` over SSH with bounded sizes, never copies a checkpoint over SSH, and approves teardown only after the final state is verified in R2 and archived. |

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

The worker's heartbeat stays `running` through every stage, including run A's boundary between its two stages. At the very end the worker drains its uploader, verifies the final milestone and `report.json` in R2, and only then writes `awaiting_transfer` with `r2_final`: the report and checkpoint keys and their SHA-256s. The rehearsal supervisor then, with the Mac's own key:

1. checks that the SSH copy of `report.json` hashes to `r2_final.report_sha256`;
2. HEADs the final checkpoint, `report.json` and the final label's other files, and compares the stored SHA-256 metadata;
3. downloads `report.json` pinned to that hash and size;
4. checks that the report closes the run: its stages are the run's stages in order, a completed run A lists both, the last stage's final checkpoint is the heartbeat's, and every earlier stage's milestone is in R2 with the recorded hash;
5. archives `report.json` and a small `summary.json` to `rehearsal-runs/<run>/<pod id>/final/` in the private artifacts repository, reading each back byte for byte;
6. only then marks `shutdown_ready`, which lets the watchdog and its own guard delete the Pod.

Any mismatch stops at its step and is retried each cycle; it never approves teardown, and the idle limit still ends the Pod 600 seconds after its last GPU activity. A worker that reports `stopped` or `failed` without `r2_final` gets a 300-second grace for its uploader, then its own report and a list of the labels that reached R2 are archived under `diagnostics/`, and teardown is approved. `updates.jsonl` and every checkpoint stay in R2; GitHub only holds the small records.

## Projected fit

`uv run --locked python -m tools.pilot_feasibility rehearsal` with the S build's counts and its default assumptions (startup 20 minutes, a 120-second compile per stage, forward-only validation at twice the training rate, 15 seconds per checkpoint save):

| Run | Updates | Reference 56,479 tok/s | Floor 48,007 tok/s |
| --- | ---: | --- | --- |
| A (stage1, stage2) | 15,260 | 10.8 h total; fits with 100 min to spare | 12.6 h total; **does not fit** by 7 min; fits only if startup takes at most 12.7 min |
| B (mixed) | 15,259 | 10.7 h total; fits with 104 min to spare | 12.5 h total; **does not fit** by 3 min; fits only if startup takes at most 17.1 min |

At the quoted $0.60/h each Pod costs at most $7.50 at its wall, inside the $7.55 stop; a quote of $0.61/h GPU ($0.62/h total) would put the wall above it. Each Pod needs 16 to 18 GB of its 40 GB container disk with headroom, and the bucket holds about 16 GB ($0.09 a month). These are projections from one measured pilot, not rehearsal measurements: a run at exactly the floor speed may be stopped by the 900-second speed-and-fit check, which projects from the speed it measures.

## Verification

`uv run --locked pytest tests/test_rehearsal_runbook.py tests/test_rehearsal_supervisor.py tests/test_rehearsal_ops.py tests/test_rehearsal_bootstrap.py tests/test_rehearsal_readiness.py tests/test_rehearsal_feasibility.py tests/test_rehearsal_watchdog.py` runs offline. The tests cover:

- every tool command in the runbook, parsed by the real parsers (the worker's against the shared contract until the runner lands), and the policy file against the watchdog command line;
- the supervisor with fake SSH, R2 and GitHub: a stage boundary never tears down, stage 1's final never completes run A, any R2 or SSH mismatch never sets `shutdown_ready`, and the happy path sets it only after the archive;
- the bootstrap run end to end against stub programs: the timer first, the 45,000-second ceiling, the 0600 key file, the Pod key never printed, the pilot path unchanged;
- receipts refused for each missing gate or weak limit, the feasibility arithmetic, and the watchdog PID file.

## Limitations

- No rehearsal run has been measured; every time and cost figure above is a projection.
- Termination is layered best-effort (in-Pod timer, Mac watchdog, supervisor guard, worker deadlines); Runpod offers no provider-enforced Pod lifetime.
- The Mac must stay awake and online for the whole run: a sleeping Mac stops the heartbeat mirror, and on waking the watchdog would read a stale heartbeat and delete the Pod.
- The worker's own `rehearsal_run` command line, `smoke` and `write-initialization` flags follow the shared contract; the runbook test parses them with the real parser once that module exists.
