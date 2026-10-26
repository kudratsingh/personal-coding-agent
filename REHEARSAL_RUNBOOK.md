# Rehearsal runbook: two concurrent A40 pretraining runs (D-35)

Operational sequence for the two owner-approved rehearsal runs of the S model on Runpod A40s: **run A** trains `stage1` (stream `stage1-web`, peak 6e-4) and then `stage2` (`stage2-python` and `stage2-web`, peak 3e-4, fresh optimizer and schedule, weights from stage 1's final checkpoint); **run B** trains one `mixed` stage on all three streams at peak 6e-4. Both start from one shared saved initialization. Each run has its own Pod, controller directory, watchdog, supervisor and known-hosts file; the runs share only the data binding, the initialization and the R2 bucket.

Commands marked **Mac** run on the owner's Mac inside the repository checkout; **Pod** commands run inside a rented Pod over SSH. Open **two terminals**, one per run. Every per-run command below is written once and uses the variables set in step 2 of its own terminal, so the same text runs for A in terminal 1 and for B in terminal 2. Commands marked **once** run in one terminal only.

Nothing in steps 1 to 4 starts a Pod. Step 4 stores the initialization in R2 (a few cents a month). Step 5 creates the Pods and starts billing; do not execute it without the owner's explicit launch authorization for these runs. Runpod offers no provider-enforced Pod lifetime: termination is layered best-effort (in-Pod timer, Mac watchdog, supervisor guard, the worker's own deadlines).

**Approved limits per run (D-35).** Pod wall 45,000 s (12.5 h); approved cost cap $10; estimate $6.2917 x 1.2 = worker cost stop **$7.55**; Mac watchdog cost stop **$7.54**, one cent inside the worker's; idle 600 s; a 900 s speed-and-fit check (stop if post-compile throughput is below 0.85 x 56,479 tok/s, or if the projected finish plus remaining validations and the 900 s finalization reserve passes the armed deadline); rising-loss and nonfinite stops as in the pilot. Quoted $0.59/h GPU, $0.60/h with the 40 GB container disk: at most $7.50 per Pod at its wall, **$15.00 for both**.

## 1. Prelaunch checks (Mac, free, once)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check && make smoke
uv run --locked python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke
uv run --locked python -m tools.pilot_preflight --output runs/rehearsal-preflight-cpu --device cpu
uv run --locked pytest tests/test_rehearsal_runbook.py tests/test_rehearsal_supervisor.py tests/test_rehearsal_ops.py tests/test_rehearsal_bootstrap.py tests/test_rehearsal_readiness.py tests/test_rehearsal_feasibility.py tests/test_rehearsal_watchdog.py tests/test_runpod_watchdog.py tests/test_pod_self_destruct.py -q
```

Expected: `make check` green; the smoke completes offline on CPU; the preflight report shows `"status": "completed"` on `cpu`; every listed test passes. The runbook test parses every tool command in this file against the real parsers.

## 2. Credentials and per-run environment (Mac, free, each terminal)

`.env` is git-ignored. It holds `RUNPOD_API_KEY`, `R2_ENDPOINT`, `R2_BUCKET`, the Mac's own R2 key `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` (stages data, uploads the initialization, verifies results) and the Pods' separate key `R2_POD_ACCESS_KEY_ID`/`R2_POD_SECRET_ACCESS_KEY`. The Pod key must be its own Cloudflare API token with Object Read & Write on this one bucket only; it reaches the Pods only over SSH stdin (step 6), never in a create body, and it is revoked after both runs (step 13).

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi   # names only, never values
done
gh auth status                                   # the private archive of reports and summaries
caffeinate -dimsu -w $$ &                        # keep the Mac awake while this terminal lives; stay on AC power
export RUN=A                                     # terminal 1: A; terminal 2: B
export RUN_LOWER=$(printf %s "$RUN" | tr AB ab)
export CTRL=runs/rehearsal-$RUN_LOWER-controller && mkdir -p "$CTRL"
export KNOWN_HOSTS=~/.ssh/rehearsal_${RUN_LOWER}_known_hosts   # one per Pod; no whitespace: ssh splits the option on spaces
export SPEC=configs/rehearsal-a40-s-$RUN_LOWER.json
export SHARED=runs/rehearsal-shared && mkdir -p "$SHARED"
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts   # must be private; the supervisor verifies this
export BUILD_DIR=runs/rehearsal-s-20261025                        # the admitted S build
export BINDING=runs/rehearsal-binding/binding.json                # the reviewed binding; never rebuilt here
export BINDING_SHA256=$(shasum -a 256 "$BINDING" | cut -d' ' -f1) && echo "$BINDING_SHA256"
```

A sleeping Mac stops the mirror: when it wakes, the watchdog would read a stale heartbeat and delete the Pod. Keep both terminals open on AC power for the whole run. The working directory must not sit under a symlinked path (macOS `/tmp` is one).

## 3. Final quote check for both Pods (Mac, free, once)

Open the Runpod console: Secure Cloud, NVIDIA A40, on-demand, 1 GPU, container disk 40 GB, no volume. Check that **two** such Pods are available and record the hourly price. Then project both runs at that price:

```sh
uv run --locked python -m tools.pilot_feasibility rehearsal --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --quoted-gpu-hourly 0.59 --output "$SHARED/feasibility.json"
```

Replace `0.59` with the console price. If it exceeds $0.59, stop and ask the owner: D-35 was approved at that quote. The report gives, per run, the time at the reference speed and at the 85% floor, the latest startup that still fits the 45,000-second wall at the floor, the cost against the $7.55 stop, the container-disk fit and the R2 storage price. Confirm the account balance covers both runs to their walls (at least $16) and that no network volume exists.

## 4. Data, binding and the shared initialization (Mac, once; R2 storage only)

Re-stage the binding: every pinned file is re-hashed locally and skipped in R2 when already stored with the same SHA-256, and the binding goes last:

```sh
uv run --locked python -m tools.rehearsal_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --build "$BUILD_DIR" --receipt "$SHARED/stage-receipt.json"
```

Write the shared initialization (seed 0) once, pin its hash and size, and upload it under a key named by its hash:

```sh
uv run --locked python -m tools.rehearsal_run write-initialization --specification configs/rehearsal-a40-s-a.json --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --output "$SHARED/init/rehearsal-init.pt"
export INIT_SHA256=$(shasum -a 256 "$SHARED/init/rehearsal-init.pt" | cut -d' ' -f1)
export INIT_SIZE=$(wc -c < "$SHARED/init/rehearsal-init.pt" | tr -d ' ')
export INIT_KEY=rehearsal-init/$INIT_SHA256/rehearsal-init.pt
uv run --locked python -m tools.r2_store upload --root "$SHARED/init" --prefix "rehearsal-init/$INIT_SHA256/" --receipt "$SHARED/init-upload.json" "$SHARED/init/rehearsal-init.pt"
printf 'export INIT_SHA256=%s INIT_SIZE=%s INIT_KEY=%s\n' "$INIT_SHA256" "$INIT_SIZE" "$INIT_KEY" > "$SHARED/init.env"
```

The SHA-256 printed by `write-initialization` must equal `INIT_SHA256`. In the other terminal, `source "$SHARED/init.env"`. Both runs start from these exact bytes; each Pod fetches them pinned to this hash and size.

## 5. Pod creation, identity and the watchdog (Mac, BILLABLE, each terminal)

Only after explicit launch authorization. Capture the billing origin before the create request so every deadline is conservative. The body is `configs/runpod-pod-a40-s-rehearsal.json` (A40, Secure Cloud, template `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, 40 GB container disk, TCP 22, no volume, no environment variables) with the run's letter appended to the name:

```sh
export ORIGIN=$(date +%s)
python3 -c 'import json,sys; b=json.load(open("configs/runpod-pod-a40-s-rehearsal.json")); b["name"]+="-"+sys.argv[1]; print(json.dumps(b))' "$RUN_LOWER" > "$CTRL/create-body.json"
curl -sS -X POST https://api.runpod.io/v2/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  --data @"$CTRL/create-body.json" | tee "$CTRL/create-response.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'
export POD_ID=<id printed above>
export SELF_DESTRUCT_DEADLINE=$((ORIGIN + 45000 - 180))      # 3 minutes inside the 45,000-second wall
printf 'export ORIGIN=%s POD_ID=%s SELF_DESTRUCT_DEADLINE=%s\n' "$ORIGIN" "$POD_ID" "$SELF_DESTRUCT_DEADLINE" > "$CTRL/run.env"
```

A reopened terminal repeats step 2 for its run and then runs `source "$SHARED/init.env"; source "$CTRL/run.env"`. Capture the identity the guards enforce, write the policy, define this terminal's watchdog helper and start it at once:

```sh
uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" \
  --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json"
cat > "$CTRL/policy.json" <<EOF
{"mode": "rehearsal", "max_wall_seconds": 45000, "max_cost_usd": 7.54, "max_idle_seconds": 600,
 "quoted_total_hourly": 0.60, "max_transfer_seconds": 0, "billing_started_at_unix": $ORIGIN,
 "idle_baseline_unix": $(date +%s)}
EOF
watchdog() {   # (re)start THIS run's watchdog with a fresh idle baseline; the other run's is untouched
  OLD=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
  BASE=$(date +%s)
  nohup uv run --locked python -m tools.runpod_watchdog --mode rehearsal --identity "$CTRL/pod-identity.json" \
    --heartbeat "$CTRL/supervisor/heartbeat.json" --receipt "$CTRL/watchdog-receipt.json" \
    --pidfile "$CTRL/watchdog.pid" --max-wall-seconds 45000 --max-idle-seconds 600 \
    --max-cost-usd 7.54 --quoted-total-hourly 0.60 --billing-started-at-unix "$ORIGIN" \
    --idle-baseline-unix "$BASE" --poll-seconds 20 \
    >> "$CTRL/watchdog.log" 2>&1 < /dev/null &
  NEW=""
  for _ in $(seq 1 60); do
    NEW=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
    if [ -n "$NEW" ] && [ "$NEW" != "$OLD" ]; then break; fi
    sleep 1
  done
  if [ -z "$NEW" ] || [ "$NEW" = "$OLD" ]; then echo "run $RUN: the new watchdog did not start; see $CTRL/watchdog.log"; return 1; fi
  if [ -n "$OLD" ] && ps -p "$OLD" -o command= | grep -qF -- "--identity $CTRL/pod-identity.json"; then kill "$OLD"; fi
  echo "run $RUN watchdog pid $NEW, idle baseline $BASE"
}
watchdog
```

Each watchdog writes its own PID to `$CTRL/watchdog.pid` and removes it when it exits. The helper starts the replacement first, then stops only the process whose PID the file held and whose command line names this run's identity file; there is no global `pkill`, so restarting run A's watchdog never touches run B's. The watchdog reads the heartbeat the supervisor mirrors to `$CTRL/supervisor/heartbeat.json`, which exists only once the supervisor runs (step 10). Until then the idle clock runs from the last baseline: run `watchdog` again at each setup milestone below, and at least every 5 minutes while waiting. Wall and cost limits never pause, and 600 seconds without a restart still deletes the Pod.

Wait until `pod-connection.json` shows `"status": "RUNNING"` and a non-null `ssh_direct` (re-run the capture until it does). If `ssh_proxy_only` is true the Pod has no public IP: terminate it (step 13) and create it again. Pin this Pod's host key in this run's own known-hosts file:

```sh
export POD_HOST=$(python3 -c 'import json; print(json.load(open("'"$CTRL"'/pod-connection.json"))["ssh_direct"]["host"])')
export POD_PORT=$(python3 -c 'import json; print(json.load(open("'"$CTRL"'/pod-connection.json"))["ssh_direct"]["port"])')
printf 'export POD_HOST=%s POD_PORT=%s\n' "$POD_HOST" "$POD_PORT" >> "$CTRL/run.env"
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS" 2>/dev/null
ssh-keygen -lf "$KNOWN_HOSTS"      # compare with the fingerprint shown in this Pod's web terminal
alias podssh="ssh -F /dev/null -p $POD_PORT -i $POD_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o BatchMode=yes root@$POD_HOST"
alias podscp="scp -F /dev/null -P $POD_PORT -i $POD_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o BatchMode=yes"
watchdog
```

A host key that changes later means another machine answers at that address: stop and terminate the Pod rather than accepting it.

## 6. Code, binding and the Pod's R2 key (Mac to Pod, billable, each terminal)

Stage the exact audited commit and the binding, then deliver the Pod's R2 key over SSH stdin into a 0600 file. `pod-env` refuses to write to a terminal and refuses the Mac's own key, so the values never appear on screen, in a process list or in a create body:

```sh
git archive --format=tar.gz -o "$CTRL/repo.tar.gz" HEAD
podssh 'mkdir -p /workspace/repo /workspace/rehearsal /workspace/rehearsal-data'
podscp "$CTRL/repo.tar.gz" root@$POD_HOST:/workspace/repo.tar.gz
podscp "$BINDING" root@$POD_HOST:/workspace/rehearsal/binding.json
uv run --locked python -m tools.rehearsal_ops pod-env | podssh 'umask 077 && cat > /workspace/rehearsal/r2-pod.env && chmod 600 /workspace/rehearsal/r2-pod.env'
podssh 'stat -c "%a %U %s" /workspace/rehearsal/r2-pod.env'     # expect 600 root and a nonzero size
watchdog
```

## 7. Bootstrap (Pod, billable, each terminal)

The bootstrap arms the in-Pod self-destruct timer first, before anything else can fail. In rehearsal mode it runs exactly this command, detached:

```sh
python3 tools/pod_self_destruct.py --pod-id "$RUNPOD_POD_ID" --deadline-unix "$SELF_DESTRUCT_DEADLINE" \
  --receipt /workspace/rehearsal/self-destruct.json --stop-file /workspace/rehearsal/TERMINATE_NOW --max-lifetime-seconds 45000
```

It then loads the R2 key file (a regular 0600 file, names printed, values never), refuses to continue if the Mac's key variables are set on the Pod, records the environment, installs the project against the image's CUDA torch, tests an R2 round trip under `rehearsal-runs/<run>/<pod id>/`, fetches this run's pinned shards, manifests and tokenizer with `tools.rehearsal_binding fetch --key-prefix R2_POD_` and re-verifies them, fetches the initialization pinned to `INIT_SHA256` and `INIT_SIZE`, runs the CPU tests (the pilot's list plus the one-pass loader, validation, R2 and every `tests/test_rehearsal_*.py`), and runs the A40 precision, fused-attention and exact-resume preflight. Launch it detached so an SSH drop cannot kill it, and keep re-baselining this run's watchdog while it runs, for at most 45 minutes:

```sh
podssh "cd /workspace/repo && tar xzf /workspace/repo.tar.gz && setsid nohup bash -c 'bash tools/pilot_pod_bootstrap.sh --mode rehearsal --run $RUN --deadline-unix $SELF_DESTRUCT_DEADLINE --workspace /workspace/rehearsal --repo /workspace/repo --r2-env /workspace/rehearsal/r2-pod.env --binding /workspace/rehearsal/binding.json --binding-sha256 $BINDING_SHA256 --data-root /workspace/rehearsal-data --initialization-key $INIT_KEY --initialization-sha256 $INIT_SHA256 --initialization-size $INIT_SIZE; echo BOOTSTRAP_EXIT=\$?' > /workspace/rehearsal/bootstrap.log 2>&1 < /dev/null &"
GIVE_UP=$(( $(date +%s) + 2700 ))
until podssh 'grep -q ^BOOTSTRAP_EXIT= /workspace/rehearsal/bootstrap.log'; do
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: bootstrap exceeded 45 minutes; terminate the Pod (step 13)"; break; fi
  sleep 30; if [ $(( $(date +%s) - BASE )) -ge 300 ]; then watchdog; fi
done
for name in bootstrap.log self-destruct.json environment.json r2-round-trip.json data-fetch.json data-verification.json initialization-fetch.json; do
  podscp root@$POD_HOST:/workspace/rehearsal/$name "$CTRL/$name"
done
podscp root@$POD_HOST:/workspace/rehearsal/preflight-a40/report.json "$CTRL/preflight-a40-report.json"
grep BOOTSTRAP_EXIT "$CTRL/bootstrap.log"; watchdog
```

Expected: `BOOTSTRAP_EXIT=0`; `self-destruct.json` shows `"state": "armed"` and `deadline_unix` equal to `SELF_DESTRUCT_DEADLINE`; `r2-round-trip.json` shows `"status": "passed"`; `preflight-a40-report.json` shows `"status": "completed"`, `"hardware_precision_and_resume_checked": true` and a passed `fused_attention`. Any other exit code: read `bootstrap.log`, then terminate the Pod (step 13) unless the fix is obvious and cheap. `bootstrap.log` never contains R2 values; check with `grep -c "$R2_POD_SECRET_ACCESS_KEY" "$CTRL/bootstrap.log"` (expect 0).

## 8. Readiness receipt (Mac, billable, each terminal)

The confirmations file is the owner's explicit statement. `termination_guarantee` must be `redundant_best_effort`. `r2_token_bucket_scoped` states that the Pod key is a separate token limited to this bucket. The round-trip and data gates are not typed by hand: the receipt tool derives them from the Pod's own receipts and refuses to write if any gate is false, the timer is not armed, a limit is already at risk, or the lifetime left cannot hold the run at the reference speed:

```sh
cat > "$CTRL/confirmations.json" <<'EOF'
{"owner_confirms_four_runpod_steps": true, "cloud_data_pre_staged": true,
 "external_watchdog_armed": true, "hardware_preflight_passed": true, "full_console_quote_confirmed": true,
 "staging_storage_cost_bound": true, "r2_token_bucket_scoped": true,
 "termination_guarantee": "redundant_best_effort", "owner_accepts_best_effort_termination": true}
EOF
uv run --locked python -m tools.pilot_readiness_receipt rehearsal --run "$RUN" --specification "$SPEC" \
  --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --preflight "$CTRL/preflight-a40-report.json" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --self-destruct-receipt "$CTRL/self-destruct.json" --confirmations "$CTRL/confirmations.json" \
  --data-fetch "$CTRL/data-fetch.json" --data-verification "$CTRL/data-verification.json" \
  --round-trip "$CTRL/r2-round-trip.json" --initialization-fetch "$CTRL/initialization-fetch.json" \
  --initialization-sha256 "$INIT_SHA256" --billing-origin-unix "$ORIGIN" --output "$CTRL/readiness-receipt.json"
podscp "$CTRL/readiness-receipt.json" root@$POD_HOST:/workspace/rehearsal/readiness-receipt.json
watchdog
```

The four Runpod steps are: funded balance verified for both runs, SSH public key registered, API key with Pod permissions created, console quote recorded. The printed line reports `fits_at_reference` and `fits_at_floor` for the time actually left; `fits_at_floor: false` means the 900-second speed-and-fit check may stop the run if it trains near the floor.

## 9. Worker launch (Pod, billable, each terminal)

Start the worker detached from the SSH session, with the R2 key loaded from the 0600 file and nothing secret on the command line:

```sh
podssh "setsid nohup bash -c 'set -a; . /workspace/rehearsal/r2-pod.env; set +a; cd /workspace/repo && exec python3 -m tools.rehearsal_run run \
  --specification $SPEC --binding /workspace/rehearsal/binding.json --binding-sha256 $BINDING_SHA256 \
  --root /workspace/rehearsal-data --initialization /workspace/rehearsal/initialization.pt \
  --initialization-sha256 $INIT_SHA256 --readiness /workspace/rehearsal/readiness-receipt.json \
  --preflight /workspace/rehearsal/preflight-a40/report.json --output /workspace/rehearsal-run' \
  > /workspace/rehearsal/run.log 2>&1 < /dev/null &"
# Wrapping in bash -c with the redirect outside keeps no SSH channel open.
until podssh 'test -s /workspace/rehearsal-run/heartbeat.json'; do sleep 10; done
```

## 10. Supervisor (Mac, billable, each terminal; second window per run)

Start this run's supervisor as soon as the worker's heartbeat exists. It mirrors the heartbeat to `$CTRL/supervisor/heartbeat.json` (from then on the watchdog needs no more re-baselining), guards the Pod through the API, reads `heartbeat.json` and `report.json` over SSH with bounded sizes, and never copies a checkpoint over SSH. In the second window repeat step 2 for this run and `source "$SHARED/init.env"; source "$CTRL/run.env"`, then:

```sh
uv run --locked python -m tools.pilot_supervisor --mode rehearsal --run "$RUN" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --identity-file "$POD_SSH_KEY" --known-hosts "$KNOWN_HOSTS" --output "$CTRL/supervisor" \
  --host root@$POD_HOST --port $POD_PORT --remote-root /workspace/rehearsal-run \
  --repository "$PRIVATE_REPO" --poll-seconds 20 2>&1 | tee -a "$CTRL/supervisor.log"
```

It refuses to start without the Mac's own R2 key or a `"mode": "rehearsal"` policy, and it mirrors one heartbeat before its API guard starts. If it stops (Mac crash, closed window), run the same command again: it continues in the same directory. Two watchdogs and two supervisors at 20-second polls make 12 API requests a minute.

## 11. Monitoring both runs (Mac, read-only)

```sh
python3 -c 'import json; h=json.load(open("'"$CTRL"'/supervisor/heartbeat.json")); print(h["status"], h["phase"], h["step"], h.get("r2_final"))'
podssh 'tail -n 2 /workspace/rehearsal/run.log; python3 -c "import json; r=json.load(open(\"/workspace/rehearsal-run/report.json\")); print(r.get(\"status\"), r.get(\"speed_check\"), [(s.get(\"name\"), s.get(\"updates\")) for s in r.get(\"stages\", [])])"'
tail -n 3 "$CTRL/watchdog.log"; cat "$CTRL/watchdog-receipt.json"
```

Expect `running` with a growing `step` (the global update count). The speed-and-fit check reports after 900 seconds of training. Validation runs every 500 global updates, and run A's stage 2 continues A's count. At run A's stage boundary the worker validates both streams in full, saves `stage1-final` and uploads it to R2, then starts stage 2 with a fresh optimizer and warmup; the heartbeat stays `running` throughout and the supervisor does nothing. Checkpoints are saved every 1,800 seconds and uploaded in the background as `<stage>-step<global update>` labels under `rehearsal-runs/<run>/<pod id>/`.

## 12. Completion and verification (Mac, automatic)

At the end the worker drains its uploader, uploads the final milestone `<stage>-final` (run A: `stage2-final`; run B: `mixed-final`) and `report.json`, checks both in R2 by HEAD, and only then writes `awaiting_transfer` with `r2_final` (the report and checkpoint keys and SHA-256s) into its heartbeat. The supervisor then, with the Mac's own key: checks that the SSH copy of `report.json` hashes to `r2_final`; HEADs the final checkpoint, `report.json` and the final label's other files and compares the stored SHA-256 metadata; downloads `report.json` pinned to its hash and size; checks that the report closes this run (both stages for a completed run A, so stage 1's end never counts) and that every stage-final milestone is in R2 with the recorded hash; archives `report.json` and a small `summary.json` to `rehearsal-runs/<run>/<pod id>/final/` in `$PRIVATE_REPO` with byte-exact read-back; and only then marks `shutdown_ready`, which lets the guards delete the Pod and confirm a 404. `updates.jsonl` (the per-update loss curve) and every checkpoint stay in R2.

A mismatch never approves teardown: the supervisor writes `controller-error.json` and retries each cycle. The idle limit still deletes the Pod 600 seconds after its last GPU activity, so act within that window. To check by hand (also when the supervisor is down):

```sh
uv run --locked python -m tools.rehearsal_ops verify-final --run "$RUN" --pod-id "$POD_ID" \
  --report-sha256 <r2_final.report_sha256> --checkpoint-sha256 <r2_final.checkpoint_sha256> --output "$CTRL/manual-verification"
```

## 13. Pod termination (Mac, each terminal)

The guards delete the Pod after `shutdown_ready`; the in-Pod timer deletes it at `SELF_DESTRUCT_DEADLINE` regardless. Confirm, or terminate by hand:

```sh
curl -sS -o /dev/null -w '%{http_code}\n' -X DELETE "https://api.runpod.io/v2/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"   # expect 204
uv run --locked python -m tools.runpod_pod_identity status --pod-id "$POD_ID"    # expect "resource not found (HTTP 404)"
kill "$(cat "$CTRL/watchdog.pid")" 2>/dev/null || true                         # this run's watchdog only, if still running
```

After both Pods are gone, revoke the Pods' R2 token in the Cloudflare dashboard and remove `R2_POD_*` from `.env`. Check under Storage that the account holds no network volume.

## 14. If one run fails

The runs are independent: never stop, restart or re-baseline the other run's watchdog or supervisor because one failed, and never reuse a Pod for the other run.

- **Bootstrap or receipt fails.** Read `$CTRL/bootstrap.log` or the receipt tool's message, terminate that Pod (step 13) and report. A new Pod for the same run is a new launch: ask the owner first. The initialization and binding are unchanged and can be reused.
- **The worker stops or fails** (speed-and-fit check, rising loss, nonfinite loss, cost, a crash). The heartbeat shows `stopped` or `failed`. If the worker still published `r2_final`, the supervisor verifies it as in step 12 and archives it. Otherwise it waits 300 seconds (`--failure-grace-seconds`) for the uploader, then archives the worker's own `report.json` and a summary listing the labels present in R2 to `.../diagnostics/`, and approves teardown. The last verified rolling checkpoint stays in R2.
- **The supervisor reports an R2 mismatch.** It never approves teardown. Inspect with `verify-final` and `uv run --locked python -m tools.r2_store status`; the idle limit deletes the Pod 600 seconds after its last GPU activity.
- **A Pod disappears or a watchdog exits.** Read `$CTRL/watchdog-receipt.json` and `$CTRL/supervisor/watchdog-receipt.json`, then confirm with the `status` command in step 13. Restart only that run's watchdog with `watchdog`.
- **Spend.** Each run's limits are its own; a failed run's spend counts toward its own $10 cap. Do not relaunch a failed run without the owner's approval.

## 15. Final billing verification (Mac, each terminal)

```sh
uv run --locked python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --output "$CTRL/billing.json"
python3 -c 'import json; print(json.load(open("'"$CTRL"'/billing.json"))["totals"])'
```

Compare the provider totals with the report's `cost` for each run. Re-fetch after an hour: Runpod bills every five minutes and records may lag. Confirm each Pod stayed below its $10 cap and note the R2 storage held by `uv run --locked python -m tools.r2_store status`.

## 16. Results

From each archived `summary.json` and the R2 `report.json`: per stage the updates, tokens, learning rate, warmup, ticket and its match, the final checkpoint SHA-256, the validations every 500 global updates and the full validation at each stage end; plus the speed check, cost and timings. The per-update loss curves are in `rehearsal-runs/<run>/<pod id>/<stage>-final/updates.jsonl` in R2. Record the measured numbers in the docs as measured rehearsal results, distinct from the projections in [docs/rehearsal-runs.md](docs/rehearsal-runs.md).
