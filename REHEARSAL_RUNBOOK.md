# Rehearsal runbook: two concurrent A40 pretraining runs (D-35)

Operational sequence for the two owner-approved rehearsal runs of the S model on Runpod A40s: **run A** trains `stage1` (stream `stage1-web`, peak 6e-4) and then `stage2` (`stage2-python` and `stage2-web`, peak 3e-4, fresh optimizer and schedule, weights from stage 1's final checkpoint); **run B** trains one `mixed` stage on all three streams at peak 6e-4. Both start from one shared saved initialization. Each run has its own Pod, controller directory, watchdog, supervisor and known-hosts file; the runs share only the data binding, the initialization and the R2 bucket.

Commands marked **Mac** run on the owner's Mac inside the repository checkout; **Pod** commands run inside a rented Pod over SSH. Open **two terminals**, one per run. Every per-run command below is written once and uses the variables set in step 2 of its own terminal, so the same text runs for A in terminal 1 and for B in terminal 2. Commands marked **once** run in one terminal only.

Nothing in steps 1 to 4 starts a Pod. Step 4 stores the initialization in R2 (a few cents a month). Step 5 creates the Pods and starts billing; do not execute it without the owner's explicit launch authorization for these runs. Runpod offers no provider-enforced Pod lifetime: termination is layered best-effort (in-Pod timer, Mac watchdog, supervisor guard, the worker's own deadlines).

**Approved limits per run (D-35).** Pod wall 60,000 s (16.7 h), raised by the owner on 2026-10-25 so that the $10 cap is the binding limit; approved cost cap $10, which is also the hard stop for runs A and B (owner, 2026-10-25): worker cost stop **$10.00** (estimate $6.2917 x 1.6 = $10.07, capped at $10); Mac watchdog cost stop **$9.99**, one cent inside the worker's; idle 600 s; rising-loss and nonfinite stops as in the pilot. Quoted $0.59/h GPU, $0.60/h with the 40 GB container disk: the 60,000 s wall falls at the $10.00 cap, and the watchdog's $9.99 stop (59,940 s) and the in-Pod timer armed 180 s inside the wall (59,820 s, about $9.97) end a Pod just before it: at most **$10.00 per Pod, $20.00 for both**.

**The two time rules.** Both the readiness receipt (step 8) and the worker at launch (step 9) apply one **launch rule**: the Pod lifetime left, to the first of the wall, the armed in-Pod deadline, the watchdog's wall and either cost stop, must hold the whole planned run at the **reference speed** of 56,479 tok/s, plus every validation at 100,000 tok/s (the interval subsets and both full streams at each quarter and end of every stage), 300 s of startup per stage, a 15 s save every 1,800 s and at each stage end, and the 900 s finalization reserve. That is 38,839 s for run A and 38,035 s for run B, so Pod setup before the worker starts may take up to about 350 minutes (A, 20,981 s) and 363 minutes (B, 21,785 s) of the 59,820 s to the armed deadline. Then, **after 900 s of training**, the worker stops if its post-compile throughput is below 90% of the reference (50,831 tok/s), **or** if its projected finish at the measured rates (remaining updates, validations, recompilation, checkpoint saves and the reserve) passes the armed deadline. The second rule binds even when the speed passes: a run that cannot finish stops early, when it has spent about $0.15 of training.

**Upload window.** While the worker drains its last checkpoints and `report.json` to R2 its GPU is idle, and its heartbeat reads `uploading` for as long as any milestone is pending or in flight. The watchdog policy's `max_transfer_seconds` is **900 s**, counted from the last GPU work (the heartbeat's `last_gpu_activity_unix`: the end of the final save or validation). An `uploading` heartbeat, or the worker's final one, may hold the Pod up to 900 s after the last GPU work, but past the 600 s idle limit only while the Mac has stamped transfer progress within the last 600 s: a new object under the run's R2 prefix, listed with the Mac's own key, or growing bytes of the supervisor's SSH pull. 900 s equals the finalization reserve the run already plans for these uploads, is half the watchdog's 1,800 s ceiling and costs at most $0.15 per Pod. The window is split three ways (`tools/rehearsal_ops.py`): the worker's final drain to R2 ends by **360 s** after the last GPU work (`DRAIN_SECONDS`); then come **120 s** of bounded checks (`TRANSFER_MARGIN_SECONDS`: every R2 call after the drain uses a bounded store and none starts late), so the worker writes its final heartbeat by 480 s; the remaining **420 s**, from 480 s to 900 s, are the Mac's reserve (`PULL_RESERVE_SECONDS`) for the supervisor's SSH pull of any milestone R2 never verified. While that pull's bytes grow, the supervisor's mirror stamps progress every 5 s, which keeps the watchdog from deleting the Pod after 600 s.

**Upload rate.** Only one S label of about 1.65 GB can still be uploading in that 360 s drain, the stage-final milestone: the worker starts no rolling save within 900 s of a stage end, and at the gated rate a rolling label uploads in at most 180 s. With a factor-2 safety margin the Pod's R2 upload rate must be at least `REQUIRED_UPLOAD_BYTES_PER_SECOND` = 1 x 1.65 GB x 2 / 360 s, about **9.17 MB/s** (defined in `tools/rehearsal_ops.py`). The bootstrap measures upload and download throughput with a probe object of at least 128 MiB (step 7), and the readiness receipt's `r2_round_trip_tested` gate refuses a Pod that uploads more slowly (step 8). A milestone also cancels any rolling upload that has not started yet. If R2 does not verify the milestone during the drain, the SSH pull needs about 1.65 GB in about 370 s of the 420 s reserve, about 4.5 MB/s from Pod to Mac (two labels, about 9 MB/s). The pilot's receipt times bound one 1.62 GB pull at 150 s or less, about 10.7 MB/s or faster.

## 1. Prelaunch checks (Mac, free, once)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check && make smoke
uv run --locked python -m tools.rehearsal_run smoke --output runs/rehearsal-smoke
uv run --locked python -m tools.pilot_preflight --output runs/rehearsal-preflight-cpu --device cpu
uv run --locked pytest tests/test_rehearsal_run.py tests/test_rehearsal_integration.py tests/test_rehearsal_runbook.py tests/test_rehearsal_supervisor.py tests/test_rehearsal_ops.py tests/test_rehearsal_bootstrap.py tests/test_rehearsal_readiness.py tests/test_rehearsal_feasibility.py tests/test_rehearsal_watchdog.py tests/test_runpod_watchdog.py tests/test_pod_self_destruct.py -q
```

Expected: `make check` green; the smoke completes offline on CPU; the preflight report shows `"status": "completed"` on `cpu`; every listed test passes. The runbook test parses every tool command in this file against the real parsers, and runs this file's own code against stubs: the step 5 create guard (one Pod per run at a time), create request, Pod ID and directory guard, the helpers file (with a pasted block whose `podssh` line must leave the next line to the shell), the step 5 wait for RUNNING, the step 7 and step 9 launches (each must return at once) and wait loops, the step 6 revision writer, the step 7 leak check, the step 13 DELETE gate and the step 15 billing check. It also checks that no block holds a `#` comment and that every block parses in an interactive zsh. The integration test builds a receipt with the receipt tool and has the worker accept it, and feeds the smoke's heartbeats, reports and in-memory bucket to the supervisor.

## 2. Credentials and per-run environment (Mac, free, each terminal)

`.env` is git-ignored. It holds `RUNPOD_API_KEY`, `R2_ENDPOINT`, `R2_BUCKET`, the Mac's own R2 key `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` (stages data, uploads the initialization, verifies results) and the Pods' separate key `R2_POD_ACCESS_KEY_ID`/`R2_POD_SECRET_ACCESS_KEY`. The Pod key must be its own Cloudflare API token with Object Read & Write on this one bucket only; it reaches the Pods only over SSH stdin (step 6), never in a create body, and it is revoked after both runs (step 13).

The blocks hold no `#` comments; every explanation sits in the text around them. An interactive zsh on macOS has `interactivecomments` off, so there a `#` starts an argument, not a comment: a trailing note becomes extra arguments, and an apostrophe in one opens a quote that swallows the rest of the block. With bracketed paste zsh parses a whole pasted block before it runs any of it, so a `setopt` line inside a block would come too late. The setting matters only for lines of your own: before you paste one that carries a comment, run `setopt interactivecomments` in that terminal (oh-my-zsh already sets it; bash and scripts need nothing).

No secret value ever appears on a process command line, where another local account could read it with `ps`. The Runpod key reaches `curl` on its standard input (`--config -`, written by the shell's builtin `printf`), and the leak check in step 7 feeds the values to `grep` on its standard input. The Python tools read keys from the environment.

Set `RUN=A` in terminal 1 and `RUN=B` in terminal 2:

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi
done
gh auth status
caffeinate -dimsu -w $$ &
export RUN=A
export RUN_LOWER=$(printf %s "$RUN" | tr AB ab)
export CTRL_ROOT=runs/rehearsal-controller
export KNOWN_HOSTS=~/.ssh/rehearsal_${RUN_LOWER}_known_hosts
export SPEC=configs/rehearsal-a40-s-$RUN_LOWER.json
export SHARED=runs/rehearsal-shared && mkdir -p "$SHARED"
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts
export BUILD_DIR=runs/rehearsal-s-20261025
export BINDING=runs/rehearsal-binding/binding.json
export BINDING_SHA256=<reviewed binding sha256>
printf '%s  %s\n' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -
```

The loop prints each variable's name and whether it is set, never a value. `gh auth status` checks the GitHub login the supervisor uses to archive reports and summaries in the private repository; `PRIVATE_REPO` must be private, and the supervisor verifies that. Each Pod gets its own controller directory `$CTRL_ROOT/<run>/<pod id>` (step 5). `KNOWN_HOSTS` is this run's own known-hosts file, written again for each new Pod in step 5; its path must hold no whitespace, because ssh splits the option on spaces. `BUILD_DIR` is the admitted S build and `BINDING` the reviewed binding, which is never rebuilt here.

`BINDING_SHA256` is the `sha256` that `tools.rehearsal_binding build` printed when the binding was written, as recorded in the owner's review of that binding. Paste it in place of the placeholder; never derive it from the file, or every later `--expected-sha256` check would only compare the file with itself. The last line should print `runs/rehearsal-binding/binding.json: OK`. If `shasum` prints anything but `OK`, the file is not the reviewed binding: stop and ask the owner.

`caffeinate` keeps the Mac awake while this terminal lives. A sleeping Mac stops the mirror: when it wakes, the watchdog would read a stale heartbeat and delete the Pod. Keep both terminals open on AC power for the whole run. The working directory must not sit under a symlinked path (macOS `/tmp` is one).

## 3. Final quote check for both Pods (Mac, free, once)

Open the Runpod console: Secure Cloud, NVIDIA A40, on-demand, 1 GPU, container disk 40 GB, no volume. Check that **two** such Pods are available and record the hourly price. Then project both runs at that price:

```sh
uv run --locked python -m tools.pilot_feasibility rehearsal --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --quoted-gpu-hourly 0.59 --output "$SHARED/feasibility.json"
```

Replace `0.59` with the console price. If it exceeds $0.59, stop and ask the owner: D-35 was approved at that quote. The report projects, per run, the time at the reference speed and at the 90% floor with its own assumptions (Pod setup included), the cost against the $10.00 stop, the container-disk fit and the R2 storage price. It is a projection only: the gate that binds is the launch rule above, which the receipt tool and the worker compute from the specification and the binding. Confirm the account balance covers both runs to their cost stops (at least $21) and that no network volume exists.

## 4. Data, binding and the shared initialization (Mac, once; R2 storage only)

Re-stage the binding: every pinned file is re-hashed locally and skipped in R2 when already stored with the same SHA-256, the build's `report.json` must list exactly the binding's shards, and the binding goes last:

```sh
uv run --locked python -m tools.rehearsal_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --build "$BUILD_DIR" --receipt "$SHARED/stage-receipt.json"
```

Write the shared initialization (seed 0) once, pin its hash and size, and upload it under a key named by its hash:

```sh
uv run --locked python -m tools.rehearsal_run write-initialization --specification configs/rehearsal-a40-s-a.json --specification configs/rehearsal-a40-s-b.json --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --output "$SHARED/init/rehearsal-init.pt"
export INIT_SHA256=$(shasum -a 256 "$SHARED/init/rehearsal-init.pt" | cut -d' ' -f1)
export INIT_SIZE=$(wc -c < "$SHARED/init/rehearsal-init.pt" | tr -d ' ')
export INIT_KEY=rehearsal-init/$INIT_SHA256/rehearsal-init.pt
uv run --locked python -m tools.r2_store upload --root "$SHARED/init" --prefix "rehearsal-init/$INIT_SHA256/" --receipt "$SHARED/init-upload.json" "$SHARED/init/rehearsal-init.pt"
printf 'export INIT_SHA256=%s INIT_SIZE=%s INIT_KEY=%s\n' "$INIT_SHA256" "$INIT_SIZE" "$INIT_KEY" > "$SHARED/init.env"
```

`write-initialization` takes both specifications and refuses unless they name runs A and B with one model and seed. The SHA-256 it prints must equal `INIT_SHA256`. In the other terminal, `source "$SHARED/init.env"`. Both runs start from these exact bytes; each Pod fetches them pinned to this hash and size.

## 5. Pod creation, identity and the watchdog (Mac, BILLABLE, each terminal)

Only after explicit launch authorization. Capture the billing origin before the create request so every deadline is conservative. The body is `configs/runpod-pod-a40-s-rehearsal.json` (A40, Secure Cloud, template `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, 40 GB container disk, TCP 22, no volume, no environment variables) with the run's letter appended to the name. The create request runs from a fresh launch directory; once the Pod ID is known, that directory becomes this Pod's own controller directory `$CTRL`.

The block creates at most one Pod per run at a time. It refuses to send the create request, and says why, when this terminal already names a Pod (`POD_ID` is set), or when any directory under `$CTRL_ROOT/$RUN_LOWER` has no `TEARDOWN` line in its `run.env`: a Pod directory is named by its Pod ID, and a `launch-<time>` directory is a create whose Pod was never confirmed. Without that check, pasting the block twice would create a second Pod, and the first would bill with no watchdog. `SELF_DESTRUCT_DEADLINE` is 3 minutes inside the 60,000-second wall:

```sh
OPEN_PODS=$(find "$CTRL_ROOT/$RUN_LOWER" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r DIR; do grep -qs '^export TEARDOWN=' "$DIR/run.env" || printf '%s ' "$DIR"; done)
if [ -n "${POD_ID:-}" ]; then
  echo "run $RUN: NOT creating a Pod: this terminal already names Pod $POD_ID (${CTRL:-no controller directory}), which may still bill, and a second Pod would leave one of them without a watchdog. Confirm Pod $POD_ID gone with step 13 in this terminal. Only after step 13 has recorded its TEARDOWN and the owner has approved a new Pod, run unset POD_ID and paste this block again."
elif [ -n "$OPEN_PODS" ]; then
  echo "run $RUN: NOT creating a Pod: no TEARDOWN is recorded in ${OPEN_PODS% }. Each names a Pod of run $RUN that may still bill (the directory name is its Pod ID), or a launch-<time> create whose Pod was never confirmed. End each Pod with step 13 in its own terminal, settle each launch directory as step 5 says, and ask the owner before creating another."
else
  export ORIGIN=$(date +%s)
  export LAUNCH=$CTRL_ROOT/$RUN_LOWER/launch-$ORIGIN && mkdir -p "$LAUNCH"
  python3 -c 'import json,sys; b=json.load(open("configs/runpod-pod-a40-s-rehearsal.json")); b["name"]+="-"+sys.argv[1]; print(json.dumps(b))' "$RUN_LOWER" > "$LAUNCH/create-body.json"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS --fail-with-body -X POST https://api.runpod.io/v2/pods \
    -H "Content-Type: application/json" --data @"$LAUNCH/create-body.json" -o "$LAUNCH/create-response.json" -w '%{http_code}\n' > "$LAUNCH/create-http-code"
  export POD_ID=$(python3 -c 'import json,re,sys; v=json.load(open(sys.argv[1])); i=v.get("id") if isinstance(v, dict) else None; ok=isinstance(i, str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,127}", i); print(i) if ok else sys.exit("no valid Pod id in " + sys.argv[1])' "$LAUNCH/create-response.json")
  export SELF_DESTRUCT_DEADLINE=$((ORIGIN + 60000 - 180))
  export CTRL=$CTRL_ROOT/$RUN_LOWER/$POD_ID
  if [ -z "$POD_ID" ]; then
    POD_NAME=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$LAUNCH/create-body.json")
    SINCE=$(python3 -c 'import sys,time; print(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(int(sys.argv[1]))))' "$ORIGIN")
    echo "run $RUN: the create request ended with HTTP $(cat "$LAUNCH/create-http-code") and $LAUNCH/create-response.json names no valid Pod id. A Pod MAY STILL EXIST: a gateway error or a dropped connection can follow a create the server carried out. Stop here and ask the owner to look in the Runpod console for a Pod named $POD_NAME created at or after $SINCE, to terminate any such Pod there, and to decide before any retry. Never delete a Pod from this terminal by name or by guess."
    export CTRL=$LAUNCH
  elif mkdir "$CTRL"; then
    mv "$LAUNCH/create-body.json" "$LAUNCH/create-response.json" "$LAUNCH/create-http-code" "$CTRL/" && rmdir "$LAUNCH"
    printf 'export ORIGIN=%s POD_ID=%s SELF_DESTRUCT_DEADLINE=%s\n' "$ORIGIN" "$POD_ID" "$SELF_DESTRUCT_DEADLINE" > "$CTRL/run.env"
    echo "run $RUN: Pod $POD_ID, controller directory $CTRL"
  else
    echo "run $RUN: $CTRL already exists, so it is NOT this Pod's: this launch did not create it, and a controller directory is never reused. Leave it untouched. Stop here and terminate Pod $POD_ID, the one $LAUNCH/create-response.json names, with step 13 in this terminal (TERMINATE=yes); never delete any other Pod."
    export CTRL=$LAUNCH
  fi
fi
```

If it printed `NOT creating a Pod`, nothing was sent and nothing changed: do not continue with this step.

`POD_ID` comes from the create response itself, never from a hand copy: the parser prints the response's `id` only when it has the shape the watchdog accepts, and otherwise prints nothing and refuses. The Runpod key still reaches `curl` on its standard input only. `--fail-with-body` makes `curl` exit nonzero on an HTTP error and still keep the reply in `create-response.json`, and `-w` records the HTTP code in `create-http-code` (`000` when no reply arrived at all). `mkdir` without `-p` refuses an existing directory, so nothing below ever starts in a controller directory that another Pod, or an earlier run of this step, left behind: no stale `run.env`, policy or mirrored heartbeat can reach this Pod's watchdog. If it printed `names no valid Pod id` or `already exists`, do not continue. Both refusals point `CTRL` at the launch directory, which holds only this request's body, response and HTTP code, so nothing later in this terminal touches the existing directory or another Pod's watchdog.

After `names no valid Pod id`, no Pod ID is known, but the request may still have created a Pod that bills with no timer or watchdog: a gateway 5xx, a timeout or a dropped connection can come after the server acted. Read `create-response.json` and `create-http-code`, and ask the owner to check the Runpod console for a Pod with the printed name created at or after the printed time (`ORIGIN`), and to terminate any such Pod there. Retry only when the owner says so: record the settled launch first, `printf 'export TEARDOWN=%s\n' "$(date +%s)" >> "$LAUNCH/run.env"`, which lets this block create again; the launch directory stays as the record. (In another terminal `LAUNCH` is unset: use the `launch-<time>` directory the refusal names.) After `already exists`, terminate the Pod the response names at once with step 13 (`TERMINATE=yes`): it bills, and no timer or watchdog guards it. Step 13 records its `TEARDOWN` in the launch directory. A new Pod for the same run (step 14) always gets a new directory, and this terminal creates it only after step 13 has recorded the old Pod's `TEARDOWN` and `unset POD_ID` has cleared its ID.

Capture the identity the guards enforce, write the policy, write this run's helpers once into `$CTRL/helpers.sh`, and start the watchdog at once. The helpers read `CTRL`, `RUN`, `ORIGIN`, `POD_ID`, `POD_HOST`, `POD_PORT`, `POD_SSH_KEY` and `KNOWN_HOSTS` from the shell at the moment of each call, so source the file after step 2 and `run.env`. `watchdog` (re)starts this run's watchdog with a fresh idle baseline and leaves the other run's untouched. `keepalive` is what every wait loop below calls on each pass (see after the block). `podssh` and `podscp` reach this Pod only, pinned to this run's known-hosts file. `podssh` never reads this terminal: its `ssh` takes its standard input from `/dev/null`, so in a shell without bracketed paste (macOS `/bin/bash` 3.2) it cannot swallow the rest of a pasted block. `podssh_stdin` is the same command with its standard input kept; only step 6 uses it, once, to pipe the Pod's R2 key:

```sh
uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" \
  --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json"
cat > "$CTRL/policy.json" <<EOF
{"mode": "rehearsal", "max_wall_seconds": 60000, "max_cost_usd": 9.99, "max_idle_seconds": 600,
 "quoted_total_hourly": 0.60, "max_transfer_seconds": 900, "billing_started_at_unix": $ORIGIN,
 "idle_baseline_unix": $(date +%s)}
EOF
[ -e "$CTRL/helpers.sh" ] || cat > "$CTRL/helpers.sh" <<'EOF'
watchdog() {
  OLD=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
  BASE=$(date +%s)
  if [ -e "$CTRL/watchdog-receipt.json" ]; then cp "$CTRL/watchdog-receipt.json" "$CTRL/watchdog-receipt.json.prev-$BASE"; fi
  nohup uv run --locked python -m tools.runpod_watchdog --mode rehearsal --identity "$CTRL/pod-identity.json" \
    --heartbeat "$CTRL/supervisor/heartbeat.json" --receipt "$CTRL/watchdog-receipt.json" \
    --pidfile "$CTRL/watchdog.pid" --max-wall-seconds 60000 --max-idle-seconds 600 \
    --max-transfer-seconds 900 \
    --max-cost-usd 9.99 --quoted-total-hourly 0.60 --billing-started-at-unix "$ORIGIN" \
    --idle-baseline-unix "$BASE" --poll-seconds 20 \
    >> "$CTRL/watchdog.log" 2>&1 < /dev/null &
  JOB=$!
  NEW=""
  for attempt in $(seq 1 60); do
    NEW=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
    if [ -n "$NEW" ] && [ "$NEW" != "$OLD" ]; then break; fi
    sleep 1
  done
  if [ -z "$NEW" ] || [ "$NEW" = "$OLD" ]; then
    if ps -p "$JOB" -o command= 2>/dev/null | grep -qF -- "--identity $CTRL/pod-identity.json"; then kill "$JOB"; fi
    echo "run $RUN: the new watchdog did not start within 60 s and was stopped; see $CTRL/watchdog.log"; return 1
  fi
  if [ -n "$OLD" ] && ps -p "$OLD" -o command= | grep -qF -- "--identity $CTRL/pod-identity.json"; then kill "$OLD"; fi
  echo "run $RUN watchdog pid $NEW, idle baseline $BASE"
}
keepalive() {
  LIVE=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
  PROCS=$(ps -A -o pid= -o command= 2>/dev/null)
  PARENT=$(ps -p "$LIVE" -o ppid= 2>/dev/null | tr -d ' ')
  STRAY=$(printf '%s\n' "$PROCS" | awk -v live="$LIVE" -v parent="$PARENT" -v id="--identity $CTRL/pod-identity.json" 'index($0, "tools.runpod_watchdog") && index($0, id) && $1 != live && $1 != parent { printf "%s ", $1 }')
  if [ -n "$STRAY" ]; then
    echo "WARNING run $RUN: a watchdog of Pod $POD_ID that $CTRL/watchdog.pid does not name still runs, with an old idle baseline: PID ${STRAY% }. Check each with ps -p <pid> -o command= and stop it with kill <pid>."
  fi
  if [ -z "$LIVE" ] || ! ps -p "$LIVE" -o command= 2>/dev/null | grep -qF -- "--identity $CTRL/pod-identity.json"; then
    GONE=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); gone=isinstance(r, dict) and r.get("pod_id") == sys.argv[2] and r.get("state") in ("terminated", "absent", "identity_mismatch"); print("state %s, reason %s, detail %s" % (r.get("state"), r.get("reason"), r.get("detail"))) if gone else None' "$CTRL/watchdog-receipt.json" "$POD_ID" 2>/dev/null)
    if [ -n "$GONE" ]; then
      echo "STOP run $RUN: POD GONE: the watchdog of Pod $POD_ID ended for good with $GONE ($CTRL/watchdog-receipt.json). It is NOT restarted. Follow step 14."
      return 3
    fi
    echo "WARNING run $RUN: NO LIVE WATCHDOG guards Pod $POD_ID: $CTRL/watchdog.pid is missing or names no live watchdog of this Pod. Read $CTRL/watchdog.log and $CTRL/watchdog-receipt.json. Restarting it now."
    BASE=0
  fi
  if [ $(( $(date +%s) - BASE )) -ge 300 ]; then
    watchdog || { echo "WARNING run $RUN: the watchdog restart FAILED; see $CTRL/watchdog.log. Trying again on the next pass."; BASE=0; }
  fi
  return 0
}
podssh() {
  podssh_stdin "$@" < /dev/null
}
podssh_stdin() {
  ssh -F /dev/null -p "$POD_PORT" -i "$POD_SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" -o BatchMode=yes "root@$POD_HOST" "$@"
}
podscp() {
  scp -F /dev/null -P "$POD_PORT" -i "$POD_SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
    -o "UserKnownHostsFile=$KNOWN_HOSTS" -o BatchMode=yes "$@"
}
EOF
source "$CTRL/helpers.sh"
watchdog
```

**A reopened terminal** (or the second window of step 10) repeats step 2 for its run, then sets this Pod's directory and sources its files, and never re-runs a block that creates a Pod or writes `run.env`, the policy or the known-hosts file: `export CTRL=$CTRL_ROOT/$RUN_LOWER/<pod id>; source "$SHARED/init.env"; source "$CTRL/run.env"; source "$CTRL/helpers.sh"`. `ls "$CTRL_ROOT/$RUN_LOWER"` lists the run's Pod directories.

Each watchdog writes its own PID to `$CTRL/watchdog.pid` and removes it when it exits. The helper starts the replacement first, then stops only the process whose PID the file held and whose command line names this Pod's identity file; there is no global `pkill`, so restarting run A's watchdog never touches run B's. Before it starts one, it copies the current receipt aside as `watchdog-receipt.json.prev-<time>`, so a restart never overwrites what the last watchdog recorded. If the new watchdog has not written its PID within 60 s, the helper stops the process it just started (after the same command-line check) and returns 1, so a failed restart never leaves a stray watchdog with an old idle baseline. The watchdog reads the heartbeat the supervisor mirrors to `$CTRL/supervisor/heartbeat.json`, which exists only once the supervisor runs (step 10); a mirrored heartbeat whose `pod_id` is not this Pod's, or whose update predates the Pod's creation, counts as absent, never as fresh. Until then the idle clock runs from the last baseline: every wait loop below calls `keepalive` on each pass, which runs `watchdog` again once 5 minutes have passed since the last baseline, and the steps call `watchdog` at each setup milestone. Wall and cost limits never pause, and 600 seconds without a restart still deletes the Pod.

`keepalive` never fails silently. When `watchdog.pid` is missing, or names a process that is not this Pod's watchdog (the same command-line check `watchdog` makes before it stops one), it first reads `watchdog-receipt.json`. A watchdog that deleted the Pod, or found it gone or replaced, writes a final receipt (state `terminated`, `absent` or `identity_mismatch`), removes its PID file and exits. For such a receipt of this Pod, `keepalive` prints a `STOP ... POD GONE` line with the receipt's state, reason (which names the trigger) and detail, never restarts the watchdog (a new one would overwrite that receipt with `absent`), and returns 3; every wait loop below stops on 3 with `stop waiting: the Pod is gone` (step 14). Otherwise it prints a `WARNING ... NO LIVE WATCHDOG` line and restarts the watchdog on that pass. When a restart fails, it prints a `WARNING ... restart FAILED` line and tries again on the next pass, instead of waiting another 5 minutes. On every pass it also warns about any watchdog of this Pod that `watchdog.pid` does not name, apart from the `uv` process that runs the named one: such a process keeps an old idle baseline and could delete the Pod early, so stop it by hand.

Then wait, for at most 20 minutes, until `pod-connection.json` shows `"status": "RUNNING"` with a direct SSH host and port. The loop captures the Pod again every 20 s and calls `keepalive`: before the first heartbeat the idle clock runs from the last baseline, so without the re-baseline a start that takes more than 600 s would have the watchdog delete the Pod. Twenty minutes is a fifth of run A's 100-minute setup allowance (**The two time rules**), which still leaves the bootstrap its 45 minutes, and costs at most $0.20 at $0.60/h. Runpod reports `RUNNING` from the create reply on, before the container is up: that reply has no runtime and no direct SSH, so its capture shows `runtime_ports` null and `ssh_proxy_only` true, and a Pod still pulling its image looks the same. The status alone therefore proves nothing, and a capture counts as proxy-only only once the container is up: `runtime_ports` is a non-empty list, none of its entries maps private port 22 to a public TCP port, and there is no direct SSH host or port. A Pod seen so on three captures in a row (40 s) has no public IP, and no file can move to or from it, so the wait stops early for it; one or two such captures are allowed, in case the port 22 mapping and the direct address appear a moment after the container. A capture whose container is not up yet, or whose port 22 is mapped before the direct address appears, keeps the wait going within its 20 minutes:

```sh
GIVE_UP=$(( $(date +%s) + 1200 )); PROXY_ONLY=0
while :; do
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); d=c.get("ssh_direct") or {}; p=c.get("runtime_ports"); up=isinstance(p, list) and len(p) > 0; ssh22=up and any(isinstance(e, dict) and e.get("type") == "tcp" and e.get("private") == 22 and bool(e.get("public")) for e in p); running=c.get("status") == "RUNNING"; ok=running and bool(d.get("host")) and bool(d.get("port")); proxy=running and up and not ssh22 and not d.get("host") and not d.get("port") and c.get("ssh_proxy_only") is True; print(c.get("status"), "with direct SSH" if ok else "with proxy SSH only" if proxy else "without direct SSH" if up else "without direct SSH: container not up yet"); sys.exit(0 if ok else 2 if proxy else 1)' "$CTRL/pod-connection.json"; STATE=$?
  if [ "$STATE" -eq 0 ]; then break; fi
  if [ "$STATE" -eq 2 ]; then PROXY_ONLY=$((PROXY_ONLY + 1)); else PROXY_ONLY=0; fi
  if [ "$PROXY_ONLY" -ge 3 ]; then echo "run $RUN: Pod $POD_ID is RUNNING with proxy SSH only: it has no public IP, so no file can reach it; terminate it (step 13, TERMINATE=yes)"; break; fi
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: Pod $POD_ID is not RUNNING with direct SSH after 20 minutes; terminate it (step 13, TERMINATE=yes)"; break; fi
  sleep 20
  uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json" > /dev/null
  keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone (watchdog receipt above); follow step 14"; break; fi
done
```

If it printed `proxy SSH only` or `not RUNNING`, do not continue: terminate the Pod (step 13, `TERMINATE=yes`) and report. If it printed `the Pod is gone`, do not continue either: follow step 14. The watchdog is no longer re-baselined, so its idle rule also deletes the Pod within 600 s. A Pod without a public IP may be created again with the owner's approval: step 5 again, after step 13 has recorded its `TEARDOWN` and `unset POD_ID` has cleared its ID. Otherwise pin this Pod's host key in this run's own known-hosts file, and compare the fingerprint `ssh-keygen` prints with the one shown in this Pod's web terminal:

```sh
export POD_HOST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["host"])' "$CTRL/pod-connection.json")
export POD_PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["port"])' "$CTRL/pod-connection.json")
printf 'export POD_HOST=%s POD_PORT=%s\n' "$POD_HOST" "$POD_PORT" >> "$CTRL/run.env"
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS" 2>/dev/null
ssh-keygen -lf "$KNOWN_HOSTS"
watchdog
```

A host key that changes later means another machine answers at that address: stop and terminate the Pod rather than accepting it.

## 6. Code, binding and the Pod's R2 key (Mac to Pod, billable, each terminal)

Record exactly which code the Pod will run, then stage it with the binding. The writer below refuses a dirty working tree and writes nothing then; from a clean tree it archives `HEAD` to `$CTRL/repo.tar.gz` and writes `$CTRL/revision.json` with the 40-hex commit, `"clean": true` and the archive's SHA-256. The readiness receipt records that object as `revision`, and the worker refuses to start unless the Pod's copy, `/workspace/rehearsal/revision.json`, equals it (it then records it in `report.json` as `code_revision`). Then deliver the Pod's R2 key over SSH stdin into a 0600 file, with `podssh_stdin`: this is the one call that keeps its standard input, since the key travels on it. `pod-env` refuses to write to a terminal and refuses the Mac's own key, so the values never appear on screen, in a process list or in a create body:

```sh
python3 - "$CTRL" <<'PY'
import hashlib, json, re, subprocess, sys
from pathlib import Path

control = Path(sys.argv[1])


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


if git("status", "--porcelain"):
    sys.exit("the working tree is dirty: commit or stash first; no code was archived")
commit = git("rev-parse", "HEAD").strip()
if not re.fullmatch(r"[0-9a-f]{40}", commit):
    sys.exit(f"HEAD is not a 40-hex commit: {commit!r}")
archive = control / "repo.tar.gz"
subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(archive), commit], check=True)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
record = {"git_commit": commit, "clean": True, "archive_sha256": digest}
(control / "revision.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(f"revision {commit}, archive sha256 {digest}")
PY
podssh 'mkdir -p /workspace/repo /workspace/rehearsal /workspace/rehearsal-data'
podscp "$CTRL/repo.tar.gz" root@$POD_HOST:/workspace/repo.tar.gz
podscp "$CTRL/revision.json" root@$POD_HOST:/workspace/revision.json
podscp "$BINDING" root@$POD_HOST:/workspace/rehearsal/binding.json
uv run --locked python -m tools.rehearsal_ops pod-env | podssh_stdin 'umask 077 && cat > /workspace/rehearsal/r2-pod.env && chmod 600 /workspace/rehearsal/r2-pod.env'
podssh 'stat -c "%a %U %s" /workspace/rehearsal/r2-pod.env'
watchdog
```

The `stat` line should print `600 root` and a nonzero size. If the writer says the tree is dirty, nothing was archived and both `podscp` lines of the code fail: commit or stash, then run this block again.

## 7. Bootstrap (Pod, billable, each terminal)

The bootstrap arms the in-Pod self-destruct timer first, before anything else can fail. In rehearsal mode it runs exactly this command, detached:

```sh
python3 tools/pod_self_destruct.py --pod-id "$RUNPOD_POD_ID" --deadline-unix "$SELF_DESTRUCT_DEADLINE" \
  --receipt /workspace/rehearsal/self-destruct.json --stop-file /workspace/rehearsal/TERMINATE_NOW --max-lifetime-seconds 60000
```

An existing `self-destruct.json` counts as armed only while its timer lives: its `pid` answers `kill -0` and its `checked_at_unix` (rewritten every 5 s) is at most 30 s old. Any other receipt is set aside as `self-destruct.json.stale-<time>` and the timer is armed again, so a re-run bootstrap never trusts a dead timer. The bootstrap then checks the shipped archive against `/workspace/revision.json` (the SHA-256 must match, the commit be 40 hex digits and the tree clean) and copies that record byte for byte to `/workspace/rehearsal/revision.json`, loads the R2 key file (a regular 0600 file, names printed, values never), refuses to continue if the Mac's key variables are set on the Pod, records the environment, installs the project against the image's CUDA torch, measures an R2 round trip under `rehearsal-runs/<run>/<pod id>/` (upload and download throughput with a probe of at least 128 MiB), fetches this run's pinned shards, manifests and tokenizer with `tools.rehearsal_binding fetch --key-prefix R2_POD_` and re-verifies them, fetches the initialization pinned to `INIT_SHA256` and `INIT_SIZE`, runs the CPU tests (the pilot's list plus the one-pass loader, validation, R2, the worker's own `tests/test_rehearsal_run.py`, the integration test and the Mac-side rehearsal tools' tests; none needs Docker or the network, and the data build's tests stay on the Mac), and runs the A40 precision, fused-attention, compile-equivalence (`tools.pilot_preflight --compile-check`) and exact-resume preflight. Launch it detached so an SSH drop cannot kill it, and keep re-baselining this run's watchdog while it runs, for at most 45 minutes. The launch refuses to run when `/workspace/rehearsal/bootstrap.log` already exists (checked from the Mac, and again on the Pod in the launch command itself), so pasting this block again never starts a second bootstrap: the wait follows the first one instead.

The launch returns at once. On the Pod each check is its own statement ending in `|| exit 1;`, so the command sent to the background is the bootstrap alone, with its output going to the log. Joining the checks to it with `&&` would send the whole list to the background as one subshell, which keeps the SSH session's output open: `podssh` would not return until the bootstrap ended, 15 to 45 minutes later, the wait below would never re-baseline the watchdog, and the watchdog's idle rule would delete the Pod after 600 s.

The wait stops early when the Pod answers that no bootstrap process runs and the log has no `BOOTSTRAP_EXIT` line: the launch itself failed, and no timer may be armed. Only `pgrep`'s own "none found" (exit 1) counts; an SSH failure (255) or a Pod without `pgrep` never stops the wait early:

```sh
if podssh 'test -e /workspace/rehearsal/bootstrap.log'; then
  echo "run $RUN: /workspace/rehearsal/bootstrap.log already exists, so a bootstrap was launched on this Pod before; not launching a second one. The wait below follows that one."
else
  podssh "test ! -e /workspace/rehearsal/bootstrap.log || exit 1; cd /workspace/repo && tar xzf /workspace/repo.tar.gz || exit 1; setsid nohup bash -c 'bash tools/pilot_pod_bootstrap.sh --mode rehearsal --run $RUN --deadline-unix $SELF_DESTRUCT_DEADLINE --workspace /workspace/rehearsal --repo /workspace/repo --revision /workspace/revision.json --archive /workspace/repo.tar.gz --r2-env /workspace/rehearsal/r2-pod.env --binding /workspace/rehearsal/binding.json --binding-sha256 $BINDING_SHA256 --data-root /workspace/rehearsal-data --initialization-key $INIT_KEY --initialization-sha256 $INIT_SHA256 --initialization-size $INIT_SIZE; echo BOOTSTRAP_EXIT=\$?' > /workspace/rehearsal/bootstrap.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 2700 ))
until podssh 'grep -q ^BOOTSTRAP_EXIT= /workspace/rehearsal/bootstrap.log'; do
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: bootstrap exceeded 45 minutes; terminate the Pod (step 13, TERMINATE=yes)"; break; fi
  podssh 'pgrep -f pilot_pod_bootstrap.sh' > /dev/null; ALIVE=$?
  if [ "$ALIVE" -eq 1 ] && ! podssh 'grep -q ^BOOTSTRAP_EXIT= /workspace/rehearsal/bootstrap.log'; then echo "run $RUN: no bootstrap runs on the Pod and bootstrap.log has no BOOTSTRAP_EXIT line: the launch failed. Read the lines below, then terminate the Pod (step 13, TERMINATE=yes)"; break; fi
  sleep 30; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone (watchdog receipt above); follow step 14"; break; fi
done
for name in bootstrap.log self-destruct.json environment.json r2-round-trip.json data-fetch.json data-verification.json initialization-fetch.json; do
  podscp root@$POD_HOST:/workspace/rehearsal/$name "$CTRL/$name"
done
podscp root@$POD_HOST:/workspace/rehearsal/preflight-a40/report.json "$CTRL/preflight-a40-report.json"
podscp root@$POD_HOST:/workspace/rehearsal/revision.json "$CTRL/pod-revision.json"
cmp "$CTRL/revision.json" "$CTRL/pod-revision.json" && echo "the Pod holds this revision"
for name in R2_POD_SECRET_ACCESS_KEY R2_SECRET_ACCESS_KEY RUNPOD_API_KEY; do printenv "$name"; done | grep -cFf - "$CTRL/bootstrap.log"
grep BOOTSTRAP_EXIT "$CTRL/bootstrap.log"; watchdog
```

Expected: `BOOTSTRAP_EXIT=0`; `self-destruct.json` shows `"state": "armed"` and `deadline_unix` equal to `SELF_DESTRUCT_DEADLINE`; `the Pod holds this revision`; `r2-round-trip.json` shows `"status": "passed"` and an `upload_bytes_per_second` of at least about 9.17 million (the receipt refuses less; see **Upload rate** above); `preflight-a40-report.json` shows `"status": "completed"`, `"hardware_precision_and_resume_checked": true`, a passed `fused_attention` and a passed `compile_equivalence` (the receipt tool refuses a compiling specification without it). The leak check prints `0`: `bootstrap.log` holds none of the three secret values. `printenv` takes only the variables' names on its command line, and `grep` reads the values as patterns from its standard input (`-f -`), so no value reaches a process list. If the wait printed `the Pod is gone`, the watchdog has ended this Pod: follow step 14. Any other exit code: read `bootstrap.log`, then terminate the Pod (step 13) unless the fix is obvious and cheap. Exit 6 means the shipped archive does not match `/workspace/revision.json`, or the workspace already holds another revision; the message says which. To run the bootstrap again after such a fix, first check that `podssh 'pgrep -f pilot_pod_bootstrap.sh'` prints nothing, then move the old log aside with `podssh 'mv /workspace/rehearsal/bootstrap.log /workspace/rehearsal/bootstrap.log.failed-$(date +%s)'` and paste this block again. If the fix is a new commit (step 6 run again, which ships a new archive and `revision.json`), also move the workspace's copy of the old revision aside before pasting, with `podssh 'mv /workspace/rehearsal/revision.json /workspace/rehearsal/revision.json.old-$(date +%s)'`: the bootstrap never replaces a `revision.json` it finds there, so without this it exits 6 every time.

## 8. Readiness receipt (Mac, billable, each terminal)

The confirmations file is the owner's explicit statement. `termination_guarantee` must be `redundant_best_effort`. `r2_token_bucket_scoped` states that the Pod key is a separate token limited to this bucket. The round-trip and data gates are not typed by hand: the receipt tool derives them from the Pod's own receipts, and `r2_round_trip_tested` also requires the measured `upload_bytes_per_second` to reach `REQUIRED_UPLOAD_BYTES_PER_SECOND`. It reads `$SPEC` exactly as the worker does, and refuses to write if any gate is false, the specification differs from D-35 or from the binding's stages, the preflight lacks a passed compile check, the policy's billing origin or 900 s transfer window is not the approved one, the timer is not armed, a limit is already at risk, or the lifetime left fails the launch rule at the reference speed. The receipt it writes is exactly what the worker accepts in step 9 (kind, run, gates, the specification, binding, preflight and initialization SHA-256s, the code revision, the Pod identity, the policy with the billing origin, and the armed deadline). Write the confirmations first:

```sh
cat > "$CTRL/confirmations.json" <<'EOF'
{"owner_confirms_four_runpod_steps": true, "cloud_data_pre_staged": true,
 "external_watchdog_armed": true, "hardware_preflight_passed": true, "full_console_quote_confirmed": true,
 "staging_storage_cost_bound": true, "r2_token_bucket_scoped": true,
 "termination_guarantee": "redundant_best_effort", "owner_accepts_best_effort_termination": true}
EOF
```

Then build the receipt. Its first line copies the timer's receipt from the Pod again, right before the receipt tool reads it, because the tool refuses a `self-destruct.json` the live timer last rewrote more than 60 s earlier, and the copy from step 7 is older than that. If the tool still refuses it as stale, paste this block again:

```sh
podscp root@$POD_HOST:/workspace/rehearsal/self-destruct.json "$CTRL/self-destruct.json"
uv run --locked python -m tools.pilot_readiness_receipt rehearsal --run "$RUN" --specification "$SPEC" \
  --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --preflight "$CTRL/preflight-a40-report.json" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --self-destruct-receipt "$CTRL/self-destruct.json" --confirmations "$CTRL/confirmations.json" \
  --data-fetch "$CTRL/data-fetch.json" --data-verification "$CTRL/data-verification.json" \
  --round-trip "$CTRL/r2-round-trip.json" --initialization-fetch "$CTRL/initialization-fetch.json" \
  --initialization-sha256 "$INIT_SHA256" --revision "$CTRL/revision.json" --billing-origin-unix "$ORIGIN" \
  --output "$CTRL/readiness-receipt.json"
podscp "$CTRL/readiness-receipt.json" root@$POD_HOST:/workspace/rehearsal/readiness-receipt.json
watchdog
```

The four Runpod steps are: funded balance verified for both runs, SSH public key registered, API key with Pod permissions created, console quote recorded. The printed line reports `fits_at_reference` (always true for a written receipt) and `fits_at_floor` for the time actually left; `fits_at_floor: false` means the 900-second projected-finish rule will stop the run if it trains near the floor. Launch the worker promptly: it applies the same launch rule again at its own start.

## 9. Worker launch (Pod, billable, each terminal)

Start the worker detached from the SSH session, with the R2 key loaded from the 0600 file and nothing secret on the command line. Its exit status lands in `run.log` as `WORKER_EXIT=<code>`. As in step 7, the launch returns at once: the guard is its own statement ending in `|| exit 1;`, and the command sent to the background is one `bash -c` with every redirect outside it, so it holds nothing of the SSH session open. The launch refuses to run when `/workspace/rehearsal/run.log` already exists (checked from the Mac, and again on the Pod in the launch command itself), so pasting this block again never starts a second worker: the wait follows the first one instead. Then wait for its first heartbeat, for at most 20 minutes, calling `keepalive` on each pass; the wait stops early, and prints the end of `run.log`, if the worker exits first:

```sh
if podssh 'test -e /workspace/rehearsal/run.log'; then
  echo "run $RUN: /workspace/rehearsal/run.log already exists, so a worker was launched on this Pod before; not launching a second one. The wait below follows that one."
else
  podssh "test ! -e /workspace/rehearsal/run.log || exit 1; setsid nohup bash -c 'set -a; . /workspace/rehearsal/r2-pod.env; set +a; cd /workspace/repo && python3 -m tools.rehearsal_run run \
    --specification $SPEC --binding /workspace/rehearsal/binding.json --binding-sha256 $BINDING_SHA256 \
    --root /workspace/rehearsal-data --initialization /workspace/rehearsal/initialization.pt \
    --initialization-sha256 $INIT_SHA256 --readiness /workspace/rehearsal/readiness-receipt.json \
    --preflight /workspace/rehearsal/preflight-a40/report.json --output /workspace/rehearsal-run; echo WORKER_EXIT=\$?' \
    > /workspace/rehearsal/run.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 1200 ))
until podssh 'test -s /workspace/rehearsal-run/heartbeat.json'; do
  if podssh 'grep -q ^WORKER_EXIT= /workspace/rehearsal/run.log'; then echo "run $RUN: the worker exited before its first heartbeat:"; podssh 'tail -n 40 /workspace/rehearsal/run.log'; break; fi
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: no heartbeat after 20 minutes:"; podssh 'tail -n 40 /workspace/rehearsal/run.log'; break; fi
  sleep 10; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone (watchdog receipt above); follow step 14"; break; fi
done
podssh 'test -s /workspace/rehearsal-run/heartbeat.json' && watchdog
```

If the wait printed `exited` or `no heartbeat`, copy the log (`podscp root@$POD_HOST:/workspace/rehearsal/run.log "$CTRL/run.log"`), terminate the Pod (step 13) and report; do not start the supervisor. If it printed `the Pod is gone`, do not start the supervisor either: follow step 14. Once the heartbeat exists, the last line re-baselines the watchdog, which gives step 10 a full 600 s to start the mirror; after a failed wait it does not, so the idle limit keeps running.

This launch does not source `/etc/rp_environment`, so `RUNPOD_POD_ID` is usually unset: the worker takes the Pod ID from the receipt's captured identity (and checks it against `RUNPOD_POD_ID` only when that is set), and writes under the receipt's `r2_run_prefix`. Before anything else it checks the receipt against the files it was given, requires `/workspace/rehearsal/revision.json` to equal the receipt's `revision`, and applies the launch rule; any mismatch ends it with no output directory (`run.log` says why).

## 10. Supervisor (Mac, billable, each terminal; second window per run)

Start this run's supervisor as soon as the worker's heartbeat exists. It mirrors the heartbeat to `$CTRL/supervisor/heartbeat.json` (from then on the watchdog needs no more re-baselining), guards the Pod through the API, reads `heartbeat.json` and `report.json` over SSH with bounded sizes, and copies a checkpoint over SSH only to rescue milestones R2 never verified (step 14). It ignores a heartbeat whose `pod_id` is not this Pod's. In the second window repeat step 2 for this run and then the reopened-terminal line of step 5 (`export CTRL=...`, then source `init.env`, `run.env` and `helpers.sh`), then:

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
python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print(h["status"], h["phase"], h["step"], h.get("pod_id"), h.get("r2_final"))' "$CTRL/supervisor/heartbeat.json"
podssh 'tail -n 2 /workspace/rehearsal/run.log; python3 -c "import json; r=json.load(open(\"/workspace/rehearsal-run/report.json\")); print(r.get(\"status\"), r.get(\"speed_check\"), [(s.get(\"name\"), s.get(\"updates\")) for s in r.get(\"stages\", [])])"'
tail -n 3 "$CTRL/watchdog.log"; cat "$CTRL/watchdog-receipt.json"
```

Expect `running` with a growing `step` (the global update count) and this Pod's ID, and `uploading` only at the very end. The speed-and-fit check reports after 900 seconds of training. Validation runs every 500 global updates, and run A's stage 2 continues A's count. At run A's stage boundary the worker validates both streams in full, saves `stage1-final` and uploads it to R2, then starts stage 2 with a fresh optimizer and warmup; the heartbeat stays `running` throughout and the supervisor does nothing. Checkpoints are saved every 1,800 seconds and uploaded in the background as `<stage>-step<global update>` labels under `rehearsal-runs/<run>/<pod id>/`, except within the last 900 s of a stage at the measured rate. A milestone whose upload fails is retried for the rest of the run.

## 12. Completion and verification (Mac, automatic)

At the end the worker resubmits every milestone not yet verified in R2 and drains its uploader until 360 s after its last GPU work (`DRAIN_SECONDS`; see **Upload window**). Its heartbeat reads `uploading` while any milestone is pending or in flight. It uploads the final milestone `<stage>-final` (run A: `stage2-final`; run B: `mixed-final`) and `report.json`, and checks both in R2 by HEAD; after the drain every R2 call uses a bounded store and none starts late, so its final heartbeat is written by 480 s after its last GPU work. Meanwhile the supervisor lists the run's prefix with the Mac's key and stamps transfer progress when a new object appears, which is what lets the watchdog keep the idle Pod for up to 900 s after its last GPU work. Only then does the worker write `awaiting_transfer` with `r2_final` (the report and checkpoint keys and SHA-256s) into its heartbeat. The supervisor then, with the Mac's own key: checks that the SSH copy of `report.json` hashes to `r2_final`; HEADs the final checkpoint, `report.json` and the final label's other files and compares the stored SHA-256 metadata; downloads `report.json` pinned to its hash and size; checks that the report closes this run (both stages for a completed run A, so stage 1's end never counts) and that every stage-final milestone is in R2 with the recorded hash; archives `report.json` and a small `summary.json` to `rehearsal-runs/<run>/<pod id>/final/` in `$PRIVATE_REPO` with byte-exact read-back; and only then marks `shutdown_ready`, which lets the guards delete the Pod and confirm a 404. `updates.jsonl` (the per-update loss curve) and every checkpoint stay in R2.

A mismatch never approves teardown: the supervisor writes `controller-error.json` and retries each cycle. The idle limit still deletes the Pod 600 seconds after its last GPU activity (up to 900 s while new objects keep reaching R2 or the supervisor's SSH pull keeps receiving bytes), so act within that window. To check by hand (also when the supervisor is down):

```sh
uv run --locked python -m tools.rehearsal_ops verify-final --run "$RUN" --pod-id "$POD_ID" \
  --report-sha256 <r2_final.report_sha256> --checkpoint-sha256 <r2_final.checkpoint_sha256> --output "$CTRL/manual-verification"
```

For a run A that closed at `stage1-final` (time-limited between its stages, or stopped in stage 2), add `--stage stage1`: `r2_final.checkpoint_key` names the closing stage.

## 13. Pod termination (Mac, each terminal)

The guards delete the Pod after `shutdown_ready`; the in-Pod timer deletes it at `SELF_DESTRUCT_DEADLINE` regardless. To confirm, read the Pod's status and its mirrored heartbeat. This block only reads; it never deletes anything. The status command prints `resource not found (HTTP 404)` once the Pod is gone:

```sh
uv run --locked python -m tools.runpod_pod_identity status --pod-id "$POD_ID"
python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print(h.get("status"), h.get("phase"), h.get("pod_id"))' "$CTRL/supervisor/heartbeat.json"
```

To terminate by hand, use the block below. It sends the DELETE only when the mirrored heartbeat the watchdog reads, `$CTRL/supervisor/heartbeat.json`, says `shutdown_ready` for this Pod with `mac_receipt_verified` and `github_receipt_verified` both true (exactly what the watchdog requires before it deletes on that status), or when the owner has decided to end the Pod and you typed `TERMINATE=yes` in this terminal first. That decision covers these cases only:

- **Before the supervisor ran.** A step from 5 to 9 told you to terminate this Pod: a refused controller directory, no RUNNING within 20 minutes, no public IP, a changed host key, a failed or stuck bootstrap, a refused receipt, or no worker heartbeat. No worker has trained on the Pod, so no milestone is at risk.
- **A stuck or failed run with no milestone at risk.** `$CTRL/supervisor/MILESTONES-NOT-SECURED.json` does not exist, and the owner has checked with `verify-final` or `tools.r2_store status` (steps 12 and 14) that every stage-final milestone the run reached is in R2.
- **After step 14's rescue.** Every file that `MILESTONES-NOT-SECURED.json` names is on the Mac with its SHA-256 checked.

The Runpod key again reaches `curl` on its standard input; the DELETE prints its HTTP code, which should be `204`. The block's first line reads the decision into `DECISION` and clears `TERMINATE` at once, so the decision never carries over to a later paste, even when the block stops partway. It records when this Pod was confirmed gone, in `run.env` as `TEARDOWN`, only once the status shows HTTP 404. Then it stops this Pod's watchdog, if one still runs: only the process `watchdog.pid` names, and only when its command line names this Pod's identity file, the same check the `watchdog` helper makes; a PID file left by another process never gets that process killed:

```sh
DECISION=${TERMINATE:-}; unset TERMINATE
APPROVAL=none
python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); sys.exit(0 if h.get("status") == "shutdown_ready" and h.get("pod_id") == sys.argv[2] and h.get("mac_receipt_verified") is True and h.get("github_receipt_verified") is True else 1)' "$CTRL/supervisor/heartbeat.json" "$POD_ID" 2>/dev/null && APPROVAL=shutdown_ready
[ "$DECISION" = yes ] && APPROVAL=TERMINATE=yes
if [ -n "$POD_ID" ] && [ "$APPROVAL" != none ]; then
  echo "run $RUN: deleting Pod $POD_ID ($APPROVAL)"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS -o /dev/null -w '%{http_code}\n' -X DELETE "https://api.runpod.io/v2/pods/$POD_ID"
else
  echo "run $RUN: no DELETE sent: the mirrored heartbeat is not a verified shutdown_ready for Pod $POD_ID, and TERMINATE was not yes"
fi
POD_STATUS=$(uv run --locked python -m tools.runpod_pod_identity status --pod-id "$POD_ID" 2>&1); printf '%s\n' "$POD_STATUS"
if printf '%s\n' "$POD_STATUS" | grep -qF 'resource not found (HTTP 404)'; then
  export TEARDOWN=$(date +%s) && printf 'export TEARDOWN=%s\n' "$TEARDOWN" >> "$CTRL/run.env"
  WATCHDOG_PID=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
  if [ -n "$WATCHDOG_PID" ] && ps -p "$WATCHDOG_PID" -o command= 2>/dev/null | grep -qF -- "--identity $CTRL/pod-identity.json"; then kill "$WATCHDOG_PID"; fi
else
  echo "run $RUN: Pod $POD_ID is not confirmed gone; TEARDOWN is not recorded and its watchdog keeps running"
fi
```

A Pod the guards already deleted shows the 404 without a DELETE, and the block records its teardown all the same.

After both Pods are gone, revoke the Pods' R2 token in the Cloudflare dashboard and remove `R2_POD_*` from `.env`. Check under Storage that the account holds no network volume.

## 14. If one run fails

The runs are independent: never stop, restart or re-baseline the other run's watchdog or supervisor because one failed, and never reuse a Pod or a controller directory for another Pod.

- **Bootstrap or receipt fails.** Read `$CTRL/bootstrap.log` or the receipt tool's message, terminate that Pod (step 13) and report. A new Pod for the same run is a new launch: ask the owner first. It starts again at step 5 with a new controller directory, once step 13 has recorded the old Pod's `TEARDOWN` and `unset POD_ID` has cleared its ID; step 5 refuses to create while either is missing. The initialization and binding are unchanged and can be reused.
- **The worker stops or fails** (speed-and-fit check, rising loss, nonfinite loss, cost, a crash). After its own drain (heartbeat `uploading`) the heartbeat shows `stopped` or `failed`. The worker publishes `r2_final` only when a verified stage-final milestone closes what it trained (for example `stage1-final` of a run A stopped in stage 2); then the supervisor verifies it as in step 12 and archives it. Otherwise it waits 60 seconds (`--failure-grace-seconds`), then archives the worker's own `report.json` and a summary listing the labels present in R2 to `.../diagnostics/`, and approves teardown. The last verified rolling checkpoint and the report stay in R2.
- **A milestone never reached R2** (heartbeat `failed`, failure `r2_milestones_unverified`). The worker writes `finalization.json` with the local export path and SHA-256 of every milestone R2 never verified. The supervisor does not approve teardown on its grace timer: it pulls those exports over SSH with the pilot's verified pull path in the 420 s reserve (from the worker's final heartbeat, by 480 s after its last GPU work, to the end of the 900 s window; see **Upload window**), checks each SHA-256, archives the records, and only then sets `shutdown_ready`. While the pulled bytes grow it stamps transfer progress every 5 s, so the watchdog keeps the Pod past the 600 s idle limit. If the pull fails too, it writes a prominent `controller-error.json` and `MILESTONES-NOT-SECURED.json` naming the files and never sets `shutdown_ready`; the watchdog's wall, cost and transfer limits remain the backstop, so act within them. To rescue a milestone by hand, copy each `pod_path` the file names from under the worker's root, for example `podscp root@$POD_HOST:/workspace/rehearsal-run/exports/<label>/checkpoint.pt "$CTRL/rescued-<label>.pt"`, and check that `shasum -a 256` prints the `sha256` it records; only then may step 13 end the Pod with `TERMINATE=yes`.
- **The supervisor reports an R2 mismatch.** It never approves teardown. Inspect with `verify-final` and `uv run --locked python -m tools.r2_store status`; the idle limit deletes the Pod 600 seconds after its last GPU activity.
- **A Pod disappears or a watchdog exits** (also when a wait loop printed `the Pod is gone`). Read `$CTRL/watchdog-receipt.json`, the `$CTRL/watchdog-receipt.json.prev-<time>` copies (each `watchdog` restart keeps the receipt it replaces, so the reason a watchdog gave for deleting the Pod survives any later restart) and `$CTRL/supervisor/watchdog-receipt.json`, then confirm with the `status` command in step 13. A Pod that answers HTTP 404 is gone: run step 13's terminate block, which records its `TEARDOWN`. An `identity_mismatch` means the Pod ID no longer answers with the captured identity: ask the owner. Restart only that run's watchdog with `watchdog` (in a reopened terminal, after the reopened-terminal line of step 5), and only while its Pod still exists: `keepalive` never restarts a watchdog whose receipt reads `terminated`, `absent` or `identity_mismatch`.
- **Spend.** Each run's limits are its own; a failed run's spend counts toward its own $10 cap. Do not relaunch a failed run without the owner's approval.

## 15. Final billing verification (Mac, each terminal)

Fetch every hourly bucket from the hour of `ORIGIN` to now, and check that the provider's records cover the Pod's whole lifetime. `BUCKETS` counts enough hourly buckets to reach back to the hour of `ORIGIN`:

```sh
export BUCKETS=$(( ($(date +%s) - ORIGIN) / 3600 + 2 ))
uv run --locked python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --last-buckets "$BUCKETS" --output "$CTRL/billing.json"
python3 - "$CTRL/billing.json" "$POD_ID" "$ORIGIN" "${TEARDOWN:-}" "$BUCKETS" "$CTRL/supervisor/heartbeat.json" <<'PY'
import calendar, json, sys, time

path, pod_id, origin, teardown, buckets, heartbeat = sys.argv[1:7]
report = json.load(open(path))
fetched = float(report["fetched_at_unix"])
origin, buckets = float(origin), int(buckets)
teardown = float(teardown) if teardown else fetched


def hour(moment):
    return int(moment // 3600) * 3600


def unix(text):
    return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))


def stamp(moment):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))


alive = origin
try:
    alive = max(alive, float(json.load(open(heartbeat))["updated_at_unix"]))
except (OSError, ValueError, KeyError, TypeError):
    pass
problems = []
if report.get("pod_id") != pod_id:
    problems.append("billing.json was fetched for another Pod")
if hour(fetched) - (buckets - 1) * 3600 > hour(origin):
    problems.append("the request did not reach back to the hour of ORIGIN")
records = report.get("records") or []
starts = []
for record in records:
    start, end = unix(record["startTime"]), unix(record["endTime"])
    if record.get("podId") != pod_id:
        problems.append(f"the record at {record['startTime']} names another Pod")
    if end - start != 3600 or start % 3600:
        problems.append(f"the record at {record['startTime']} is not one hourly bucket")
    if start < hour(origin) or start > teardown:
        problems.append(f"the record at {record['startTime']} lies outside this Pod's lifetime")
    starts.append(start)
if len(set(starts)) != len(starts):
    problems.append("an hour is billed twice")
missing = [h for h in range(hour(origin + 300), hour(alive) + 1, 3600) if h not in starts]
if missing:
    problems.append("no record for the hours from " + ", ".join(stamp(h) for h in missing))
total = sum(float(record.get("totalAmount") or 0) for record in records)
provider = float((report.get("totals") or {}).get("totalAmount") or 0)
if abs(total - provider) > 1e-6:
    problems.append(f"the records sum to {total} but the provider totals say {provider}")
if total >= 10.0:
    problems.append(f"the Pod cost {total:.4f} USD, at or above its $10 cap")
covered = f"{stamp(hour(origin))} to {stamp(hour(alive) + 3600)}"
print(json.dumps({"hours": len(starts), "covered": covered, "total_usd": round(total, 4)}))
if problems:
    sys.exit("billing does NOT cover this Pod's lifetime:\n- " + "\n- ".join(problems))
print(f"billing covers this Pod from {stamp(origin)} to its last heartbeat; under the $10 cap")
PY
```

The check fails unless the request reached back to the hour of `ORIGIN`, every record is this Pod's and lies within its lifetime (from `ORIGIN` to `TEARDOWN`, recorded in step 13), every hour from `ORIGIN` to the last mirrored heartbeat has a record (the Pod certainly billed until that heartbeat; billing starts with the create request, and a Pod created in the hour after `ORIGIN`'s skips that hour, so the first hour required is the one that holds `ORIGIN` + 300 s), the records sum to the provider totals, and the total is below the $10 cap. Runpod bills every five minutes and records may lag: a check that fails right after teardown only for the last hour is re-run an hour later (the same block, which recomputes `BUCKETS`); a failure that persists means the records do not cover this Pod, so report it. Compare the total with the report's `cost` for each run, and note the R2 storage held by `uv run --locked python -m tools.r2_store status`.

## 16. Results

From each archived `summary.json` and the R2 `report.json`: per stage the updates, tokens, learning rate, warmup, ticket and its match, the final checkpoint SHA-256, the validations every 500 global updates and the full validation at each stage end; plus the speed check, cost, timings and `code_revision`. The per-update loss curves are in `rehearsal-runs/<run>/<pod id>/<stage>-final/updates.jsonl` in R2. Record the measured numbers in the docs as measured rehearsal results, distinct from the projections in [docs/rehearsal-runs.md](docs/rehearsal-runs.md).
