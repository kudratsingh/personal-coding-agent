# Post-training runbook: run 1 on one A40 (probe, ARC, evaluation, SFT, RL screening)

Operational sequence for post-training run 1 of the 135M-parameter S rehearsal model, on one Runpod A40 Pod, from run B's `mixed-final` checkpoint. The worker (`tools/posttrain_run.py`) runs seven stages in order and uploads every artifact to R2 under `posttrain-runs/1/<pod id>/` as soon as it exists:

1. **probe:** generation speed at batch 1, 64, 128 and 256 on 256 real chat-v1 prompt lengths around the pool dev median, top-p 0.95, bf16. Below 8,279 generated tok/s at batch 128 or 9,268 at batch 256 the run stops here (`generation_below_planning_speed`) with the measured rates in `report.json`, at most about $0.39 in (the 18-minute setup allowance, the 6-minute probe and the whole 15-minute finalization reserve at $0.60/h; step 3's projection prints it). The floors are 5% under the rates the first launch's probe measured, 8,715 tok/s at batch 128 and 9,757 at batch 256; that launch stopped here below both of its floors, 5,000 at batch 64 (it measured 4,607 tok/s there) and 10,000 at batch 256, so evaluation and screening now generate at batch 128.
2. **arc-base:** ARC-Easy label log-likelihood (float32) on run A's `stage1-final` and `stage2-final` and run B's `mixed-final`.
3. **eval-base:** base-B completions at batch 128 in bf16: pool dev and test greedy plus n = 8 (temperature 0.8, top-p 0.95, seed 0) under `base-pool-completion-v1`; HumanEval, HumanEval+ (task 32 excluded) and MBPP+ greedy under `base-completion-rstrip-v1`.
4. **sft:** one pass over the 3,390 SFT rows (106 updates of 16 x 2 rows), peak 6e-5, 3% warmup, cosine to 10%, a fresh AdamW, compiled; assistant-token dev loss at each quarter, the lowest kept as weights only (about 541 MB); after 50 updates the slot throughput must reach 90% of 56,765 slots/s or the run stops (`sft_throughput_below_limit`).
5. **arc-sft:** ARC-Easy on the selected SFT weights.
6. **eval-sft:** the same seven completions files from the SFT weights, under `chat-v1`.
7. **screen:** RL screening attempts from the SFT weights on the 1,000 candidates (8 samples, temperature 1.0, top-p 1.0, at most min(256, 1,024 - prompt) new tokens, at batch 128), for `scratchcoder rl-screen` on the Mac.

Then the pretraining rehearsal's finalization: the drain, the HEAD checks, `report.json` and the handover to the Mac supervisor, which verifies everything in R2 before it lets the Pod be deleted.

Commands marked **Mac** run on the owner's Mac inside the repository checkout; **Pod** commands run inside the rented Pod over SSH. One terminal drives the run, a second runs the supervisor (step 12). Nothing in steps 1 to 6 starts a Pod; steps 4 and 5 store inputs in R2 (cents a month). Step 7 creates the Pod and starts billing: do not execute it without the owner's explicit go for run 1. Runpod offers no provider-enforced Pod lifetime: termination is layered best-effort (the in-Pod timer and idle rule, the Mac watchdog, the supervisor's guard, the worker's own limits).

**Limits, all derived from the $3.00 ceiling** (`tools/posttrain_ops.py` `ceiling_limits`; `limits` prints them): at the quoted $0.60/h (GPU $0.59/h plus the 40 GB disk) the worker stops at **$3.00**, the Mac watchdog at **$2.99**, the Pod wall is **18,000 s**, and the in-Pod timer fires at **17,820 s**, 180 s inside it. The fixed 60,000 s of the pretraining rehearsal is never used: it stays only as the in-Pod timer's absolute cap. The watchdog's idle limit is 600 s, as in the pretraining rehearsal; the in-Pod timer also deletes the Pod after **900 s** with no GPU work and no progress (the worker's heartbeat, or the bootstrap's own progress stamps), whatever the Mac does.

**The time rules.** Both the readiness receipt (step 10) and the worker at launch (step 11) apply one **launch rule**: the Pod lifetime left must hold every stage at the planning rates (generation at 8,279 tok/s at batch 128 with every pool attempt priced at its token cap, SFT at 56,765 slots/s, ARC and checkpoint loads at their planning seconds) plus the 900 s finalization reserve: 6,878 s, so setup may take up to about 182 minutes of the 17,820 s. Then the **projected-finish rule**: at the start of every stage the worker stops with `projected_finish_after_deadline` if the work remaining, at the probe's measured generation rate (or the completed evaluation files' observed rate once lower) and the measured SFT and ARC rates once known, plus the reserve, would pass the end of the lifetime. A stage still running 900 s before the end of the lifetime stops there (`finalization_reserve`) and finalizes in full, so `report.json` and `r2_final` always reach R2 before the in-Pod timer fires. The projection for the run at the planning speeds (step 3) is about 2.2 hours and $1.33.

**Upload window and rate.** The final drain, the checks and the Mac's SSH reserve split the pretraining rehearsal's 900 s window as before (360 s, 120 s, 420 s). Every post-training label is small except one weights-only label of about 541 MB (SFT, RL and more-SFT start fresh optimizers), so the receipt requires an R2 upload rate of `posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND`, one 0.55 GB label in the 360 s drain with a factor-2 margin: about **3.06 MB/s** (the pretraining rehearsal's 1.65 GB labels needed 9.17 MB/s). The run's only weights label is uploaded about 45 minutes before the final drain (ARC on the SFT weights, the SFT evaluation and the screening at the planning rates).

**Hidden tests never leave the Mac.** The Pod receives Pod exports (`scratchcoder pod-task-export`): the pool dev and test splits and the RL candidates with every hidden test replaced by a placeholder naming only its salted SHA-256. `eval-execute` and `rl-screen` derive the export again from the full task file they are given and compare SHA-256s; the export's record is never trusted, and an export given as the full file is refused. The real test split is touched only in step 4, at the go, to export it, and its completions are sealed (step 17), not executed.

**R2 safeguards.** The Pod's token (`R2_POD_ACCESS_KEY_ID`/`R2_POD_SECRET_ACCESS_KEY`) is its own bucket-scoped Cloudflare token for this run only, shipped over SSH stdin, revoked at teardown and kept out of `.env` during the Mac execution windows. The worker refuses any R2 write or delete outside `posttrain-runs/1/<pod id>/` before a request is sent (`posttrain_ops.GuardedStore`).

## 1. Prelaunch checks (Mac, free)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check
uv run --locked python -m tools.posttrain_run smoke --output runs/posttrain-smoke
uv run --locked python -m tools.pilot_preflight --output runs/posttrain-preflight-cpu --device cpu
uv run --locked pytest tests/test_posttrain_run.py tests/test_posttrain_integration.py tests/test_posttrain_guards.py tests/test_posttrain_binding.py tests/test_posttrain_readiness.py tests/test_posttrain_bootstrap.py tests/test_posttrain_feasibility.py tests/test_posttrain_runbook.py tests/test_pod_task_export.py tests/test_mac_window_check.py tests/test_pod_self_destruct.py tests/test_runpod_watchdog.py -q
```

Expected: `make check` green; the smoke prints `"status": "completed"` with every stage `completed` and `"objects_outside_prefix": []`; the CPU preflight shows `"status": "completed"`; every listed test passes. The runbook test parses every tool command in this file with the real parsers and checks that every block parses in bash and zsh and holds no `#` comment (zsh on macOS has `interactivecomments` off). The integration test replays the smoke's heartbeats into the real supervisor and exercises each stop.

## 2. Credentials and environment (Mac, free)

`.env` is git-ignored. It holds `RUNPOD_API_KEY`, `R2_ENDPOINT`, `R2_BUCKET`, the Mac's own R2 key `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY` and, for this run only, the Pod's token `R2_POD_ACCESS_KEY_ID`/`R2_POD_SECRET_ACCESS_KEY`: a new Cloudflare API token with Object Read & Write on this one bucket. It reaches the Pod only over SSH stdin (step 8) and is revoked in step 15.

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi
done
gh auth status
caffeinate -dimsu -w $$ &
export RUN=1
export CTRL_ROOT=runs/posttrain-controller
export KNOWN_HOSTS=~/.ssh/posttrain_1_known_hosts
export SPEC=configs/rehearsal-a40-s-post-1.json
export SHARED=runs/posttrain-shared && mkdir -p "$SHARED"
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts
export PUBLIC_REPO=kudratsingh/personal-coding-agent
export POOL=runs/task-pool-20261025/pool
export SFT=runs/sft-data-20261026
export ARC_DATA=runs/arc-review
export TOKENIZER=runs/tokenizer-freeze-65536-20261020/tokenizer.json
export BINDING=$SHARED/binding.json
export WALL=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field wall_seconds)
export WATCHDOG_COST=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field watchdog_cost_stop_usd)
export TIMER=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field self_destruct_seconds)
echo "run $RUN: wall $WALL s, timer $TIMER s, watchdog cost stop $WATCHDOG_COST USD"
```

The loop prints each variable's name and whether it is set, never a value. The last line must print `wall 18000 s, timer 17820 s, watchdog cost stop 2.99 USD`. `PRIVATE_REPO` must be private (the supervisor verifies it). Keep this terminal open on AC power for the whole run: a sleeping Mac stops the mirror, and the watchdog would then read a stale heartbeat and delete the Pod. The working directory must not sit under a symlinked path.

## 3. Quote check and projection (Mac, free)

Open the Runpod console: Secure Cloud, NVIDIA A40, on-demand, 1 GPU, container disk 40 GB, no volume. Record the hourly price, then project run 1 at it:

```sh
uv run --locked python -m tools.pilot_feasibility posttrain --specification "$SPEC" --quoted-gpu-hourly 0.59 --output "$SHARED/feasibility.json"
```

Replace `0.59` with the console price; above $0.59, stop and ask the owner. The table gives hours and dollars per stage at half, the planning and twice the planning generation speed; at the planning speeds run 1 is about 2.2 hours and $1.33 against the $3.00 ceiling, and at half speed the probe stops the run after at most about $0.39. Confirm the account balance covers $3.00 plus a margin and any other project's Pod over the time until run 2, and that no network volume exists.

## 4. Pod exports and benchmark releases (Mac, at the go, free)

Export the pool dev split, the RL candidates and the ARC-Easy test questions, and check each against the specification's pin; then export the pool test split. **The test export is the one launch-time touch of the sealed split**: it reads `test.jsonl` once to replace its hidden tests, and nothing is scored. Download the pinned benchmark releases (size- and SHA-256-checked):

```sh
export EXPORTS=$SHARED/exports && mkdir -p "$EXPORTS"
uv run --locked scratchcoder pod-task-export --kind pool --tasks "$POOL/dev.jsonl" --output "$EXPORTS/dev-export.jsonl" --record "$EXPORTS/dev-export.json"
uv run --locked scratchcoder pod-task-export --kind candidates --tasks "$POOL/train.jsonl" --candidates "$SFT/rl_candidates.jsonl" --output "$EXPORTS/candidates-export.jsonl" --record "$EXPORTS/candidates-export.json"
uv run --locked --extra evaluation scratchcoder pod-task-export --kind arc --arc-directory "$ARC_DATA" --output "$EXPORTS/arc-easy-test.jsonl" --record "$EXPORTS/arc-export.json"
python3 -c 'import hashlib,json,sys; s=json.load(open(sys.argv[1]))["inputs"]; bad=[n for n,p in (("pool-dev",sys.argv[2]),("rl-candidates",sys.argv[3]),("arc-easy-test",sys.argv[4])) if hashlib.sha256(open(p,"rb").read()).hexdigest()!=s[n]["sha256"]]; print("exports match the specification" if not bad else "MISMATCH: " + ", ".join(bad))' "$SPEC" "$EXPORTS/dev-export.jsonl" "$EXPORTS/candidates-export.jsonl" "$EXPORTS/arc-easy-test.jsonl"
uv run --locked scratchcoder pod-task-export --kind pool --tasks "$POOL/test.jsonl" --output "$EXPORTS/test-export.jsonl" --record "$EXPORTS/test-export.json"
uv run --locked python -m tools.posttrain_binding check-exports --specification "$SPEC" --input pool-dev="$EXPORTS/dev-export.jsonl" --input pool-test="$EXPORTS/test-export.jsonl" --input rl-candidates="$EXPORTS/candidates-export.jsonl"
uv run --locked python -m tools.posttrain_binding benchmarks --output "$SHARED/benchmarks"
```

The pin check must print `exports match the specification`; anything else means the export code or a source file changed since the specification was pinned: stop and ask the owner. The test export has no pin until step 5, so `check-exports` is what keeps the sealed split's hidden tests on the Mac: it must print `{"exports": {"pool-dev": 2000, "pool-test": 2000, "rl-candidates": 1000}, "hidden_tests": 0}`, and it refuses any file in which a task still carries its hidden tests, naming task IDs only. The binding (step 5) and the Pod's verification repeat the check. The ARC export needs the optional `evaluation` extra (PyArrow) and the three pinned ARC-Easy Parquet files in `$ARC_DATA`.

## 5. Binding and staging (Mac, at the go; R2 storage only)

Bind every input by R2 key, size and SHA-256. The three parent checkpoints and the tokenizer are already in R2 at the keys the specification pins (run A's and run B's controller records give their labels); every other input is staged under `posttrain-inputs/<sha256>/<file>`. The binding refuses any file whose size or SHA-256 differs from a pin the specification holds:

```sh
uv run --locked python -m tools.posttrain_binding build --specification "$SPEC" --run 1 --output "$BINDING" \
  --input sft-train="$SFT/train.jsonl" --input sft-code-dev="$SFT/code_dev.jsonl" \
  --input sft-language-dev="$SFT/language_dev.jsonl" --input rl-candidates-list="$SFT/rl_candidates.jsonl" \
  --input pool-dev="$EXPORTS/dev-export.jsonl" --input pool-test="$EXPORTS/test-export.jsonl" \
  --input rl-candidates="$EXPORTS/candidates-export.jsonl" --input arc-easy-test="$EXPORTS/arc-easy-test.jsonl" \
  --input humaneval="$SHARED/benchmarks/HumanEval.jsonl.gz" \
  --input humaneval-plus="$SHARED/benchmarks/HumanEvalPlus-OriginFmt.jsonl.gz" \
  --input mbpp-plus="$SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz"
```

The owner reviews the binding and the `sha256` it printed. Paste that value, never one derived from the file, then stage the inputs (each re-hashed first) and HEAD the existing ones with the Mac's key:

```sh
export BINDING_SHA256=<reviewed binding sha256>
printf '%s  %s\n' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -
uv run --locked python -m tools.posttrain_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --receipt "$SHARED/stage-receipt.json"
```

`shasum` must print `OK`.

## 6. The Mac's copy of the parent checkpoints (Mac, before the Pod; R2 egress is free)

Before any Pod exists, copy run B's `mixed-final` and run A's two finals to the Mac, each pinned to its size and SHA-256, with the Mac's own key. The readiness receipt requires this copy's receipt (`mac_parent_copies_verified`):

```sh
uv run --locked python -m tools.posttrain_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --root runs/posttrain-parents --only a-stage1-final --only a-stage2-final --only b-mixed-final --receipt "$SHARED/mac-copy.json" --key-prefix R2_
```

## 7. Pod creation, identity and the watchdog (Mac, BILLABLE)

Only after the owner's explicit go. The block refuses to create a Pod when this terminal already names one or when a controller directory has no `TEARDOWN`, exactly as the rehearsal runbook's step 5 does. The timer deadline and the policy come from the specification's ceiling:

```sh
OPEN_PODS=$(find "$CTRL_ROOT/$RUN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r DIR; do grep -qs '^export TEARDOWN=' "$DIR/run.env" || printf '%s ' "$DIR"; done)
if [ -n "${POD_ID:-}" ]; then
  echo "run $RUN: NOT creating a Pod: this terminal already names Pod $POD_ID, which may still bill. Confirm it gone with step 15 first."
elif [ -n "$OPEN_PODS" ]; then
  echo "run $RUN: NOT creating a Pod: no TEARDOWN is recorded in ${OPEN_PODS% }. End each such Pod with step 15 and ask the owner before creating another."
else
  export ORIGIN=$(date +%s)
  export LAUNCH=$CTRL_ROOT/$RUN/launch-$ORIGIN && mkdir -p "$LAUNCH"
  cp configs/runpod-pod-a40-s-posttrain.json "$LAUNCH/create-body.json"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS --fail-with-body -X POST https://api.runpod.io/v2/pods \
    -H "Content-Type: application/json" --data @"$LAUNCH/create-body.json" -o "$LAUNCH/create-response.json" -w '%{http_code}\n' > "$LAUNCH/create-http-code"
  export POD_ID=$(python3 -c 'import json,re,sys; v=json.load(open(sys.argv[1])); i=v.get("id") if isinstance(v, dict) else None; ok=isinstance(i, str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,127}", i); print(i) if ok else sys.exit("no valid Pod id in " + sys.argv[1])' "$LAUNCH/create-response.json")
  export SELF_DESTRUCT_DEADLINE=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --origin "$ORIGIN" --field self_destruct_deadline_unix)
  export CTRL=$CTRL_ROOT/$RUN/$POD_ID
  if [ -z "$POD_ID" ]; then
    echo "run $RUN: the create request ended with HTTP $(cat "$LAUNCH/create-http-code") and names no valid Pod id. A Pod MAY STILL EXIST: ask the owner to check the Runpod console for posttrain-a40-s-1 and decide before any retry."
    export CTRL=$LAUNCH
  elif mkdir "$CTRL"; then
    mv "$LAUNCH/create-body.json" "$LAUNCH/create-response.json" "$LAUNCH/create-http-code" "$CTRL/" && rmdir "$LAUNCH"
    printf 'export ORIGIN=%s POD_ID=%s SELF_DESTRUCT_DEADLINE=%s\n' "$ORIGIN" "$POD_ID" "$SELF_DESTRUCT_DEADLINE" > "$CTRL/run.env"
    echo "run $RUN: Pod $POD_ID, controller directory $CTRL, timer deadline $SELF_DESTRUCT_DEADLINE"
  else
    echo "run $RUN: $CTRL already exists and is NOT this Pod's. Terminate Pod $POD_ID with step 15 (TERMINATE=yes)."
    export CTRL=$LAUNCH
  fi
fi
```

`SELF_DESTRUCT_DEADLINE` is `ORIGIN` + 17,820 s. Capture the identity, write the watchdog policy from the derived limits, write the helpers (the rehearsal runbook's step 5 helpers with the post-training mode and the derived wall and cost stop), and start the watchdog at once. With `--policy` the watchdog refuses to start unless its flags give exactly the limits of `policy.json`, the file the readiness receipt checks:

```sh
uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" \
  --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json"
uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --origin "$ORIGIN" --idle-baseline "$(date +%s)" --policy-output "$CTRL/policy.json"
[ -e "$CTRL/helpers.sh" ] || cat > "$CTRL/helpers.sh" <<'EOF'
watchdog() {
  OLD=$(cat "$CTRL/watchdog.pid" 2>/dev/null || true)
  BASE=$(date +%s)
  if [ -e "$CTRL/watchdog-receipt.json" ]; then cp "$CTRL/watchdog-receipt.json" "$CTRL/watchdog-receipt.json.prev-$BASE"; fi
  nohup uv run --locked python -m tools.runpod_watchdog --mode posttrain --identity "$CTRL/pod-identity.json" \
    --heartbeat "$CTRL/supervisor/heartbeat.json" --receipt "$CTRL/watchdog-receipt.json" \
    --pidfile "$CTRL/watchdog.pid" --max-wall-seconds "$WALL" --max-idle-seconds 600 \
    --max-transfer-seconds 900 \
    --max-cost-usd "$WATCHDOG_COST" --quoted-total-hourly 0.60 --billing-started-at-unix "$ORIGIN" \
    --idle-baseline-unix "$BASE" --poll-seconds 20 --policy "$CTRL/policy.json" \
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
  if [ -z "$LIVE" ] || ! ps -p "$LIVE" -o command= 2>/dev/null | grep -qF -- "--identity $CTRL/pod-identity.json"; then
    GONE=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); gone=isinstance(r, dict) and r.get("pod_id") == sys.argv[2] and r.get("state") in ("terminated", "absent", "identity_mismatch"); print("state %s, reason %s" % (r.get("state"), r.get("reason"))) if gone else None' "$CTRL/watchdog-receipt.json" "$POD_ID" 2>/dev/null)
    if [ -n "$GONE" ]; then
      echo "STOP run $RUN: POD GONE: the watchdog of Pod $POD_ID ended for good with $GONE. It is NOT restarted. Follow step 16."
      return 3
    fi
    echo "WARNING run $RUN: NO LIVE WATCHDOG guards Pod $POD_ID. Restarting it now."
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

**A reopened terminal** repeats step 2 and then sources this Pod's files, and never re-runs a block that creates a Pod or writes `run.env`, the policy or the known-hosts file: `export CTRL=$CTRL_ROOT/$RUN/<pod id>; export BINDING_SHA256=<reviewed binding sha256>; source "$CTRL/run.env"; source "$CTRL/helpers.sh"`.

Wait, for at most 20 minutes, until the Pod is RUNNING with a direct SSH host and port, re-baselining the watchdog on every pass, then pin its host key and compare the fingerprint with the one in the Pod's web terminal:

```sh
GIVE_UP=$(( $(date +%s) + 1200 ))
while :; do
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); d=c.get("ssh_direct") or {}; ok=c.get("status") == "RUNNING" and bool(d.get("host")) and bool(d.get("port")); print(c.get("status"), "with direct SSH" if ok else "without direct SSH"); sys.exit(0 if ok else 1)' "$CTRL/pod-connection.json" && break
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: Pod $POD_ID is not RUNNING with direct SSH after 20 minutes; terminate it (step 15, TERMINATE=yes)"; break; fi
  sleep 20
  uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json" > /dev/null
  keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 16"; break; fi
done
export POD_HOST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["host"])' "$CTRL/pod-connection.json")
export POD_PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["port"])' "$CTRL/pod-connection.json")
printf 'export POD_HOST=%s POD_PORT=%s\n' "$POD_HOST" "$POD_PORT" >> "$CTRL/run.env"
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS" 2>/dev/null
ssh-keygen -lf "$KNOWN_HOSTS"
watchdog
```

## 8. Code, binding and the Pod's R2 key (Mac to Pod, billable)

Record exactly which code runs: the writer refuses a dirty tree, archives `HEAD` and writes `revision.json` with the commit, `"clean": true` and the archive's SHA-256 (the rehearsal runbook's step 6 writer). Then ship the archive, the revision and the binding, and deliver the Pod's R2 key over SSH stdin into a 0600 file:

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
podssh 'mkdir -p /workspace/repo /workspace/posttrain /workspace/posttrain-inputs'
podscp "$CTRL/repo.tar.gz" root@$POD_HOST:/workspace/repo.tar.gz
podscp "$CTRL/revision.json" root@$POD_HOST:/workspace/revision.json
podscp "$BINDING" root@$POD_HOST:/workspace/posttrain/binding.json
uv run --locked python -m tools.rehearsal_ops pod-env | podssh_stdin 'umask 077 && cat > /workspace/posttrain/r2-pod.env && chmod 600 /workspace/posttrain/r2-pod.env'
podssh 'stat -c "%a %U %s" /workspace/posttrain/r2-pod.env'
watchdog
```

The `stat` line should print `600 root` and a nonzero size. `pod-env` writes only `R2_ENDPOINT`, `R2_BUCKET` and the Pod's own token, refuses to write to a terminal and refuses the Mac's key.

## 9. Bootstrap (Pod, billable)

The bootstrap arms the in-Pod timer first, with the run's ceiling-derived timer lifetime (`$TIMER`, 17,820 s: the timer refuses any deadline further away) and the 900 s idle rule over the worker's heartbeat and its own progress stamps, then checks the shipped archive against the revision, loads the 0600 key file, records the environment, installs the project, measures an R2 round trip under `posttrain-runs/1/<pod id>/` (refused below 3.06 MB/s by the receipt), fetches every bound input (stamping its progress file after each one, so a live 1.6 GB download is never idle) and re-verifies them, runs the CPU tests and the A40 preflight with its compile check. Launch it detached, refusing a second launch, and keep re-baselining the watchdog while it runs, for at most 45 minutes:

```sh
if podssh 'test -e /workspace/posttrain/bootstrap.log'; then
  echo "run $RUN: a bootstrap was launched on this Pod before; the wait below follows that one."
else
  podssh "test ! -e /workspace/posttrain/bootstrap.log || exit 1; cd /workspace/repo && tar xzf /workspace/repo.tar.gz || exit 1; setsid nohup bash -c 'bash tools/pilot_pod_bootstrap.sh --mode posttrain --run $RUN --deadline-unix $SELF_DESTRUCT_DEADLINE --lifetime-seconds $TIMER --idle-seconds 900 --workspace /workspace/posttrain --repo /workspace/repo --revision /workspace/revision.json --archive /workspace/repo.tar.gz --r2-env /workspace/posttrain/r2-pod.env --binding /workspace/posttrain/binding.json --binding-sha256 $BINDING_SHA256 --data-root /workspace/posttrain-inputs --heartbeat /workspace/posttrain-run/heartbeat.json; echo BOOTSTRAP_EXIT=\$?' > /workspace/posttrain/bootstrap.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 2700 ))
until podssh 'grep -q ^BOOTSTRAP_EXIT= /workspace/posttrain/bootstrap.log'; do
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: bootstrap exceeded 45 minutes; terminate the Pod (step 15, TERMINATE=yes)"; break; fi
  sleep 30; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 16"; break; fi
done
for name in bootstrap.log self-destruct.json environment.json r2-round-trip.json data-fetch.json data-verification.json setup-progress.json; do
  podscp root@$POD_HOST:/workspace/posttrain/$name "$CTRL/$name"
done
podscp root@$POD_HOST:/workspace/posttrain/preflight-a40/report.json "$CTRL/preflight-a40-report.json"
podscp root@$POD_HOST:/workspace/posttrain/revision.json "$CTRL/pod-revision.json"
cmp "$CTRL/revision.json" "$CTRL/pod-revision.json" && echo "the Pod holds this revision"
for name in R2_POD_SECRET_ACCESS_KEY R2_SECRET_ACCESS_KEY RUNPOD_API_KEY; do printenv "$name"; done | grep -cFf - "$CTRL/bootstrap.log"
grep BOOTSTRAP_EXIT "$CTRL/bootstrap.log"; watchdog
```

Expected: `BOOTSTRAP_EXIT=0`; `self-destruct.json` shows `"state": "armed"`, `deadline_unix` equal to `SELF_DESTRUCT_DEADLINE`, `"idle_seconds": 900.0` and both activity files; `the Pod holds this revision`; `r2-round-trip.json` shows `"status": "passed"` and an `upload_bytes_per_second` of at least about 3.06 million; the preflight is `completed` with passed `fused_attention` and `compile_equivalence` checks; the leak check prints `0`. Any other exit: read `bootstrap.log`, then terminate the Pod (step 15).

## 10. Readiness receipt (Mac, billable)

The confirmations are the owner's statements; `r2_token_bucket_scoped` states that the Pod's token is a separate token limited to this bucket and this run. The round-trip, input and Mac-copy gates are derived from receipts, never typed. First stamp the Pod's setup progress (the in-Pod idle rule counts 900 s from the bootstrap's last stamp), copy the live timer's receipt, then build the receipt, which refuses anything the worker would refuse:

```sh
cat > "$CTRL/confirmations.json" <<'EOF'
{"owner_confirms_four_runpod_steps": true, "cloud_data_pre_staged": true,
 "external_watchdog_armed": true, "hardware_preflight_passed": true, "full_console_quote_confirmed": true,
 "staging_storage_cost_bound": true, "r2_token_bucket_scoped": true,
 "termination_guarantee": "redundant_best_effort", "owner_accepts_best_effort_termination": true}
EOF
podssh 'python3 /workspace/repo/tools/pod_progress.py --file /workspace/posttrain/setup-progress.json --source runbook:receipt'
podscp root@$POD_HOST:/workspace/posttrain/self-destruct.json "$CTRL/self-destruct.json"
uv run --locked python -m tools.pilot_readiness_receipt posttrain --specification "$SPEC" \
  --binding "$BINDING" --binding-sha256 "$BINDING_SHA256" --preflight "$CTRL/preflight-a40-report.json" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --self-destruct-receipt "$CTRL/self-destruct.json" --confirmations "$CTRL/confirmations.json" \
  --data-fetch "$CTRL/data-fetch.json" --data-verification "$CTRL/data-verification.json" \
  --round-trip "$CTRL/r2-round-trip.json" --mac-copy "$SHARED/mac-copy.json" \
  --revision "$CTRL/revision.json" --billing-origin-unix "$ORIGIN" --output "$CTRL/readiness-receipt.json"
podscp "$CTRL/readiness-receipt.json" root@$POD_HOST:/workspace/posttrain/readiness-receipt.json
watchdog
```

The printed line reports `fits_at_planning` (always true for a written receipt) and the seconds left. If the receipt refuses a stale timer receipt, paste the block again. Launch the worker promptly: it applies the launch rule again at its own start.

## 11. Worker launch (Pod, billable)

Start the worker detached, with the R2 key loaded from the 0600 file and nothing secret on the command line, refusing a second launch; then wait for its first heartbeat, for at most 20 minutes:

```sh
if podssh 'test -e /workspace/posttrain/run.log'; then
  echo "run $RUN: a worker was launched on this Pod before; the wait below follows that one."
else
  podssh "test ! -e /workspace/posttrain/run.log || exit 1; setsid nohup bash -c 'set -a; . /workspace/posttrain/r2-pod.env; set +a; cd /workspace/repo && python3 -m tools.posttrain_run run \
    --specification $SPEC --binding /workspace/posttrain/binding.json --binding-sha256 $BINDING_SHA256 \
    --root /workspace/posttrain-inputs --readiness /workspace/posttrain/readiness-receipt.json \
    --preflight /workspace/posttrain/preflight-a40/report.json --revision /workspace/posttrain/revision.json \
    --output /workspace/posttrain-run; echo WORKER_EXIT=\$?' > /workspace/posttrain/run.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 1200 ))
until podssh 'test -s /workspace/posttrain-run/heartbeat.json'; do
  if podssh 'grep -q ^WORKER_EXIT= /workspace/posttrain/run.log'; then echo "run $RUN: the worker exited before its first heartbeat:"; podssh 'tail -n 40 /workspace/posttrain/run.log'; break; fi
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: no heartbeat after 20 minutes:"; podssh 'tail -n 40 /workspace/posttrain/run.log'; break; fi
  sleep 10; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 16"; break; fi
done
podssh 'test -s /workspace/posttrain-run/heartbeat.json' && watchdog
```

Before anything else the worker checks the receipt against the files it was given, the revision, the binding against the specification, every input's bytes, the A40 preflight and the launch rule; any mismatch ends it with no output directory (`run.log` says why).

## 12. Supervisor (Mac, billable; second window)

Start the supervisor as soon as the heartbeat exists. It mirrors the heartbeat for the watchdog, guards the Pod through the API, reads the status files over SSH with bounded sizes, lists the run's prefix with the Mac's own key for transfer progress, and copies the weights label over SSH only to rescue it if R2 never verified it. In the second window repeat step 2 and the reopened-terminal line of step 7, then:

```sh
uv run --locked python -m tools.pilot_supervisor --mode posttrain --run "$RUN" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --identity-file "$POD_SSH_KEY" --known-hosts "$KNOWN_HOSTS" --output "$CTRL/supervisor" \
  --host root@$POD_HOST --port $POD_PORT --remote-root /workspace/posttrain-run \
  --repository "$PRIVATE_REPO" --poll-seconds 20 2>&1 | tee -a "$CTRL/supervisor.log"
```

It refuses to start without the Mac's own R2 key or a `"mode": "posttrain"` policy. If it stops, run the same command again: it continues in the same directory.

## 13. Monitoring (Mac, read-only)

```sh
python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print(h["status"], h["stage"], h["phase"], h["step"], h.get("generated_tokens"), h.get("r2_final"))' "$CTRL/supervisor/heartbeat.json"
podssh 'tail -n 2 /workspace/posttrain/run.log; python3 -c "import json; r=json.load(open(\"/workspace/posttrain-run/report.json\")); print(r.get(\"status\"), r.get(\"probe\", {}) and r[\"probe\"].get(\"measured_tokens_per_second\"), [(s[\"name\"], s[\"status\"]) for s in r[\"stages\"]])"'
tail -n 3 "$CTRL/watchdog.log"; cat "$CTRL/watchdog-receipt.json"
```

Expect `running` through every stage and `uploading` only at the very end. The probe reports first (about 6 minutes in); if it reports `below_planning_speed`, the run ends there by design. Each completions file appears in R2 as `eval-<stage>-<file>/completions.jsonl` as soon as it is written; the SFT weights as `sft-weights/checkpoint.pt`; the screening attempts as `screen/attempts.jsonl`.

## 14. Completion and verification (Mac, automatic)

At the end the worker drains its uploads, checks every label by HEAD, uploads `report.json` and writes `awaiting_transfer` with `r2_final` (the report's key and SHA-256 and the artifact count). The supervisor then checks that the SSH copy of `report.json` hashes to it, downloads the report pinned to it with the Mac's own key, HEADs every artifact the report lists and compares each SHA-256 and size, archives the report and a summary privately with read-back, and only then marks `shutdown_ready`; the watchdog deletes the Pod and confirms a 404. A stopped run (the probe, the speed check, the projected finish) that verified every label it made publishes `r2_final` too and is verified the same way. To check by hand:

```sh
uv run --locked python -m tools.posttrain_ops verify-final --run "$RUN" --pod-id "$POD_ID" \
  --report-sha256 <r2_final.report_sha256> --artifact-count <r2_final.artifact_count> --output "$CTRL/manual-verification"
```

## 15. Pod termination and token revocation (Mac)

The guards delete the Pod after `shutdown_ready`; the in-Pod timer deletes it at `SELF_DESTRUCT_DEADLINE` and after 900 s idle regardless. To terminate by hand, use the block below: it sends the DELETE only on a verified `shutdown_ready` for this Pod or after the owner's decision typed as `TERMINATE=yes`, and records `TEARDOWN` only once the Pod answers 404:

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

Once the Pod is gone, the **owner** revokes the Pod's R2 token in the Cloudflare dashboard, removes the `R2_POD_*` lines from `.env`, and checks under Storage that no network volume exists. The token must be revoked and out of `.env` before step 17's execution window opens.

## 16. If the run fails

- **Bootstrap or receipt fails:** read `bootstrap.log` or the receipt tool's message, terminate the Pod (step 15) and report. A new Pod is a new launch: ask the owner first.
- **The worker stops** (`generation_below_planning_speed`, `sft_throughput_below_limit`, `projected_finish_after_deadline`, `finalization_reserve`, the cost, wall or idle limit) **or fails:** after its drain the heartbeat shows `stopped` or `failed`. With `r2_final` the supervisor verifies and archives as in step 14; without it, it archives the worker's own report to `.../diagnostics/` after 60 s and approves teardown. Every artifact made before the stop is in R2. Report the measured rates; do not relaunch without the owner.
- **A label never reached R2** (`failed`, failure `r2_milestones_unverified`): `finalization.json` lists each unverified label's files and SHA-256s. The supervisor checks each in R2 with the Mac's key; the weights label (`sft-weights`: `checkpoint.pt` and `record.json`) it pulls over SSH with the pilot's verified pull inside the 420 s reserve and checks. Any other label it writes to `MILESTONES-NOT-SECURED.json` and never approves teardown, but **the Pod is still deleted soon**, whatever the supervisor decides: the Mac watchdog deletes it 600 s after the worker's last GPU work (the heartbeat's `last_gpu_activity_unix`), or up to its 900 s transfer window while the supervisor sees transfer progress, and the in-Pod timer's idle rule deletes it 900 s after that work. The worker hands over 480 s after its last GPU work, so the copy has about 2 to 7 minutes. Such a label is a JSON or JSON Lines file of a few MB at most, which copies in seconds: run the block below at once. It copies every listed file and checks each SHA-256. A `pod_progress.py` stamp holds only the in-Pod rule, not the watchdog, so it is no way to hold the Pod; if the window closes first, the Pod and the label are gone: report it. Once every file prints `OK`, end the Pod with `TERMINATE=yes` (step 15).

```sh
python3 - "$CTRL/supervisor/MILESTONES-NOT-SECURED.json" > "$CTRL/rescue-list.txt" <<'PY'
import json, re, sys

for row in json.load(open(sys.argv[1]))["files"]:
    for pod_path in row["pod_paths"]:
        name = pod_path.rsplit("/", 1)[1]
        digest = row["sha256"][name]
        if not re.fullmatch(r"exports/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", pod_path) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            sys.exit(f"refusing an unsafe entry: {pod_path}")
        print(row["label"], name, pod_path, digest)
PY
while read -r LABEL NAME POD_PATH SHA256; do
  podscp root@$POD_HOST:/workspace/posttrain-run/$POD_PATH "$CTRL/rescued-$LABEL-$NAME" < /dev/null
  printf '%s  %s\n' "$SHA256" "$CTRL/rescued-$LABEL-$NAME" | shasum -a 256 -c -
done < "$CTRL/rescue-list.txt"
```

- **The Pod disappears or a watchdog exits:** read `$CTRL/watchdog-receipt.json` and its `.prev-<time>` copies and confirm with the `status` command in step 15. The in-Pod timer's own receipt names `idle_limit` when the idle rule deleted the Pod.

## 17. After the run: execution on the Mac (Mac execution window)

Model-written code now runs on the Mac. Before the window opens, the **owner** empties Docker Desktop's file sharing (Settings, Resources, File sharing: remove every entry, including the defaults `/Users`, `/Volumes`, `/private` and `/tmp`; the built-in `/var/folders` share cannot be removed and is a known residual, since the sandbox never asks for a mount), clears the SSH agent (`ssh-add -D`; Docker Desktop forwards the login session's agent into its VM whatever the executor's environment holds), and confirms the Pod token is revoked and gone from `.env` (step 15). Then, in a terminal where `.env` is sourced only to fetch the artifacts, record the sandbox image's local ID once, fetch the run's artifacts pinned to the verified report, and check the window:

```sh
export ART=$SHARED/artifacts
export SANDBOX_IMAGE_ID=$(docker image inspect --format '{{.Id}}' python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe)
uv run --locked python -m tools.posttrain_ops fetch-artifacts --report "$CTRL/supervisor/final/report.json" --output "$ART" --receipt "$SHARED/artifacts-fetch.json"
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check.json"
```

The window check fails in this terminal while `.env` is sourced or `SSH_AUTH_SOCK` is set, as macOS sets it in every terminal (its environment check names the variables, never their values); every execution below runs through `mac_window_check exec`, which repeats the agent, image and file-sharing checks before each batch and runs the executor in an environment without any `.env` variable, secret or SSH agent. The agent check asks `ssh-add -l` of the shell's and the login session's agent sockets, each of which must hold no identity; the file-sharing check must fail to mount every one of the four removable default shares, and records whether `/var/folders` mounts without failing on it. If any check fails, nothing runs.

Execute the four dev completions files and the dev constant-answer baseline (about 52,000 calls with the screening, about 2.4 hours):

```sh
mkdir -p "$SHARED/results"
for name in eval-base-pool-dev-greedy eval-base-pool-dev-n8 eval-sft-pool-dev-greedy eval-sft-pool-dev-n8; do
  uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$ART/$name/completions.jsonl" --tasks "$POOL/dev.jsonl" --results "$SHARED/results/$name.jsonl" --tokenizer "$TOKENIZER" --workers 8
done
uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-baseline --tasks "$POOL/dev.jsonl" --output "$SHARED/results/pool-dev-baseline.json"
```

Each `eval-execute` derives the dev export again from `$POOL/dev.jsonl` and refuses completions generated from anything else; it exits with status 3 while an attempt is unresolved, and running the same line again continues it.

Seal the four test completions files: their SHA-256s go into the hash-chained seal ledger, and nothing executes until all four endpoints are frozen after run 2:

```sh
mkdir -p "$SHARED/seal"
for name in eval-base-pool-test-greedy eval-base-pool-test-n8 eval-sft-pool-test-greedy eval-sft-pool-test-n8; do
  uv run --locked scratchcoder eval-seal --completions "$ART/$name/completions.jsonl" --ledger "$SHARED/seal/ledger.jsonl"
done
```

Score the screening attempts. The header binds them to the candidates export, which `rl-screen` derives again from the full train split, and to the screening protocol's sampling (8 samples, temperature 1.0, top-p 1.0, 256 new tokens, the screening seed base); `--require-header` refuses a file without it. `--timeout` is the worst case for one batch of 256 attempts on 8 workers: 32 rounds of the longest attempt (the 27 s budget of a 100-case I/O task plus the channel's 15 s slack, 4 tries and 7 s of backoff), 32 x 175 = 5,600 s:

```sh
uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder rl-screen --tasks "$POOL/train.jsonl" --candidates "$SFT/rl_candidates.jsonl" --attempts "$ART/screen/attempts.jsonl" --output "$SHARED/screening.json" --timeout 5600 --workers 8 --require-header
```

The record's `outcome` is `kept` (the RL task list for run 2) or `floor` (fewer than 320 kept: report to the owner). Its per-attempt rewards and timings re-price run 2 at the gate.

Score the six public-benchmark files on GitHub Actions (the repository is public, so no minutes are billed). Each dispatch checks the local file against its R2 object and signs a GET link for that one object, expiring within an hour:

```sh
mkdir -p "$SHARED/benchmarks-results"
for name in eval-base-humaneval-greedy eval-base-humaneval-plus-greedy eval-base-mbpp-plus-greedy eval-sft-humaneval-greedy eval-sft-humaneval-plus-greedy eval-sft-mbpp-plus-greedy; do
  SUITE=$(printf '%s' "$name" | sed -e 's/^eval-base-//' -e 's/^eval-sft-//' -e 's/-greedy$//')
  uv run --locked python -m tools.benchmark_dispatch --completions "$ART/$name/completions.jsonl" --suite "$SUITE" --key "posttrain-runs/1/$POD_ID/$name/completions.jsonl" --repo "$PUBLIC_REPO" --output "$SHARED/benchmarks-results/$name"
done
```

After the rehearsal's last window (after run 2), the owner resets Docker Desktop's VM data.

## 18. Final billing verification (Mac)

Fetch every hourly bucket from the hour of `ORIGIN` to now, and record the total only once two reads agree (Runpod bills every five minutes and records lag: on 2026-10-26 a read 19 minutes after teardown undercounted by about 6%). Run the block, wait at least 30 minutes, run it again:

```sh
export BUCKETS=$(( ($(date +%s) - ORIGIN) / 3600 + 2 ))
uv run --locked python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --last-buckets "$BUCKETS" --output "$CTRL/billing-$(date +%s).json"
python3 - "$CTRL" "$POD_ID" <<'PY'
import glob, json, sys

control, pod_id = sys.argv[1:3]
reads = []
total = 0.0
for path in sorted(glob.glob(f"{control}/billing-*.json")):
    report = json.load(open(path))
    if report.get("pod_id") != pod_id:
        sys.exit(f"{path} was fetched for another Pod")
    records = report.get("records") or []
    total = sum(float(record.get("totalAmount") or 0) for record in records)
    provider = float((report.get("totals") or {}).get("totalAmount") or 0)
    if abs(total - provider) > 1e-6:
        sys.exit(f"{path}: the records sum to {total} but the provider totals say {provider}")
    reads.append((path, round(total, 4)))
print(json.dumps(reads))
if total >= 3.0:
    sys.exit(f"the Pod cost {total:.4f} USD, at or above its $3.00 ceiling")
if len(reads) < 2 or reads[-1][1] != reads[-2][1]:
    sys.exit("the last two reads differ (or there is one read): wait and fetch again")
print(f"billing settled at {reads[-1][1]} USD over two reads, under the $3.00 ceiling")
PY
```

Compare the settled total with the report's `cost`, and note the R2 storage with `uv run --locked python -m tools.r2_store status`.

## 19. Results

From the verified `report.json` and the archived summary: the probe's measured rates by batch size, ARC-Easy accuracy for A `stage1-final`, A `stage2-final`, B and the SFT weights, each quarter's SFT dev loss and the selected quarter, the speed check, every projection, the cost and timings, and `code_revision`. From the Mac: the dev results files, the seal ledger, the screening record and the benchmark artifacts. Record the measured numbers in the docs as measured results, distinct from the projections in [docs/posttrain-runs.md](docs/posttrain-runs.md); run 2 is re-priced from them at the owner's gate.
