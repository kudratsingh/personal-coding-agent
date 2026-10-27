# Post-training runbook: run 2 on one A40 (RL with the Mac in the loop, the more-SFT control, evaluation)

Operational sequence for post-training run 2 of the 135M-parameter S rehearsal model, on one Runpod A40 Pod, from run 1's selected SFT weights. Run 1's [runbook](POSTTRAIN_RUNBOOK.md) is the base: the Pod creation, the bootstrap, the receipt, the supervisor, the teardown and the billing reads are its steps with run 2's values, and this page says so where they are. The worker (`tools/posttrain_run.py`, run 2's stages in `tools/posttrain_run2.py`) runs five stages in order and uploads every artifact to R2 under `posttrain-runs/2/<pod id>/` as soon as it exists:

1. **rl:** 100 updates of 32 kept tasks x 8 samples at temperature 1.0 and top-p 1.0, at most min(256, 1,024 - prompt) new tokens, a fresh AdamW at 3e-6 after 5 warmup updates, no KL; dev checks at updates 0, 12, 25, 50, 75 and 100 on the fixed 200-task dev subset, each generated at batch 256. Rewards and dev verdicts come from the Mac through the reward bridge (step 11); each check's weights are uploaded at once as `rl-check-<update>`.
2. **more-sft:** the compute-matched control from the same SFT weights on the kept tasks' references, its GPU-busy budget RL's measured busy seconds, checks after 1, 2, 4 and 8 passes and at the budget's end (batch 256, as RL's), through the bridge; each check's weights uploaded as `more-sft-check-<step>`.
3. **arc-post:** ARC-Easy (base protocol) on the RL-selected and the control-selected checkpoints.
4. **eval-rl**, **eval-more-sft:** the seven completions files of each selected checkpoint under `chat-v1`, as run 1's `eval-sft`.

Then run 1's finalization, with the bridge directories uploaded as one label (`bridge`) first, so a late reply is kept. Any RL stop that is a result (completed, collapse, dev patience, leakage, non-finite) lets the control run.

**Same weights.** When an arm's selected checkpoint holds the SFT weights the run loaded, every tensor compared bit for bit (a file's SHA-256 differs between two saves of one model), the arm is not evaluated again: no `arc-post-<arm>` report and no `eval-<arm>` files. The control starts from RL's update-0 check, the SFT weights, so this is what happens when neither arm beats its start. `report.json`'s `selection.endpoints.<arm>` records the comparison and that run 1's SFT results stand for the arm, each pinned in the launch specification's `sft_results` (R2 key, size and SHA-256, fixed at the gate). Step 18 executes each distinct checkpoint once and maps the arm to SFT's results; the scorecard marks its rows as not an independent measurement.

**An arm's stop is not the run's.** A wall or cost stop ends the run. A stop of the Mac's side in either arm (`reward_wait`, a timeout on either side; `reward_host_failure`; `bridge_lost`) and a margin stop (`wait_margin`) end only that arm's training, after its latest weights reach R2: its endpoint is selected from the dev checks already scored, and the stages that need no Mac wait run under the projected-finish rule, the finalization reserve and every limit. After such a stop in RL the control is skipped (its dev checks need the Mac) and RL's endpoint is evaluated; after a stop of the Mac's side in the control the control is not evaluated and RL's is; after a margin stop in the control both are, each as the projected-finish rule admits it (when both do not fit at `arc-post`, before any of the control's evaluation runs, but RL's alone does, RL's; a refusal at a later stage stops the run on the projected finish). With no RL dev check scored there is nothing to select and the run finalizes at once. The run then ends `stopped` with the arm's reason (`report.json` `arm_stop`). While that evaluation runs the heartbeat reads `running`, never `awaiting_rewards`, so the watchdog's 600 s idle rule and the in-Pod 900 s rule apply as to any stage.

Commands marked **Mac** run on the owner's Mac inside the repository checkout; **Pod** commands run inside the rented Pod over SSH. One terminal drives the run, a second runs the supervisor (step 13), and a third, prepared in step 6 before the Pod exists, runs the reward bridge (step 11) in a terminal where `.env` has never been sourced. Nothing in steps 1 to 6 starts a Pod; step 5 stores inputs in R2 (cents a month). Step 7 creates the Pod and starts billing: do not execute it without the owner's explicit go for run 2 at the gate. Runpod offers no provider-enforced Pod lifetime: termination is layered best effort.

**Limits, all derived from one cost limit fixed at the gate** (`tools/posttrain_ops.py` `ceiling_limits`; `limits` prints them). The owner fixes the ceiling. The gate's estimate is the run 2 projection's total at run 1's measured rates and the gate's quoted price; step 4 reads it from the projection's own output, never from memory. The cost limit is the lower of the ceiling and 120% of the estimate, rounded down to the cent, and every other limit derives from it. At an estimate of $3.53 (the projection at the planning speeds) and the planned $5.80 ceiling the cost limit is **$4.23**: the worker stops at **$4.23**, the Mac watchdog at **$4.22**, the Pod wall is **25,380 s** and the in-Pod timer fires at **25,200 s**, 180 s inside it. Only an estimate of $4.84 or more lets the $5.80 ceiling bind. The launch specification writes the estimate and the cost limit down; the gate, the readiness receipt and the worker refuse a missing, non-finite or non-positive estimate and a written limit other than its derivation. The fixed 60,000 s of the pretraining rehearsal is never used: it stays only as the in-Pod timer's absolute cap. The watchdog's idle limit is 600 s; the in-Pod timer also deletes the Pod after **900 s** with no GPU work and no progress.

**The bridge's bounds, derived in one place** (`posttrain_ops.bridge_timeouts`; the specification writes them down and must equal the derivation; `limits --field` prints each):

| Bound | Value | Arithmetic |
| --- | ---: | --- |
| One program's longest try | 42 s | the channel's longest budget, 27 s (a 100-case I/O task), plus its 15 s host slack |
| One reward attempt, all tries | 175 s | 4 tries x 42 s + the 1, 2 and 4 s backoff waits |
| Mac `--reward-timeout` | **5,600 s** | ceil(256 attempts / 8 workers) = 32 rounds x 175 s |
| Mac `--dev-timeout` | **9,492 s** | ceil(1,800 programs / 8 workers) = 225 rounds x 42 s, + 42 s for a program still running |
| Bridge allowance | 1,397.5 s | one 2 s Mac poll, a listing, a read and a write each tried 5 times to its 30, 120 and 120 s timeout, 3 x (1 + 2 + 4 + 8) s of backoff, one 0.5 s Pod poll |
| The Mac's checks before it scores a request | 857 s | one execution-window pass at its worst, 730 s (11 `docker` calls to their 60 s bound, two `ssh-add -l` to 30 s, one `launchctl getenv` to 10 s), and the image check, 127 s (4 tries of 30 s and 1 + 2 + 4 s of backoff) |
| Pod reward timeout | **7,854.5 s** | 5,600 + 1,397.5 + 857 |
| Pod dev timeout | **11,746.5 s** | 9,492 + 1,397.5 + 857 |
| Watchdog idle extension while `awaiting_rewards` | 7,914.5 s (reward), 11,806.5 s (dev) | the Pod's timeout for that wait + 60 s of slack |
| Wait margin | 11,746.5 s, $1.96 | one full wait (the longer kind) at the quoted rate |
| Lost bridge | 600 s | a wait with no Mac lease renewal for this long; 300 s inside the in-Pod idle rule |

**Liveness while the Pod waits on the Mac.** While the bridge client waits, the worker heartbeats `awaiting_rewards` with the request it waits on (`awaiting`: request ID, kind, index, count, start and age). The Mac watchdog, under run 2's policy, then lets the GPU idle for that kind's extension and never more, so a worker that hangs while claiming to wait is still deleted. The Mac's bridge server renews its lease, `server.lease` in the Pod's bridge root and dated by the Pod's own clock, with every poll and every few seconds while it reads, scores or answers a request; the worker stamps the heartbeat's `progress_unix` from a lease renewed within its 60 s, which the in-Pod idle rule counts as progress, and a stale lease counts as nothing. A dead Mac therefore lets the Pod go: the worker gives the wait up 600 s after the last renewal (`bridge_lost`), uploads its weights and finalizes, and if it could not, the in-Pod timer deletes the Pod 900 s after that renewal. `should_stop` is not read during a wait, so before every update and every request the worker stops (`wait_margin`) unless one full wait fits before the cost stop, the wall and the finalization reserve; unlike the cost and wall stops it finalizes in full.

**The time rules.** The readiness receipt (step 10) and the worker at launch (step 12) apply the launch rule: the lifetime left must hold every stage at the planning rates (generation at 5,000 tok/s at batch 64 and 10,000 at batch 256, every attempt priced at its token cap, training at 56,765 slots/s, the Mac at 48 s per reward request and 300 s per dev check) plus the 900 s finalization reserve: 20,288 s, so setup may take up to about 4,912 s (1.4 hours) of the 25,200 s at a $4.23 limit. Then the projected-finish rule before every stage, at the rates RL has measured once it has run; a stage still running 900 s before the end of the lifetime stops there and finalizes in full. The projection at the planning speeds (step 4) is about 5.9 hours and $3.53.

**Hidden tests never leave the Mac.** The Pod receives the withheld RL task file and the withheld dev subset (every hidden test replaced by a fixed placeholder), the facts file (each task's parseability and best constant answer's reward, computed on the Mac; the answer itself is left out) and run 1's Pod exports. `scratchcoder rl-pod-export` writes the three run-2 files on the Mac from the full train and dev splits and the screening record; the binding, the Pod's verification and the worker refuse any file in which a task carries its hidden tests, and the worker checks the facts file against its pinned SHA-256. The sealed test split is not read again for run 2: its Pod export is run 1's.

**R2 safeguards.** As run 1: the Pod's own bucket-scoped token for this run only, over SSH stdin, revoked at teardown; the worker refuses any R2 write or delete outside `posttrain-runs/2/<pod id>/` before a request is sent.

## 1. Prelaunch checks (Mac, free)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check
uv run --locked python -m tools.posttrain_run smoke --output runs/posttrain-smoke
uv run --locked python -m tools.posttrain_run smoke --run 2 --output runs/posttrain-smoke-2
uv run --locked python -m tools.pilot_preflight --output runs/posttrain-preflight-cpu --device cpu
uv run --locked pytest tests/test_posttrain_run2.py tests/test_posttrain_run2_guards.py tests/test_posttrain_run2_integration.py tests/test_posttrain_run2_readiness.py tests/test_posttrain_run2_feasibility.py tests/test_posttrain_run2_runbook.py tests/test_rl_pod_export.py tests/test_rl_bridge.py tests/test_rl_bridge_ssh.py tests/test_rl_bridge_end_to_end.py tests/test_runpod_watchdog.py tests/test_pod_self_destruct.py -q
SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_posttrain_run2_sandbox.py tests/test_rl_bridge_end_to_end.py -q
```

Expected: `make check` green; both smokes print `"status": "completed"` with every stage `completed` and `"objects_outside_prefix": []` (run 2's also `rl` and `more_sft` results and the selected labels); every listed test passes, the sandbox ones too, with no `scratchcoder-*` container left (`docker ps -a --filter name=scratchcoder` prints nothing). The runbook test parses every tool command here with the real parsers.

## 2. Credentials and environment (Mac, free)

`.env` holds what run 1's step 2 lists, with a new Pod token for this run only. Then:

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi
done
gh auth status
caffeinate -dimsu -w $$ &
export RUN=2
export CTRL_ROOT=runs/posttrain-controller
export KNOWN_HOSTS=~/.ssh/posttrain_2_known_hosts
export SPEC_TEMPLATE=configs/rehearsal-a40-s-post-2.json
export SHARED=runs/posttrain-shared-2 && mkdir -p "$SHARED"
export SPEC=$SHARED/rehearsal-a40-s-post-2-gate.json
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts
export PUBLIC_REPO=kudratsingh/personal-coding-agent
export POOL=runs/task-pool-20261025/pool
export TOKENIZER=runs/tokenizer-freeze-65536-20261020/tokenizer.json
export RUN1_SHARED=runs/posttrain-shared
export RUN1_CTRL=$CTRL_ROOT/1/<run 1 pod id>
export RUN1_REPORT=$RUN1_CTRL/supervisor/final/report.json
export RL_EXPORT=$SHARED/rl-pod-export
export BINDING=$SHARED/binding.json
export BRIDGE_STATE=$SHARED/bridge-state
export LEDGER=$RUN1_SHARED/seal/ledger.jsonl
```

The loop prints names, never values. `RUN1_REPORT` is run 1's verified report, which its supervisor archived; `RUN1_SHARED` holds run 1's exports, artifacts, screening record, results and seal ledger (run 1's steps 4 and 17). Keep this terminal open on AC power for the whole run.

## 3. Run 1's records (Mac, free)

Run 1's step 17 fetched its artifacts pinned to the verified report, scored the screening attempts and executed the dev completions. Check that the files run 2 reads are there and that the screening kept enough tasks:

```sh
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["outcome"], r["counts"]["kept"])' "$RUN1_SHARED/screening.json"
shasum -a 256 "$RUN1_SHARED/artifacts/sft-weights/checkpoint.pt" "$RUN1_SHARED/exports/test-export.jsonl" "$RUN1_SHARED/exports/dev-export.jsonl"
```

The first line must print `kept` and a count of at least 320; `floor` goes to the owner, and run 2 does not start.

## 4. The gate (Mac, the owner decides; free)

**Re-price run 2 from run 1's measured rates.** The projection prints the proposal's table at the planning speeds, with the Mac twice as slow, and at run 1's rates: generation from the probe, slowed by the evaluation files' observed rate when lower; SFT's slot rate; ARC's seconds; the Mac's time per reward request and per dev check from the screening's and the SFT dev results' per-attempt scoring seconds. Price it against the proposed ceiling:

```sh
uv run --locked python -m tools.pilot_feasibility posttrain --specification "$SPEC_TEMPLATE" --ceiling-usd 5.80 --quoted-gpu-hourly 0.59 \
  --run1-report "$RUN1_REPORT" --screening "$RUN1_SHARED/screening.json" \
  --dev-results "$RUN1_SHARED/results/eval-sft-pool-dev-greedy.jsonl" --dev-results "$RUN1_SHARED/results/eval-sft-pool-dev-n8.jsonl" \
  --output "$SHARED/feasibility.json"
```

Replace `0.59` with the console's price; above $0.59, stop and ask the owner (the gate step refuses such a projection). The projection prints **the gate's estimate**, its total at run 1's measured rates, the stop 120% of it gives and what that stop leaves for setup after the launch rule's need at the planning rates (`the gate refuses it` when less than the 900 s setup); the gate step reads the estimate from `$SHARED/feasibility.json`. A faster run 1 makes a margin stop in the control likely, before its budget is spent ([Can a margin stop happen?](docs/posttrain-runs.md#an-arms-stop)): report that to the owner with the estimate. The owner fixes the ceiling (about $5.80, about 1.6 times the estimate) and gives the go or not. Record the ceiling:

```sh
export CEILING=<the owner's ceiling in dollars>
```

**The Pod's RL files.** Write the withheld RL task file, the withheld dev subset and the facts file from the full train and dev splits and the screening record's kept list. It never opens the test split or the holdouts:

```sh
uv run --locked scratchcoder rl-pod-export --train-tasks "$POOL/train.jsonl" --kept "$RUN1_SHARED/screening.json" --dev-tasks "$POOL/dev.jsonl" --output "$RL_EXPORT"
```

It prints each file's name, SHA-256 and size; the same inputs always give the same bytes.

**The launch specification.** Fill the template's gate values from the gate's own files: the ceiling, the projection's estimate and the cost limit they give, run 1's selected SFT weights (key, size and SHA-256 of the `sft-weights` label in run 1's verified report), run 1's SFT evaluation artifacts (`sft_results`: the key, size and SHA-256 of its `arc-sft-sft` report and its seven `eval-sft-<file>` completions files, which an arm with the SFT weights reuses), the three files just written (hashed from their bytes, cross-checked with their record) and run 1's pool test export:

```sh
uv run --locked python -m tools.posttrain_approved run2-gate --ceiling-usd "$CEILING" --feasibility "$SHARED/feasibility.json" --run1-report "$RUN1_REPORT" --rl-pod-export "$RL_EXPORT" --pool-test-export "$RUN1_SHARED/exports/test-export.jsonl" --output "$SPEC"
uv run --locked python -m tools.posttrain_binding check-exports --specification "$SPEC" --input pool-dev="$RUN1_SHARED/exports/dev-export.jsonl" --input pool-test="$RUN1_SHARED/exports/test-export.jsonl"
```

The first prints the launch specification's SHA-256, the ceiling, the estimate, the cost limit and what the derived limits leave for setup (`setup_allowance_seconds`: the lifetime less the 20,288 s the launch rule needs at the planning rates); the owner reviews the file. It refuses a projection of another template, of another run 1 report or above the $0.59 quote, and limits that leave less than the planned 900 s for setup (at a $5.80 ceiling, an estimate below about $2.98): the readiness receipt would refuse such a launch only after the Pod had billed its setup, so it goes back to the owner now. `check-exports` must print `{"exports": {"pool-dev": 2000, "pool-test": 2000}, "hidden_tests": 0}`. Then read the limits and the bridge's bounds off it:

```sh
export COST_LIMIT=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field limit_usd)
export WALL=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field wall_seconds)
export WATCHDOG_COST=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field watchdog_cost_stop_usd)
export TIMER=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field self_destruct_seconds)
export REWARD_EXTENSION=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field awaiting_reward_extension_seconds)
export DEV_EXTENSION=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field awaiting_dev_extension_seconds)
export MAC_REWARD_TIMEOUT=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field mac_reward_timeout_seconds)
export MAC_DEV_TIMEOUT=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field mac_dev_timeout_seconds)
export POD_REWARD_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["rl-tasks"]["sha256"])' "$SPEC")
export POD_DEV_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["rl-dev"]["sha256"])' "$SPEC")
echo "run $RUN: cost limit $COST_LIMIT USD, wall $WALL s, timer $TIMER s, watchdog cost stop $WATCHDOG_COST USD, Mac bounds $MAC_REWARD_TIMEOUT / $MAC_DEV_TIMEOUT s"
```

The last line prints the limits derived from the cost limit the gate wrote. At a $5.80 ceiling and a $3.53 estimate it must print `cost limit 4.23 USD, wall 25380 s, timer 25200 s, watchdog cost stop 4.22 USD, Mac bounds 5600.0 / 9492.0 s`; at another estimate the numbers follow from `ceiling_limits`.

## 5. Binding and staging (Mac, at the go; R2 storage only)

Bind every input by R2 key, size and SHA-256. The tokenizer and run 1's SFT weights are already in R2 at the keys the launch specification pins (the Mac's copy of the weights is checked against the pin); every other input is staged under `posttrain-inputs/<sha256>/<file>`:

```sh
uv run --locked python -m tools.posttrain_binding build --specification "$SPEC" --run 2 --output "$BINDING" \
  --input sft-weights="$RUN1_SHARED/artifacts/sft-weights/checkpoint.pt" \
  --input rl-tasks="$RL_EXPORT/rl-tasks.jsonl" --input rl-dev="$RL_EXPORT/rl-dev.jsonl" --input rl-facts="$RL_EXPORT/rl-facts.json" \
  --input pool-dev="$RUN1_SHARED/exports/dev-export.jsonl" --input pool-test="$RUN1_SHARED/exports/test-export.jsonl" \
  --input arc-easy-test="$RUN1_SHARED/exports/arc-easy-test.jsonl" \
  --input humaneval="$RUN1_SHARED/benchmarks/HumanEval.jsonl.gz" \
  --input humaneval-plus="$RUN1_SHARED/benchmarks/HumanEvalPlus-OriginFmt.jsonl.gz" \
  --input mbpp-plus="$RUN1_SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz"
```

The owner reviews the binding and the `sha256` it printed. Paste that value, then stage:

```sh
export BINDING_SHA256=<reviewed binding sha256>
printf '%s  %s\n' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -
uv run --locked python -m tools.posttrain_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --receipt "$SHARED/stage-receipt.json"
```

## 6. The Mac's copy of the SFT weights and the bridge's terminal (Mac, before the Pod; free)

The readiness receipt requires the Mac's own pinned copy of run 2's parent:

```sh
uv run --locked python -m tools.posttrain_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --root runs/posttrain-parents-2 --only sft-weights --receipt "$SHARED/mac-copy.json" --key-prefix R2_
```

**Prepare the reward bridge now, before any billing.** The bridge (step 11) runs model-written code in Docker Desktop on the Mac while the Pod is up, so the run-1 execution-window rules hold from here until step 16 ends the Pod, with one difference, an accepted rule: the Pod's R2 token cannot be revoked yet (the worker uploads with it), so it stays in `.env` during the bridge window under the child-environment rule. The bridge runs only through `mac_window_check exec`, in an environment without any `.env` variable, and its own repeated checks (step 11) refuse to score anything once a `.env` name, a secret or an SSH agent variable appears in its environment; no container can reach `.env` itself. The **owner** now empties Docker Desktop's file sharing (Settings, Resources, File sharing: remove every entry, including the defaults `/Users`, `/Volumes`, `/private` and `/tmp`; the built-in `/var/folders` share cannot be removed and is a known residual, since the sandbox never asks for a mount) and clears the SSH agent (`ssh-add -D`). Nothing in steps 7 to 16 needs either: every SSH call uses `$POD_SSH_KEY` and the pinned known-hosts file only, never the agent. Doing it here keeps it off the clock: once the Pod exists, steps 11 to 13 must run back to back (step 11 says why).

Open the third terminal and **never source `.env` in it**. Export only what the bridge needs (paste the values from the first terminal; none is a secret) and check the window:

```sh
export RUN=2
export SHARED=runs/posttrain-shared-2
export RUN1_SHARED=runs/posttrain-shared
export POOL=runs/task-pool-20261025/pool
export BRIDGE_STATE=$SHARED/bridge-state
export KNOWN_HOSTS=~/.ssh/posttrain_2_known_hosts
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export POD_REWARD_SHA256=<from step 4>
export POD_DEV_SHA256=<from step 4>
export MAC_REWARD_TIMEOUT=<from step 4>
export MAC_DEV_TIMEOUT=<from step 4>
export SANDBOX_IMAGE_ID=$(docker image inspect --format '{{.Id}}' python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe)
unset SSH_AUTH_SOCK SSH_AGENT_PID
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check-bridge.json"
```

The window check must pass: no `.env` variable or secret in this terminal, no identity in the shell's or the login session's SSH agent, the pinned image, and none of the four removable default shares mountable (it records whether `/var/folders` mounts without failing on it). macOS sets `SSH_AUTH_SOCK` in every terminal, so the block unsets it in this one; the agent check still asks the login session's agent. If it fails, fix it now; the Pod is not created until it passes. Leave this terminal open: step 11 adds the Pod's address and starts the bridge in it.

## 7. Pod creation, identity and the watchdog (Mac, BILLABLE)

Only after the owner's explicit go. Run 1's step 7 with run 2's create body (`configs/runpod-pod-a40-s-posttrain-2.json`, named `posttrain-a40-s-2`), its `SPEC` and its timer deadline:

```sh
OPEN_PODS=$(find "$CTRL_ROOT/$RUN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r DIR; do grep -qs '^export TEARDOWN=' "$DIR/run.env" || printf '%s ' "$DIR"; done)
if [ -n "${POD_ID:-}" ]; then
  echo "run $RUN: NOT creating a Pod: this terminal already names Pod $POD_ID, which may still bill. Confirm it gone with step 16 first."
elif [ -n "$OPEN_PODS" ]; then
  echo "run $RUN: NOT creating a Pod: no TEARDOWN is recorded in ${OPEN_PODS% }. End each such Pod with step 16 and ask the owner before creating another."
else
  export ORIGIN=$(date +%s)
  export LAUNCH=$CTRL_ROOT/$RUN/launch-$ORIGIN && mkdir -p "$LAUNCH"
  cp configs/runpod-pod-a40-s-posttrain-2.json "$LAUNCH/create-body.json"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS --fail-with-body -X POST https://api.runpod.io/v2/pods \
    -H "Content-Type: application/json" --data @"$LAUNCH/create-body.json" -o "$LAUNCH/create-response.json" -w '%{http_code}\n' > "$LAUNCH/create-http-code"
  export POD_ID=$(python3 -c 'import json,re,sys; v=json.load(open(sys.argv[1])); i=v.get("id") if isinstance(v, dict) else None; ok=isinstance(i, str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,127}", i); print(i) if ok else sys.exit("no valid Pod id in " + sys.argv[1])' "$LAUNCH/create-response.json")
  export SELF_DESTRUCT_DEADLINE=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --origin "$ORIGIN" --field self_destruct_deadline_unix)
  export CTRL=$CTRL_ROOT/$RUN/$POD_ID
  if [ -z "$POD_ID" ]; then
    echo "run $RUN: the create request ended with HTTP $(cat "$LAUNCH/create-http-code") and names no valid Pod id. A Pod MAY STILL EXIST: ask the owner to check the Runpod console for posttrain-a40-s-2 and decide before any retry."
    export CTRL=$LAUNCH
  elif mkdir "$CTRL"; then
    mv "$LAUNCH/create-body.json" "$LAUNCH/create-response.json" "$LAUNCH/create-http-code" "$CTRL/" && rmdir "$LAUNCH"
    printf 'export ORIGIN=%s POD_ID=%s SELF_DESTRUCT_DEADLINE=%s\n' "$ORIGIN" "$POD_ID" "$SELF_DESTRUCT_DEADLINE" > "$CTRL/run.env"
    echo "run $RUN: Pod $POD_ID, controller directory $CTRL, timer deadline $SELF_DESTRUCT_DEADLINE"
  else
    echo "run $RUN: $CTRL already exists and is NOT this Pod's. Terminate Pod $POD_ID with step 16 (TERMINATE=yes)."
    export CTRL=$LAUNCH
  fi
fi
```

Capture the identity and write run 2's watchdog policy, which carries the two derived idle extensions, then write the helpers and start the watchdog. The watchdog helper is run 1's with the extensions; with `--policy` the watchdog refuses to start unless its flags give exactly the policy file's limits:

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
    --max-transfer-seconds 900 --awaiting-reward-extension-seconds "$REWARD_EXTENSION" \
    --awaiting-dev-extension-seconds "$DEV_EXTENSION" \
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
      echo "STOP run $RUN: POD GONE: the watchdog of Pod $POD_ID ended for good with $GONE. It is NOT restarted. Follow step 17."
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

**A reopened terminal** repeats step 2 and the exports of step 4, then sources this Pod's files, and never re-runs a block that creates a Pod or writes `run.env`, the policy or the known-hosts file: `export CTRL=$CTRL_ROOT/$RUN/<pod id>; export BINDING_SHA256=<reviewed binding sha256>; source "$CTRL/run.env"; source "$CTRL/helpers.sh"`.

Wait for the Pod to be RUNNING with a direct SSH host and port and pin its host key: run 1's step 7 wait block, unchanged (it reads `$CTRL`, `$RUN` and `$KNOWN_HOSTS`):

```sh
GIVE_UP=$(( $(date +%s) + 1200 ))
while :; do
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); d=c.get("ssh_direct") or {}; ok=c.get("status") == "RUNNING" and bool(d.get("host")) and bool(d.get("port")); print(c.get("status"), "with direct SSH" if ok else "without direct SSH"); sys.exit(0 if ok else 1)' "$CTRL/pod-connection.json" && break
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: Pod $POD_ID is not RUNNING with direct SSH after 20 minutes; terminate it (step 16, TERMINATE=yes)"; break; fi
  sleep 20
  uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json" > /dev/null
  keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 17"; break; fi
done
export POD_HOST=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["host"])' "$CTRL/pod-connection.json")
export POD_PORT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ssh_direct"]["port"])' "$CTRL/pod-connection.json")
printf 'export POD_HOST=%s POD_PORT=%s\n' "$POD_HOST" "$POD_PORT" >> "$CTRL/run.env"
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$KNOWN_HOSTS" 2>/dev/null
ssh-keygen -lf "$KNOWN_HOSTS"
watchdog
```

## 8. Code, specification, binding and the Pod's R2 key (Mac to Pod, billable)

Run 1's step 8 writer of `revision.json` and `repo.tar.gz` (refusing a dirty tree), then ship them with the binding and, unlike run 1, the launch specification, which is not in the repository:

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
podscp "$SPEC" root@$POD_HOST:/workspace/posttrain/specification.json
uv run --locked python -m tools.rehearsal_ops pod-env | podssh_stdin 'umask 077 && cat > /workspace/posttrain/r2-pod.env && chmod 600 /workspace/posttrain/r2-pod.env'
podssh 'stat -c "%a %U %s" /workspace/posttrain/r2-pod.env; sha256sum /workspace/posttrain/specification.json'
shasum -a 256 "$SPEC"
watchdog
```

The `stat` line should print `600 root` and a nonzero size, and both SHA-256 lines must agree.

## 9. Bootstrap (Pod, billable)

Run 1's step 9 with `--run 2` and run 2's timer lifetime (`$TIMER`, 25,200 s at a $4.23 limit):

```sh
if podssh 'test -e /workspace/posttrain/bootstrap.log'; then
  echo "run $RUN: a bootstrap was launched on this Pod before; the wait below follows that one."
else
  podssh "test ! -e /workspace/posttrain/bootstrap.log || exit 1; cd /workspace/repo && tar xzf /workspace/repo.tar.gz || exit 1; setsid nohup bash -c 'bash tools/pilot_pod_bootstrap.sh --mode posttrain --run $RUN --deadline-unix $SELF_DESTRUCT_DEADLINE --lifetime-seconds $TIMER --idle-seconds 900 --workspace /workspace/posttrain --repo /workspace/repo --revision /workspace/revision.json --archive /workspace/repo.tar.gz --r2-env /workspace/posttrain/r2-pod.env --binding /workspace/posttrain/binding.json --binding-sha256 $BINDING_SHA256 --data-root /workspace/posttrain-inputs --heartbeat /workspace/posttrain-run/heartbeat.json; echo BOOTSTRAP_EXIT=\$?' > /workspace/posttrain/bootstrap.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 2700 ))
until podssh 'grep -q ^BOOTSTRAP_EXIT= /workspace/posttrain/bootstrap.log'; do
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: bootstrap exceeded 45 minutes; terminate the Pod (step 16, TERMINATE=yes)"; break; fi
  sleep 30; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 17"; break; fi
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

Expected as run 1's step 9: `BOOTSTRAP_EXIT=0`, an armed timer at `SELF_DESTRUCT_DEADLINE` with `"idle_seconds": 900.0` over both activity files, the revision, a passed round trip under `posttrain-runs/2/<pod id>/` of at least about 3.06 MB/s, a completed preflight and a leak count of `0`.

## 10. Readiness receipt (Mac, billable)

Run 1's step 10 with run 2's specification and the Mac's copy of the SFT weights. The receipt requires the launch specification to be the approved run 2 with every gate value filled, the policy's idle extensions to be exactly the derived bounds, and refuses anything the worker would refuse:

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

## 11. The reward bridge (Mac, the third terminal; a Mac execution window while the Pod is up)

Start the bridge before the worker, so dev check 0, RL's first request, is answered at once. Until RL creates the bridge root on the Pod the bridge's listings find no root and it polls with `pod_ready False`; that is an idle poll, not a failure, and it takes no lease. **Steps 11, 12 and 13 run back to back:** the watchdog last re-based its idle clock at the end of step 10, and it reads the worker's heartbeat only through the supervisor's mirror (step 13), so until the supervisor runs it deletes the Pod 600 s after its last restart.

In the third terminal of step 6, add the Pod's address from the first terminal (`cat "$CTRL/run.env"` there prints it), then start the bridge, through the window check so it runs in an environment without any `.env` variable, secret or SSH agent:

```sh
export POD_HOST=<the Pod's host>
export POD_PORT=<the Pod's port>
uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder rl-bridge \
  --host root@$POD_HOST --port $POD_PORT --identity-file "$POD_SSH_KEY" --known-hosts "$KNOWN_HOSTS" \
  --remote-root /workspace/posttrain-bridge --train-tasks "$POOL/train.jsonl" --kept "$RUN1_SHARED/screening.json" \
  --dev-tasks "$POOL/dev.jsonl" --pod-reward-sha256 "$POD_REWARD_SHA256" --pod-dev-sha256 "$POD_DEV_SHA256" \
  --state "$BRIDGE_STATE" --reward-timeout "$MAC_REWARD_TIMEOUT" --dev-timeout "$MAC_DEV_TIMEOUT" \
  --image-id "$SANDBOX_IMAGE_ID" --window-check-interval 900 --dotenv .env
```

- **The timeouts** are the Mac's bounds, the Pod's less the bridge allowance and the Mac's checks: 5,600 s for a reward request and 9,492 s for a dev check. A request that runs past its bound gets a `timeout` reply the Pod can still read.
- **The window, repeated.** The bridge runs the window checks itself, on its own process: at its start, and again before scoring a request once its last pass began `--window-check-interval` (900 s) ago or more. Each pass checks its environment (no `.env` name, secret or SSH agent variable), every SSH agent the Mac names (this terminal's, whose path `exec` hands down, none here since step 6 unset it, and the login session's, asked afresh), the pinned image's ID and the four removable default shares (each refused with Docker's `mounts denied` answer; `/var/folders`, the known residual, must mount, the pass's positive control), and leaves a receipt in `$BRIDGE_STATE/window-checks/`, numbered after any receipt already there. The interval is at most 900 s; the bridge refuses a longer one. A pass takes seconds; at its worst, 730 s, it and the image check are inside the Pod's timeouts (the 857 s row above), and the bridge keeps its lease renewed meanwhile. A failed or errored pass stops the bridge with **exit 4** before anything more is scored, its reason in `progress.json` and the receipt; the Pod then sees a lost bridge 600 s later and evaluates what it can (above). Fix the window, then start the same command again (below) if the Pod still waits.
- **`.env`:** `--dotenv .env` names the file whose variable names the bridge refuses to run with; it refuses to start when no `.env` can be found, so give `--no-dotenv` only if the Mac has none. Exit **2** means it refused to start (a secret in its environment, a missing `.env`, a sealed split or a bad record): read its message, fix the terminal, start it again.
- **Exit 3** means another bridge server holds this Pod's lease. Stop that one (only one server may score a Pod: two would score each request twice), or wait for its lease to lapse 60 s after its last renewal.
- **A dead bridge** (the Mac slept, the process crashed, an SSH outage outlasted its retries, a failed window pass): start the same command again with the same `--state` at once. A restart over the same state is the same server and takes its lease back at once; every reply it sent is stored there, so nothing is scored twice. Its start pass takes seconds. It must be back within 600 s of its last lease renewal, or the worker gives the wait up as `bridge_lost`, ends that arm's training and evaluates what needs no Mac; the in-Pod timer would delete the Pod 900 s after that renewal if the worker could not.

Leave the bridge running until step 16 ends the Pod; then stop it with Ctrl-C (exit 0). Go back to the first terminal for step 12 at once.

## 12. Worker launch (Pod, billable)

Re-base the watchdog, launch the worker and wait for its first heartbeat:

```sh
watchdog
if podssh 'test -e /workspace/posttrain/run.log'; then
  echo "run $RUN: a worker was launched on this Pod before; the wait below follows that one."
else
  podssh "test ! -e /workspace/posttrain/run.log || exit 1; setsid nohup bash -c 'set -a; . /workspace/posttrain/r2-pod.env; set +a; cd /workspace/repo && python3 -m tools.posttrain_run run \
    --specification /workspace/posttrain/specification.json --binding /workspace/posttrain/binding.json --binding-sha256 $BINDING_SHA256 \
    --root /workspace/posttrain-inputs --readiness /workspace/posttrain/readiness-receipt.json \
    --preflight /workspace/posttrain/preflight-a40/report.json --revision /workspace/posttrain/revision.json \
    --output /workspace/posttrain-run; echo WORKER_EXIT=\$?' > /workspace/posttrain/run.log 2>&1 < /dev/null &"
fi
GIVE_UP=$(( $(date +%s) + 1200 ))
until podssh 'test -s /workspace/posttrain-run/heartbeat.json'; do
  if podssh 'grep -q ^WORKER_EXIT= /workspace/posttrain/run.log'; then echo "run $RUN: the worker exited before its first heartbeat:"; podssh 'tail -n 40 /workspace/posttrain/run.log'; break; fi
  if [ "$(date +%s)" -ge "$GIVE_UP" ]; then echo "run $RUN: no heartbeat after 20 minutes:"; podssh 'tail -n 40 /workspace/posttrain/run.log'; break; fi
  sleep 10; keepalive; if [ $? -eq 3 ]; then echo "run $RUN: stop waiting: the Pod is gone; follow step 17"; break; fi
done
podssh 'test -s /workspace/posttrain-run/heartbeat.json' && watchdog
```

Start the supervisor (step 13) as soon as this block has found the heartbeat: the watchdog's idle clock runs from the restart that ends it. Prepare the second window (step 2 and the reopened-terminal line of step 7) while the block waits. RL starts at once, and its first request, dev check 0, goes to the bridge started in step 11; the worker gives a wait up as `bridge_lost` 600 s after it starts with no Mac lease renewal.

## 13. Supervisor (Mac, billable; the second window)

Start the supervisor as soon as the heartbeat exists. Run 1's step 12 with `--run "$RUN"` and run 2's policy:

```sh
uv run --locked python -m tools.pilot_supervisor --mode posttrain --run "$RUN" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --identity-file "$POD_SSH_KEY" --known-hosts "$KNOWN_HOSTS" --output "$CTRL/supervisor" \
  --host root@$POD_HOST --port $POD_PORT --remote-root /workspace/posttrain-run \
  --repository "$PRIVATE_REPO" --poll-seconds 20 2>&1 | tee -a "$CTRL/supervisor.log"
```

Then, in the third terminal, confirm that the bridge sees the Pod (the worker creates the bridge root at RL's start) and that the first request was answered:

```sh
python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print("pod_ready", p["pod_ready"], "state", p["state"], "answered", p["answered"], "last", p["last_answered"])' "$BRIDGE_STATE/progress.json"
```

It must print `pod_ready True` within a minute of RL's start.

## 14. Monitoring (Mac, read-only)

```sh
python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); print(h["status"], h["stage"], h["phase"], h["step"], h.get("awaiting"), h.get("progress_unix"), h.get("r2_final"))' "$CTRL/supervisor/heartbeat.json"
podssh 'tail -n 2 /workspace/posttrain/run.log; python3 -c "import json; r=json.load(open(\"/workspace/posttrain-run/report.json\")); print(r.get(\"status\"), [(s[\"name\"], s[\"status\"]) for s in r[\"stages\"]], (r.get(\"rl\") or {}).get(\"checks\"))"'
tail -n 3 "$CTRL/watchdog.log"; cat "$CTRL/watchdog-receipt.json"
```

Expect `running` and `awaiting_rewards` alternating through RL and the control (the `awaiting` object names each request's kind and index, and `progress_unix` follows the Mac's lease), `running` through ARC and the evaluation, and `uploading` only at the very end. In the third terminal `$BRIDGE_STATE/progress.json`'s `window_check` names the last pass and its time. Each dev check's weights appear in R2 as `rl-check-<update>/checkpoint.pt` or `more-sft-check-<step>/checkpoint.pt` as soon as the check ends.

## 15. Completion and verification (Mac, automatic)

As run 1's step 14: the worker drains its uploads, checks every label by HEAD, uploads `report.json` and writes `awaiting_transfer` with `r2_final`; the supervisor verifies everything in R2 with the Mac's own key, archives the report and a summary privately with read-back, and only then marks `shutdown_ready`; the watchdog deletes the Pod and confirms a 404. A stopped run (`reward_wait`, `reward_host_failure`, `bridge_lost`, `wait_margin`, the projected finish, the reserve) that verified every label publishes `r2_final` too. To check by hand:

```sh
uv run --locked python -m tools.posttrain_ops verify-final --run "$RUN" --pod-id "$POD_ID" \
  --report-sha256 <r2_final.report_sha256> --artifact-count <r2_final.artifact_count> --output "$CTRL/manual-verification"
```

## 16. Pod termination and token revocation (Mac)

Run 1's step 15 block, unchanged (it reads `$CTRL`, `$RUN` and `$POD_ID`):

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

Once the Pod is gone, stop the bridge (Ctrl-C in the third terminal). The **owner** revokes the Pod's R2 token in the Cloudflare dashboard, removes the `R2_POD_*` lines from `.env`, and checks under Storage that no network volume exists.

## 17. If the run fails

- **Bootstrap or receipt fails:** as run 1: read the log or the message, terminate the Pod (step 16), report; a new Pod is a new launch, after the owner.
- **An arm stopped early** (`reward_wait`: a timeout on either side; `reward_host_failure`: the Mac's sandbox could not resolve a host failure; `bridge_lost`: no Mac lease for 600 s; `wait_margin`: no time or money left for one full wait): the latest weights are in R2 (`rl-stop-weights`, `more-sft-stop-weights`, or the last check's label), the evaluation that needs no Mac ran (top of this page), and the heartbeat shows `stopped` with `r2_final`; the supervisor verifies and archives as in step 15. Step 18 then executes what was evaluated. Another stop (the projected finish, the reserve, the cost, wall or idle limit) ends the run as it stands. Report the stop, the measured rates, `arm_stop` and the bridge's `progress.json`, `events.jsonl` and window receipts; do not relaunch without the owner.
- **The Mac refuses a request** (`BridgeRefused`: a wrong task-file SHA-256, a task outside the kept list or the dev subset) or the worker fails otherwise: the run ends `failed`, its latest weights uploaded as `rl-latest-weights` or `more-sft-latest-weights` first; the configuration needs a person.
- **A label never reached R2:** run 1's step 16 rescue rules and block apply, with this run's `$CTRL`; the weights labels are pulled over SSH by the supervisor.

## 18. After the run: execution on the Mac (Mac execution window)

With the Pod gone and its token revoked, fetch the run's artifacts pinned to the verified report and check the window, as run 1's step 17:

```sh
export ART=$SHARED/artifacts
export RUN2_REPORT=$CTRL/supervisor/final/report.json
uv run --locked python -m tools.posttrain_ops fetch-artifacts --report "$RUN2_REPORT" --output "$ART" --receipt "$SHARED/artifacts-fetch.json"
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check.json"
```

**Each endpoint's source, recorded once.** An arm is `pod` (evaluated on run 2's Pod), `reuses-run-1-sft` (its selected weights were the SFT weights, so run 1's SFT results stand for it) or `not-evaluated` (the run stopped before it was evaluated), as the verified report's `selection.endpoints` records. The plan checks a reuse three ways before anything executes: run 2's record of the run 1 artifacts it stands on equals the launch specification's `sft_results` pins, those pins equal run 1's verified report, and the copies run 1's step 17 fetched to `$RUN1_SHARED/artifacts` have the pinned size and SHA-256. Every later command reads this plan, so a reused arm is executed once, as SFT, and never as a measurement of its own:

```sh
uv run --locked python -m tools.posttrain_endpoints plan --specification "$SPEC" --run1-report "$RUN1_REPORT" --run2-report "$RUN2_REPORT" --run1-artifacts "$RUN1_SHARED/artifacts" --output "$SHARED/endpoints.json"
```

It prints each endpoint's source, for example `{"base": "run 1", "sft": "run 1", "rl": "pod", "more-sft": "reuses-run-1-sft (sft)"}`. The plan is never rewritten.

Execute the dev completions of the arms evaluated on the Pod (about 18,000 calls, about 0.85 hours, per arm):

```sh
mkdir -p "$SHARED/results"
for name in $(uv run --locked python -m tools.posttrain_endpoints list --plan "$SHARED/endpoints.json" --list run2-pool-dev); do
  uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$ART/$name/completions.jsonl" --tasks "$POOL/dev.jsonl" --results "$SHARED/results/$name.jsonl" --tokenizer "$TOKENIZER" --workers 8
done
```

Seal their test completions in run 1's ledger, beside run 1's four:

```sh
for name in $(uv run --locked python -m tools.posttrain_endpoints list --plan "$SHARED/endpoints.json" --list run2-pool-test); do
  uv run --locked scratchcoder eval-seal --completions "$ART/$name/completions.jsonl" --ledger "$LEDGER"
done
```

Score their benchmark files on GitHub Actions:

```sh
mkdir -p "$SHARED/benchmarks-results"
for name in $(uv run --locked python -m tools.posttrain_endpoints list --plan "$SHARED/endpoints.json" --list run2-benchmarks); do
  SUITE=$(printf '%s' "$name" | sed -e 's/^eval-rl-//' -e 's/^eval-more-sft-//' -e 's/-greedy$//')
  uv run --locked python -m tools.benchmark_dispatch --completions "$ART/$name/completions.jsonl" --suite "$SUITE" --key "posttrain-runs/2/$POD_ID/$name/completions.jsonl" --repo "$PUBLIC_REPO" --output "$SHARED/benchmarks-results/$name"
done
```

**Only after the owner freezes all four endpoints** (base B, SFT, RL and more-SFT, from the dev and benchmark results), write the unseal record from the plan: each endpoint evaluated on a Pod with its checkpoint SHA-256 and its two test completions files' SHA-256s, an arm that reuses SFT as `{"reuses": "sft", ...}`, an arm never evaluated as `{"not_evaluated": ...}`. One checkpoint is never listed under two names, and the record is never rewritten:

```sh
uv run --locked python -m tools.posttrain_endpoints unseal --plan "$SHARED/endpoints.json" --output "$SHARED/unseal.json"
```

Then execute the test completions of every endpoint that executes, one logged read each, in the window (about 18,000 calls, about 0.85 hours, per endpoint). This is the one read of the sealed test split; a read of SFT's file stands for an arm that reuses it, and the ledger's `opened` event names that arm (`reused_by`). Each `eval-execute` exits 3 while an attempt is unresolved, and running the same line again continues its read:

```sh
for name in eval-base-pool-test-greedy eval-base-pool-test-n8 eval-sft-pool-test-greedy eval-sft-pool-test-n8; do
  uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$RUN1_SHARED/artifacts/$name/completions.jsonl" --tasks "$POOL/test.jsonl" --results "$SHARED/results/$name.jsonl" --tokenizer "$TOKENIZER" --workers 8 --seal-ledger "$LEDGER" --unseal "$SHARED/unseal.json"
done
for name in $(uv run --locked python -m tools.posttrain_endpoints list --plan "$SHARED/endpoints.json" --list run2-pool-test); do
  uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$ART/$name/completions.jsonl" --tasks "$POOL/test.jsonl" --results "$SHARED/results/$name.jsonl" --tokenizer "$TOKENIZER" --workers 8 --seal-ledger "$LEDGER" --unseal "$SHARED/unseal.json"
done
```

Build the scorecard. Its inputs file records the names, each reuse with its reason, and every results file, benchmark artifact and ARC report of both runs, each checked to exist (each benchmark dispatch wrote one `benchmark-<suite>-<completions sha256>.json` into its directory, which the file names directly). It reads base's and SFT's dev results, benchmark results and ARC reports from `$RUN1_SHARED`, where run 1's step 17 wrote them, and every endpoint's pool-test results from `$SHARED/results`, where the reads above wrote them:

```sh
uv run --locked python -m tools.posttrain_endpoints scorecard-inputs --plan "$SHARED/endpoints.json" --run1-shared "$RUN1_SHARED" --shared "$SHARED" --artifacts "$ART" --output "$SHARED/scorecard-inputs.json"
uv run --locked scratchcoder scorecard --inputs "$SHARED/scorecard-inputs.json" --arc-tasks "$RUN1_SHARED/exports/arc-easy-test.jsonl" \
  --pool-baseline "$RUN1_SHARED/results/pool-dev-baseline.json" --output "$SHARED/scorecard.json" --markdown "$SHARED/scorecard.md"
```

The scorecard reports every row beside its constant-answer baseline, the paired comparisons (SFT against base, RL against SFT, more-SFT against SFT, RL against more-SFT) and the HumanEval leak flag. An arm that reuses SFT has SFT's rows under its name, marked `reuses sft: not independent`, no comparison with SFT (one checkpoint), and its comparison with the other arm says it is SFT's results. After this last window the owner resets Docker Desktop's VM data.

## 19. Final billing verification (Mac)

Run 1's step 18 block with run 2's cost limit (`$COST_LIMIT`, the lower of the ceiling and 120% of the estimate): record the total only once two reads at least 30 minutes apart agree:

```sh
export BUCKETS=$(( ($(date +%s) - ORIGIN) / 3600 + 2 ))
uv run --locked python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --last-buckets "$BUCKETS" --output "$CTRL/billing-$(date +%s).json"
python3 - "$CTRL" "$POD_ID" "$COST_LIMIT" <<'PY'
import glob, json, sys

control, pod_id, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
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
if total >= limit:
    sys.exit(f"the Pod cost {total:.4f} USD, at or above its {limit:.2f} USD cost limit")
if len(reads) < 2 or reads[-1][1] != reads[-2][1]:
    sys.exit("the last two reads differ (or there is one read): wait and fetch again")
print(f"billing settled at {reads[-1][1]} USD over two reads, under the {limit:.2f} USD cost limit")
PY
```

## 20. Results

From the verified `report.json` and the archived summary: RL's checks and its selection, its busy seconds and the seconds it excluded, its stop; the control's budget, busy seconds and measured mismatch, its checks and its selection; ARC-Easy for both selected checkpoints; the bridge's waits; every projection, the cost and timings; and `code_revision`. From the Mac: the dev and test results, the seal ledger, the unseal record, the benchmark artifacts and the scorecard. Record the measured numbers in the docs as measured results, distinct from the projections in [docs/posttrain-runs.md](docs/posttrain-runs.md). No "RL beyond more SFT" claim is made without the control.
