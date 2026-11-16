# Post-training runbook: run 1c on one A40 (agent SFT from run 1b's SFT weights)

Run 1c trains run 1b's selected SFT weights (they passed the format gate) for one pass on the agent SFT data, checks the result with run 1b's format gate on the Pod and generates pool dev and the three benchmarks greedily, on one Runpod A40 Pod, uploading every artifact under `posttrain-runs/1c/<pod id>/`. **It is prepared, not approved:** nothing here runs before the owner's final go on the bound specification (step 3), and nothing billable runs before step 6. The training data is frozen by hash in the owner's request; this page names it only as `$AGENT_ROWS`.

This page covers only what differs from [run 1b's runbook](POSTTRAIN_RUN1B_RUNBOOK.md) and [run 1's](POSTTRAIN_RUNBOOK.md). Where a step says "run 1's step N", run that step's blocks unchanged in a terminal where step 2 and step 3 below set the variables.

**What differs from run 1b.**

1. **Stages:** `probe` (on the parent itself, `sft-weights`), `sft`, `format-gate` (run 1b's gate, required: a specification of run 1c without it is refused), `eval-sft` with four greedy chat-v1 files (`pool-dev-greedy`, `humaneval-greedy`, `humaneval-plus-greedy`, `mbpp-plus-greedy`), then run 1's finalization. No ARC, no pool test file, no screening: the specification has no pool test export, no RL or screening input and no `screen` stage, and `check_run1c` refuses any of them. Its `screening` section is run 1b's and unused (the schema requires one).
2. **Floors:** the standing rule over the four hosts measured before this specification (run 1's first launch, run 1, run 1b's first launch, run 1b): **7,991** generated tok/s at batch 128 and **9,049** at batch 256, which are also the planning speeds; the SFT speed check's reference is the mean of run 1's and run 1b's measured rates, **59,141.39** slots/s, floor 53,227.25 (`posttrain_approved.RUN1C_FLOORS`, `RUN1C_SFT_REFERENCE`).
3. **SFT:** run 1b's optimizer (a fresh AdamW, 0.9 / 0.95, weight decay 0.1, clip 1.0), peak 6e-5, 3% warmup, cosine to 10%, 16 x 2 rows per update, sequence 1,024, seed 0, bf16 with fused attention and compile; **one pass**; **no special-row redraw and no row learning-rate scale** (the parent's trained special rows are kept and train at the plain rate). The dev loss over run 1b's code and language dev sets is measured **before the first update (update 0, the parent)** and at every quarter, and the stage keeps the **final weights** after the pass, not the lowest-loss quarter; `sft.json` records `parent_dev_loss` and `dev_loss_change` (the kept weights' loss minus the parent's, set by set, on the same rows).
4. **Bind values:** the training rows' count, size and SHA-256, the update count, the estimate and the cost limit are `null` placeholders in `configs/rehearsal-a40-s-post-1c.json`; `python -m tools.posttrain_approved run1c-bind` fills them from the frozen rows and writes the launch specification (step 3). The worker, the readiness receipt and `check_run1c` refuse a placeholder.
5. **Limits:** the ceiling is **$1.50** and the run also stops at **120% of its estimate**, whichever is lower: the estimate is the projection of the bound specification at its planning rates (setup, every stage and the finalization reserve; `pilot_feasibility.posttrain_estimate_usd`, the `1x generation` row of `pilot_feasibility posttrain`), and the cost limit is the lower of $1.50 and 120% of it, rounded down to the cent. The worker stops at the cost limit, the Mac watchdog one cent inside it, the Pod wall is the cost limit's worth of seconds at $0.60/h and the in-Pod timer fires 180 s before that wall. The 600 s watchdog idle limit, the **900 s** in-Pod idle rule, the 900 s transfer window and the finalization reserve are run 1b's.
6. **Environment record:** before the first stage the worker uploads `environment/environment.json`: the image the provider's Pod record names (the readiness receipt's `pod_image`), its digest when that name pins one, the Python, platform and torch build, and every installed distribution with its version and the list's SHA-256. The same record is in `report.json` and in every label's `record.json`.

## 1. Prelaunch checks (Mac, free)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check
uv run --locked python -m tools.posttrain_run smoke --run 1c --output runs/posttrain-smoke-1c
uv run --locked pytest tests/test_run1c_posttrain.py tests/test_run1c_integration.py tests/test_run1c_bootstrap.py tests/test_run1c_runbook.py tests/test_executed_specifications.py tests/test_posttrain_run1b.py tests/test_sft_format_gate.py tests/test_posttrain_run.py tests/test_posttrain_bootstrap.py -q
```

Expected: `make check` green; the smoke prints `"status": "completed"` with every stage `completed`, `"objects_outside_prefix": []`, a `format_gate` with `"passed": true`, `"sft_kept_weights": "final"`, an `sft_parent_dev_loss` at update 0 and `"environment_uploaded": true`. The integration test replays run 1c's smoke into the real supervisor and watchdog, once with the gate passing and once with it failing (`format_gate_failed`, nothing after the gate), under the policy the smoke's cost limit derives. The smoke's parent is a weights-only file written by `rl.checkpoint.save_weights`, the format of run 1b's SFT weights.

## 2. Credentials and environment (Mac, free)

Run 1's step 2 for `.env` (a new Pod token for this run only), then set run 1c's values:

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi
done
caffeinate -dimsu -w $$ &
export RUN=1c
export CTRL_ROOT=runs/posttrain-controller
export KNOWN_HOSTS=~/.ssh/posttrain_1c_known_hosts
export SPEC_TEMPLATE=configs/rehearsal-a40-s-post-1c.json
export SHARED=runs/posttrain-shared-1c && mkdir -p "$SHARED"
export SPEC=$SHARED/rehearsal-a40-s-post-1c-launch.json
export RUN1_SHARED=runs/posttrain-shared
export RUN1B_SHARED=runs/posttrain-shared-1b
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts
export PUBLIC_REPO=kudratsingh/personal-coding-agent
export POOL=runs/task-pool-20261025/pool
export SFT=runs/sft-data-20261027
export TOKENIZER=runs/tokenizer-freeze-65536-20261020/tokenizer.json
export EXPORTS=$RUN1_SHARED/exports
export BINDING=$SHARED/binding.json
export AGENT_ROWS=<frozen training rows>
export AGENT_ROWS_SHA256=<approved training rows sha256>
```

`AGENT_ROWS_SHA256` is the training rows' SHA-256 as the owner's request quotes it. Paste it in place of the placeholder; never derive it from the file, or the bind's check would only compare the file with itself. `SFT` is run 1b's data: run 1c's dev sets are run 1b's, byte for byte. `EXPORTS` is run 1's exports directory; run 1c reads only its pool dev export, and the sealed test split is not read.

## 3. Bind the training rows (Mac, free; the owner reviews the result)

Check the frozen rows against the SHA-256 in the owner's request, then write the launch specification (the command refuses an existing output, so it never rewrites one) and read its derived limits:

```sh
shasum -a 256 "$AGENT_ROWS"
uv run --locked python -m tools.posttrain_approved run1c-bind --sft-train "$AGENT_ROWS" --expected-sha256 "$AGENT_ROWS_SHA256" --output "$SPEC"
export COST_LIMIT=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field limit_usd)
export WALL=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field wall_seconds)
export WATCHDOG_COST=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field watchdog_cost_stop_usd)
export TIMER=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field self_destruct_seconds)
echo "run $RUN: cost limit $COST_LIMIT USD, wall $WALL s, timer $TIMER s, watchdog cost stop $WATCHDOG_COST USD"
```

The first line must print the request's SHA-256, and the bind refuses the rows unless their SHA-256 is `$AGENT_ROWS_SHA256`. It also refuses the file of another input (a dev set) and any row whose source is not an agent edit or repair row or a code replay source (`posttrain_approved.RUN1C_ROW_SOURCES`; a language row, for one), so the bound rows are a code-only mix. The bind prints one JSON line: the launch specification's `sha256`, the rows' `sft_train_sha256`, `train_rows`, `planned_updates`, the `gate_estimate_usd` and `cost_limit_usd`, the watchdog stop, the wall, the timer and the launch rule's figures (`lifetime_seconds`, `planning_seconds_needed`, `setup_allowance_seconds`). It refuses rows the worker would drop (longer than the sequence length), and a cost limit whose lifetime cannot hold the planned setup plus the whole run at the planning rates. The owner's final go quotes the launch specification's SHA-256, the estimate and the cost limit; nothing below runs without it.

Check the Mac's copy of the parent against its pin (run 1b's fetched artifacts):

```sh
shasum -a 256 "$RUN1B_SHARED/artifacts/sft-weights/checkpoint.pt"
python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"]["sft-weights"]["sha256"])' "$SPEC"
```

Both lines must print `f9064cd0e277bf7c2158b0560879494e2b0f3480c89587f45d47c96f0a59c059`.

## 4. Quote check and projection (Mac, free)

```sh
uv run --locked python -m tools.pilot_feasibility posttrain --specification "$SPEC" --quoted-gpu-hourly 0.59 --output "$SHARED/feasibility.json"
python3 -c 'import json,sys; f=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); t=f["scenarios"]["1x generation"]["total_usd"]; print("the projection is the bound estimate" if t == s["limits"]["gate_estimate_usd"] else f"MISMATCH: {t}")' "$SHARED/feasibility.json" "$SPEC"
```

Replace `0.59` with the console price; above $0.59, stop and ask the owner (the estimate, and so the cost limit, were priced at $0.59). The second line must print `the projection is the bound estimate`. At half the planning speed the probe stops the run after the setup allowance, the probe and the finalization reserve. Confirm the account balance covers $1.50 plus a margin and that no network volume exists.

## 5. Exports, benchmarks, binding and staging (Mac, at the go; R2 storage only)

Check the pool dev export against the specification and that it carries no hidden test, fetch the pinned benchmark releases, bind every input (the tokenizer and the parent are in R2 already; the rest is staged under `posttrain-inputs/<sha256>/<file>`):

```sh
uv run --locked python -m tools.posttrain_binding check-exports --specification "$SPEC" --input pool-dev="$EXPORTS/dev-export.jsonl"
uv run --locked python -m tools.posttrain_binding benchmarks --output "$RUN1_SHARED/benchmarks"
uv run --locked python -m tools.posttrain_binding build --specification "$SPEC" --run 1c --output "$BINDING" \
  --input sft-train="$AGENT_ROWS" --input sft-code-dev="$SFT/code_dev.jsonl" \
  --input sft-language-dev="$SFT/language_dev.jsonl" --input pool-dev="$EXPORTS/dev-export.jsonl" \
  --input humaneval="$RUN1_SHARED/benchmarks/HumanEval.jsonl.gz" \
  --input humaneval-plus="$RUN1_SHARED/benchmarks/HumanEvalPlus-OriginFmt.jsonl.gz" \
  --input mbpp-plus="$RUN1_SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz"
```

`check-exports` must print `{"exports": {"pool-dev": 2000}, "hidden_tests": 0}`. The owner reviews the binding and the `sha256` it printed; paste that value, then stage and record the Mac's pinned copy of the parent (the readiness receipt requires it; a copy already in `runs/posttrain-parents` with the pinned bytes is kept):

```sh
export BINDING_SHA256=<reviewed binding sha256>
printf '%s  %s\n' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -
uv run --locked python -m tools.posttrain_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --receipt "$SHARED/stage-receipt.json"
uv run --locked python -m tools.posttrain_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --root runs/posttrain-parents --only sft-weights --receipt "$SHARED/mac-copy.json" --key-prefix R2_
```

## 6. The Pod and the run (BILLABLE)

Only after the owner's explicit final go for run 1c on the bound specification. Create the Pod with run 1c's create body (`configs/runpod-pod-a40-s-posttrain-1c.json`, named `posttrain-a40-s-1c`); the block refuses to create one when this terminal names a Pod or a controller directory has no `TEARDOWN`:

```sh
OPEN_PODS=$(find "$CTRL_ROOT/$RUN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r DIR; do grep -qs '^export TEARDOWN=' "$DIR/run.env" || printf '%s ' "$DIR"; done)
if [ -n "${POD_ID:-}" ]; then
  echo "run $RUN: NOT creating a Pod: this terminal already names Pod $POD_ID, which may still bill. Confirm it gone with run 1's step 15 first."
elif [ -n "$OPEN_PODS" ]; then
  echo "run $RUN: NOT creating a Pod: no TEARDOWN is recorded in ${OPEN_PODS% }. End each such Pod with run 1's step 15 and ask the owner before creating another."
else
  export ORIGIN=$(date +%s)
  export LAUNCH=$CTRL_ROOT/$RUN/launch-$ORIGIN && mkdir -p "$LAUNCH"
  cp configs/runpod-pod-a40-s-posttrain-1c.json "$LAUNCH/create-body.json"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS --fail-with-body -X POST https://api.runpod.io/v2/pods \
    -H "Content-Type: application/json" --data @"$LAUNCH/create-body.json" -o "$LAUNCH/create-response.json" -w '%{http_code}\n' > "$LAUNCH/create-http-code"
  export POD_ID=$(python3 -c 'import json,re,sys; v=json.load(open(sys.argv[1])); i=v.get("id") if isinstance(v, dict) else None; ok=isinstance(i, str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,127}", i); print(i) if ok else sys.exit("no valid Pod id in " + sys.argv[1])' "$LAUNCH/create-response.json")
  export SELF_DESTRUCT_DEADLINE=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --origin "$ORIGIN" --field self_destruct_deadline_unix)
  export CTRL=$CTRL_ROOT/$RUN/$POD_ID
  if [ -z "$POD_ID" ]; then
    echo "run $RUN: the create request ended with HTTP $(cat "$LAUNCH/create-http-code") and names no valid Pod id. A Pod MAY STILL EXIST: ask the owner to check the Runpod console for posttrain-a40-s-1c and decide before any retry."
    export CTRL=$LAUNCH
  elif mkdir "$CTRL"; then
    mv "$LAUNCH/create-body.json" "$LAUNCH/create-response.json" "$LAUNCH/create-http-code" "$CTRL/" && rmdir "$LAUNCH"
    printf 'export ORIGIN=%s POD_ID=%s SELF_DESTRUCT_DEADLINE=%s\n' "$ORIGIN" "$POD_ID" "$SELF_DESTRUCT_DEADLINE" > "$CTRL/run.env"
    echo "run $RUN: Pod $POD_ID, controller directory $CTRL, timer deadline $SELF_DESTRUCT_DEADLINE"
  else
    echo "run $RUN: $CTRL already exists and is NOT this Pod's. Terminate Pod $POD_ID with run 1's step 15 (TERMINATE=yes)."
    export CTRL=$LAUNCH
  fi
fi
```

`SELF_DESTRUCT_DEADLINE` is `ORIGIN` plus `$TIMER`. Then run 1's step 7 from "Capture the identity" on; [run 2's step 8](POSTTRAIN_RUN2_RUNBOOK.md#8-code-specification-binding-and-the-pods-r2-key-mac-to-pod-billable), which also ships the launch specification (it is not in the repository) to `/workspace/posttrain/specification.json` and checks its SHA-256 on the Pod; and run 1's steps 9 to 16, with run 1c's values from steps 2 and 3: the policy (`posttrain_ops limits --policy-output`, the wall and the cost stop exactly those derived from the cost limit, which the receipt and the worker require), the watchdog, the bootstrap (`--run 1c --lifetime-seconds $TIMER`), the receipt (`--specification "$SPEC" --mac-copy "$SHARED/mac-copy.json"`, no `--sft-gate`), the worker with `--specification /workspace/posttrain/specification.json` as in [run 2's step 12](POSTTRAIN_RUN2_RUNBOOK.md#12-worker-launch-pod-billable), the supervisor (`--run 1c`), monitoring, verification, termination and token revocation. What to expect that differs:

- **Bootstrap:** the CPU tests include `tests/test_run1c_posttrain.py`, `tests/test_run1c_integration.py` and `tests/test_executed_specifications.py`; the R2 round trip runs under `posttrain-runs/1c/<pod id>/`.
- **Receipt:** it refuses a launch specification with a placeholder or that differs from the approved run 1c in anything but the bind values, a recomputed estimate other than the written one, a cost limit other than its derivation, a policy other than the cost limit's, and a Pod connection record with no image (`pod_image`).
- **Monitoring:** the first label is `environment`; then `probe`, `sft` (its dev-loss checks at update 0 and at each quarter; one weights save, after the last update), `format-gate`, the four `eval-sft-<file>` labels. `report.json`'s `sft` has `kept_weights: final`, `parent_dev_loss` and `dev_loss_change`; `format_gate` the gate's verdict.
- **A failed gate** ends the run `stopped` with `format_gate_failed`: `eval-sft` is `skipped`, nothing more is paid, and the run is verified and archived as any other. Report the failed checks to the owner; do not relaunch without the owner, and no agent evaluation of those weights follows.
- **The cost stop:** the worker stops with `cost_limit` at `$COST_LIMIT` and the watchdog deletes the Pod at `$WATCHDOG_COST`, both below the $1.50 ceiling when the estimate is below $1.25.

## 7. Run 1c's billing (Mac)

Run 1's step 18 against run 1c's cost limit: fetch the hourly buckets, wait at least 30 minutes, fetch again:

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

## 8. After the run: the CPU format gate and execution on the Mac (Mac execution window)

The execution window's rules are run 1's step 17 (Docker Desktop's file sharing emptied, the SSH agent cleared, the Pod token revoked and out of `.env`). Fetch run 1c's artifacts pinned to its verified report into its own directory and check the window before every batch:

```sh
export ART=$SHARED/artifacts
export SANDBOX_IMAGE_ID=$(docker image inspect --format '{{.Id}}' python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe)
uv run --locked python -m tools.posttrain_ops fetch-artifacts --report "$CTRL/supervisor/final/report.json" --output "$ART" --receipt "$SHARED/artifacts-fetch.json"
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check.json"
```

**Re-run the format gate on the CPU (required).** The owner's rule is that the CPU format gate runs after every SFT run, on the fetched weights and the same inputs as the Pod's gate. Only exit 0 with a written record is a pass; on any other result stop here and report to the owner (no execution, no agent evaluation):

```sh
uv run --locked scratchcoder sft-format-gate --checkpoint "$ART/sft-weights/checkpoint.pt" --tokenizer "$TOKENIZER" --rows "$SFT/code_dev.jsonl" --tasks "$EXPORTS/dev-export.jsonl" --benchmark humaneval="$RUN1_SHARED/benchmarks/HumanEval.jsonl.gz" --benchmark mbpp-plus="$RUN1_SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz" --output "$SHARED/format-gate-cpu.json" --threads 4
```

Then execute the pool dev completions (run the window check again first if time has passed) and score the three benchmark files on GitHub Actions. There are no pool test files to seal:

```sh
mkdir -p "$SHARED/results" "$SHARED/benchmarks-results"
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check-dev.json"
uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$ART/eval-sft-pool-dev-greedy/completions.jsonl" --tasks "$POOL/dev.jsonl" --results "$SHARED/results/eval-sft-pool-dev-greedy.jsonl" --tokenizer "$TOKENIZER" --workers 8
for name in eval-sft-humaneval-greedy eval-sft-humaneval-plus-greedy eval-sft-mbpp-plus-greedy; do
  SUITE=$(printf '%s' "$name" | sed -e 's/^eval-sft-//' -e 's/-greedy$//')
  uv run --locked python -m tools.benchmark_dispatch --completions "$ART/$name/completions.jsonl" --suite "$SUITE" --key "posttrain-runs/1c/$POD_ID/$name/completions.jsonl" --repo "$PUBLIC_REPO" --output "$SHARED/benchmarks-results/$name"
done
```

**The local agent evaluation is a separate step.** The agent evaluation of the kept weights runs on the Mac on CPU, in the window-checked sandbox, at one clean commit, under its own approval and call budget; it is not part of this runbook and starts only after the CPU format gate above passed.

## 9. Results

From run 1c's verified `report.json` and summary: the probe's rates, the parent's dev loss at update 0 and each quarter's (`sft.parent_dev_loss`, `sft.quarters`), the change from the parent to the kept final weights per dev set (`sft.dev_loss_change`; the language set measures what the agent data costs the language rows, which it does not contain), the format gate's checks on the Pod and the CPU gate's record, the cost against the estimate and the cost limit, the timings and the environment record. Run 1b's final-quarter dev losses are the cross-run reference; the within-run comparison is the one on the same rows. From the Mac: the pool dev results and the benchmark artifacts. Record measured numbers as measured results in [docs/posttrain-runs.md](docs/posttrain-runs.md), apart from its projections.

## Pinning the image (proposed, not done)

The create body names the image by tag (`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, run 1b's), and the provider's Pod record reports the tag, not a digest, so run 1c's environment record carries the tag with `digest: null` and the installed packages (torch among them) as the record of what ran. Pinning needs the tag's digest from the registry, which needs the network: resolve it on the Mac at the go (for example `docker buildx imagetools inspect runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`), write `"image": "runpod/pytorch@sha256:<digest>"` into the create body in a reviewed commit, and the environment record then carries that digest. Whether to pin before this run is the owner's decision; no digest is guessed here.
