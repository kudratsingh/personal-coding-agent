# Post-training runbook: run 1b on one A40 (the SFT re-run with the format gate)

Run 1 completed, but its SFT weights failed the SFT format gate: the tied embedding rows of the chat special tokens, which pretraining never trained, were nearly collinear (pairwise cosines of about 0.997), so after `<|assistant_start|>` the model ended its turn and every completion was empty ([format gate](docs/walkthroughs/sft-format-gate.md)). Run 1's base results stand; its SFT results are recorded as a formatting failure, and its two sealed SFT test files are withdrawn from the seal ledger (step 8). Run 1b re-runs SFT from run B's `mixed-final` under the owner's approval of 2026-10-27, on one Runpod A40 Pod, and uploads every artifact under `posttrain-runs/1b/<pod id>/`.

This page covers only what differs from [run 1's runbook](POSTTRAIN_RUNBOOK.md). Where a step says "run 1's step N", run that step's blocks unchanged in a terminal where step 2 below set the variables: every run 1 block reads `RUN`, `SPEC`, `SHARED`, `SFT`, `WALL`, `TIMER` and `WATCHDOG_COST`, so it acts for run 1b.

**What differs from run 1.**

1. **Stages:** `probe` (run 1's, the same floors: 8,279 generated tok/s at batch 128, 9,268 at batch 256), `sft`, `format-gate`, `arc-sft`, `eval-sft` (the seven chat-v1 files at batch 128), `screen`, then run 1's finalization. No `arc-base` or `eval-base`: run 1's base results stand.
2. **SFT:** two passes over the 3,390 rows (212 updates of 16 x 2 rows), peak 6e-5, 3% warmup (6 updates), cosine to 10%, dev loss at updates 53, 106, 159 and 212, the lowest kept. Before the first update the tied embedding rows of IDs 257 to 271 (the chat specials and the reserved tokens; not `<|bos|>`, which pretraining trains) are redrawn from the ordinary rows' mean and per-dimension spread with seed 0, and they train at 10x the learning rate ([special-token rows](docs/walkthroughs/special-token-rows.md)).
3. **Data:** `runs/sft-data-20261027`, the SFT data tool's rebuild with `--omit-public-tests`: 950 of the 2,871 code train rows and 645 of the 2,000 code dev rows leave their public tests out of the user turn, chosen by a hash of the row ID. The language rows and the RL candidate list are run 1's, byte for byte.
4. **The format gate on the Pod:** after `sft` and before anything reads the weights, the gate runs on the dev-loss-selected checkpoint (only that one) on the GPU, at the CPU command's thresholds: the first assistant token, the block's close, the separation of `<|assistant_end|>`, `<|python_start|>` and `<|python_end|>` (largest cosine strictly below 0.9) and a greedy smoke of 16 pool dev prompts. A smoke of 8 HumanEval and 8 MBPP+ chat-v1 turns is recorded, not judged. Below any bar the run stops with `format_gate_failed`: the gate record (`format-gate/gate.json`) and the weights (`sft-weights`) are already in R2, the run finalizes in full and nothing paid follows.
5. **MBPP+ prompts:** the MBPP+ chat-v1 user turn is EvalPlus's prompt, the task text and its first test in a docstring, so the function name is known. The chat-v1 benchmark protocol's record and hash changed with it (run 1's SFT benchmark files were generated under the old one); the base protocols did not ([evaluation](docs/evaluation.md)).
6. **Limits, all derived from the $1.50 ceiling** (`posttrain_ops.ceiling_limits`): at $0.60/h the worker stops at **$1.50**, the Mac watchdog at **$1.49**, the Pod wall is **9,000 s** and the in-Pod timer fires at **8,820 s**. The 600 s watchdog idle limit, the **900 s** in-Pod idle rule, the 900 s transfer window and the finalization reserve are run 1's.

The projection at the planning speeds is about 1.54 hours and $0.92 (step 4). The launch rule needs 4,468 s at the planning rates, so setup may take up to about 73 minutes of the 8,820 s.

## 1. Prelaunch checks (Mac, free)

```sh
git fetch origin && git checkout main && git pull --ff-only origin main
make setup && make check
uv run --locked python -m tools.posttrain_run smoke --run 1b --output runs/posttrain-smoke-1b
uv run --locked pytest tests/test_posttrain_run1b.py tests/test_posttrain_run1b_integration.py tests/test_posttrain_run1b_runbook.py tests/test_special_rows.py tests/test_sft_format_gate.py tests/test_seal_withdrawal.py tests/test_benchmark_prompts.py tests/test_posttrain_run.py tests/test_posttrain_integration.py tests/test_posttrain_bootstrap.py -q
```

Expected: `make check` green; the smoke prints `"status": "completed"` with every stage `completed`, `"objects_outside_prefix": []` and a `format_gate` with `"passed": true`; every listed test passes. The integration test replays run 1b's smoke into the real supervisor and watchdog twice, once with the gate passing and once with it failing (`format_gate_failed`, nothing after the gate).

## 2. Credentials and environment (Mac, free)

Run 1's step 2 for `.env` (a new Pod token for this run only), then set run 1b's values:

```sh
set -a; source .env; set +a
for name in RUNPOD_API_KEY R2_ENDPOINT R2_BUCKET R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
  if [ -n "$(printenv "$name")" ]; then echo "$name is set"; else echo "MISSING $name"; fi
done
caffeinate -dimsu -w $$ &
export RUN=1b
export CTRL_ROOT=runs/posttrain-controller
export KNOWN_HOSTS=~/.ssh/posttrain_1b_known_hosts
export SPEC=configs/rehearsal-a40-s-post-1b.json
export SHARED=runs/posttrain-shared-1b && mkdir -p "$SHARED"
export RUN1_SHARED=runs/posttrain-shared
export POD_SSH_KEY=~/.ssh/runpod_ed25519
export PRIVATE_REPO=kudratsingh/personal-coding-agent-artifacts
export PUBLIC_REPO=kudratsingh/personal-coding-agent
export POOL=runs/task-pool-20261025/pool
export SFT=runs/sft-data-20261027
export TOKENIZER=runs/tokenizer-freeze-65536-20261020/tokenizer.json
export EXPORTS=$RUN1_SHARED/exports
export BINDING=$SHARED/binding.json
export WALL=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field wall_seconds)
export WATCHDOG_COST=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field watchdog_cost_stop_usd)
export TIMER=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --field self_destruct_seconds)
echo "run $RUN: wall $WALL s, timer $TIMER s, watchdog cost stop $WATCHDOG_COST USD"
```

The last line must print `wall 9000 s, timer 8820 s, watchdog cost stop 1.49 USD`. `EXPORTS` is run 1's exports directory: run 1b reads the same Pod exports, so the sealed test split is not read again.

## 3. The rebuilt SFT data (Mac, free)

The data was built once with the project's SFT data tool from run 1's pool train and dev splits and language conversations (the command below refuses an existing output, so it never rebuilds over it):

```sh
uv run --locked python -m tools.sft_data build --pool-spec configs/task-pool-rehearsal.json --pool-dir "$POOL" --language runs/sft-oasst2-20261026/language/conversations.jsonl --output "$SFT" --omit-public-tests
```

Check every data file against the specification's pins and the record's rule:

```sh
python3 -c 'import hashlib,json,sys; s=json.load(open(sys.argv[1]))["inputs"]; d=sys.argv[2]; files={"sft-train":"train.jsonl","sft-code-dev":"code_dev.jsonl","sft-language-dev":"language_dev.jsonl","rl-candidates-list":"rl_candidates.jsonl"}; bad=[n for n,f in files.items() if hashlib.sha256(open(d+"/"+f,"rb").read()).hexdigest()!=s[n]["sha256"]]; r=json.load(open(d+"/record.json"))["code"]["public_tests_omitted"]; print("SFT data matches the specification" if not bad else "MISMATCH: " + ", ".join(bad), r["train_rows"], r["dev_rows"])' "$SPEC" "$SFT"
```

It must print `SFT data matches the specification 950 645`.

## 4. Quote check and projection (Mac, free)

As run 1's step 3, with run 1b's specification:

```sh
uv run --locked python -m tools.pilot_feasibility posttrain --specification "$SPEC" --quoted-gpu-hourly 0.59 --output "$SHARED/feasibility.json"
```

Replace `0.59` with the console price; above $0.59, stop and ask the owner. At the planning speeds run 1b is about 1.54 hours and $0.92 against the $1.50 ceiling (setup 0.30 h, probe 0.10, SFT 0.09, the format gate 0.04, ARC 0.02, evaluation 0.67, screening 0.07, finalization 0.25); at half speed the probe stops the run after at most about $0.39. Confirm the account balance covers $1.50 plus a margin and that no network volume exists.

## 5. Exports, benchmarks, binding and staging (Mac, at the go; R2 storage only)

Run 1b reads run 1's Pod exports and benchmark releases, unchanged. Check them against the specification (the test export is pinned only now, from run 1's bytes) and that none carries a hidden test:

```sh
python3 -c 'import hashlib,json,sys; s=json.load(open(sys.argv[1]))["inputs"]; bad=[n for n,p in (("pool-dev",sys.argv[2]),("rl-candidates",sys.argv[3]),("arc-easy-test",sys.argv[4])) if hashlib.sha256(open(p,"rb").read()).hexdigest()!=s[n]["sha256"]]; print("exports match the specification" if not bad else "MISMATCH: " + ", ".join(bad))' "$SPEC" "$EXPORTS/dev-export.jsonl" "$EXPORTS/candidates-export.jsonl" "$EXPORTS/arc-easy-test.jsonl"
uv run --locked python -m tools.posttrain_binding check-exports --specification "$SPEC" --input pool-dev="$EXPORTS/dev-export.jsonl" --input pool-test="$EXPORTS/test-export.jsonl" --input rl-candidates="$EXPORTS/candidates-export.jsonl"
uv run --locked python -m tools.posttrain_binding benchmarks --output "$RUN1_SHARED/benchmarks"
```

The first must print `exports match the specification` and `check-exports` `{"exports": {"pool-dev": 2000, "pool-test": 2000, "rl-candidates": 1000}, "hidden_tests": 0}`. Then bind every input; run B's `mixed-final` and the tokenizer are in R2 already, the rest is staged under `posttrain-inputs/<sha256>/<file>` (the new SFT rows among them):

```sh
uv run --locked python -m tools.posttrain_binding build --specification "$SPEC" --run 1b --output "$BINDING" \
  --input sft-train="$SFT/train.jsonl" --input sft-code-dev="$SFT/code_dev.jsonl" \
  --input sft-language-dev="$SFT/language_dev.jsonl" --input rl-candidates-list="$SFT/rl_candidates.jsonl" \
  --input pool-dev="$EXPORTS/dev-export.jsonl" --input pool-test="$EXPORTS/test-export.jsonl" \
  --input rl-candidates="$EXPORTS/candidates-export.jsonl" --input arc-easy-test="$EXPORTS/arc-easy-test.jsonl" \
  --input humaneval="$RUN1_SHARED/benchmarks/HumanEval.jsonl.gz" \
  --input humaneval-plus="$RUN1_SHARED/benchmarks/HumanEvalPlus-OriginFmt.jsonl.gz" \
  --input mbpp-plus="$RUN1_SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz"
```

The owner reviews the binding and the `sha256` it printed; paste that value, then stage and copy run B's `mixed-final` to the Mac, pinned (the readiness receipt requires this copy; one already in `runs/posttrain-parents` with the pinned bytes is kept):

```sh
export BINDING_SHA256=<reviewed binding sha256>
printf '%s  %s\n' "$BINDING_SHA256" "$BINDING" | shasum -a 256 -c -
uv run --locked python -m tools.posttrain_binding stage --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --receipt "$SHARED/stage-receipt.json"
uv run --locked python -m tools.posttrain_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" --root runs/posttrain-parents --only b-mixed-final --receipt "$SHARED/mac-copy.json" --key-prefix R2_
```

## 6. The Pod and the run (BILLABLE)

Only after the owner's explicit go for run 1b. Create the Pod with run 1b's create body (`configs/runpod-pod-a40-s-posttrain-1b.json`, named `posttrain-a40-s-1b`); the block refuses to create one when this terminal names a Pod or a controller directory has no `TEARDOWN`:

```sh
OPEN_PODS=$(find "$CTRL_ROOT/$RUN" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r DIR; do grep -qs '^export TEARDOWN=' "$DIR/run.env" || printf '%s ' "$DIR"; done)
if [ -n "${POD_ID:-}" ]; then
  echo "run $RUN: NOT creating a Pod: this terminal already names Pod $POD_ID, which may still bill. Confirm it gone with run 1's step 15 first."
elif [ -n "$OPEN_PODS" ]; then
  echo "run $RUN: NOT creating a Pod: no TEARDOWN is recorded in ${OPEN_PODS% }. End each such Pod with run 1's step 15 and ask the owner before creating another."
else
  export ORIGIN=$(date +%s)
  export LAUNCH=$CTRL_ROOT/$RUN/launch-$ORIGIN && mkdir -p "$LAUNCH"
  cp configs/runpod-pod-a40-s-posttrain-1b.json "$LAUNCH/create-body.json"
  printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" | curl --config - -sS --fail-with-body -X POST https://api.runpod.io/v2/pods \
    -H "Content-Type: application/json" --data @"$LAUNCH/create-body.json" -o "$LAUNCH/create-response.json" -w '%{http_code}\n' > "$LAUNCH/create-http-code"
  export POD_ID=$(python3 -c 'import json,re,sys; v=json.load(open(sys.argv[1])); i=v.get("id") if isinstance(v, dict) else None; ok=isinstance(i, str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,127}", i); print(i) if ok else sys.exit("no valid Pod id in " + sys.argv[1])' "$LAUNCH/create-response.json")
  export SELF_DESTRUCT_DEADLINE=$(uv run --locked python -m tools.posttrain_ops limits --specification "$SPEC" --origin "$ORIGIN" --field self_destruct_deadline_unix)
  export CTRL=$CTRL_ROOT/$RUN/$POD_ID
  if [ -z "$POD_ID" ]; then
    echo "run $RUN: the create request ended with HTTP $(cat "$LAUNCH/create-http-code") and names no valid Pod id. A Pod MAY STILL EXIST: ask the owner to check the Runpod console for posttrain-a40-s-1b and decide before any retry."
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

`SELF_DESTRUCT_DEADLINE` is `ORIGIN` + 8,820 s. Then run 1's step 7 from "Capture the identity" on, and run 1's steps 8 to 16, unchanged: the policy, the watchdog, the bootstrap (`--run 1b --lifetime-seconds 8820`), the receipt (`--mac-copy "$SHARED/mac-copy.json"`), the worker (`--specification configs/rehearsal-a40-s-post-1b.json`), the supervisor (`--run 1b`), monitoring, verification, termination and token revocation all take run 1b's values from step 2's variables. What to expect that differs:

- **Bootstrap:** `self-destruct.json` shows `deadline_unix` equal to `SELF_DESTRUCT_DEADLINE`; the R2 round trip ran under `posttrain-runs/1b/<pod id>/`; the CPU tests include the special-row, format-gate and run 1b worker tests.
- **Monitoring:** the stages are `probe`, `sft` (212 updates, about 5 minutes with the compile and four dev-loss checks, at updates 53, 106, 159 and 212), `format-gate` (planned at 2 minutes), `arc-sft`, `eval-sft`, `screen`. `report.json`'s `sft.special_rows` records the redraw (rows 257 to 271, seed 0, the rows' SHA-256 before and after) and `format_gate` the gate's verdict, its checks and the benchmark smoke's counts. R2 shows `format-gate/gate.json` right after the weights.
- **A failed gate** ends the run `stopped` with `format_gate_failed`: `arc-sft`, `eval-sft` and `screen` are `skipped`, nothing more is paid, and the supervisor verifies and archives the stopped run as any other (run 1's step 14). Report the gate record's failed checks to the owner; do not relaunch without the owner. Run 2 cannot start from those weights (its gate step and its readiness receipt refuse weights without a passing record of their own).

## 7. Run 1b's billing (Mac)

Run 1's step 18 with run 1b's ceiling: fetch the hourly buckets, wait at least 30 minutes, fetch again:

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
if total >= 1.5:
    sys.exit(f"the Pod cost {total:.4f} USD, at or above its $1.50 ceiling")
if len(reads) < 2 or reads[-1][1] != reads[-2][1]:
    sys.exit("the last two reads differ (or there is one read): wait and fetch again")
print(f"billing settled at {reads[-1][1]} USD over two reads, under the $1.50 ceiling")
PY
```

## 8. After the run: execution on the Mac (Mac execution window)

The execution window's rules are run 1's step 17 (Docker Desktop's file sharing emptied, the SSH agent cleared, the Pod token revoked and out of `.env`). Fetch run 1b's artifacts pinned to its verified report into its own directory, so they never mix with run 1's labels of the same names, and check the window:

```sh
export ART=$SHARED/artifacts
export SANDBOX_IMAGE_ID=$(docker image inspect --format '{{.Id}}' python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe)
uv run --locked python -m tools.posttrain_ops fetch-artifacts --report "$CTRL/supervisor/final/report.json" --output "$ART" --receipt "$SHARED/artifacts-fetch.json"
uv run --locked python -m tools.mac_window_check check --image-id "$SANDBOX_IMAGE_ID" --dotenv .env --receipt "$SHARED/window-check.json"
```

**Withdraw run 1's SFT test files.** Run 1 sealed four pool-test files: two of base B, which stand, and two of its SFT weights, which failed the format gate. One append-only ledger event withdraws every sealed file of that checkpoint with its reason; they can never be sealed again, listed in an unseal record or executed, and run 1b's SFT weights stand for the SFT endpoint. It is refused once any read of the test split has opened, and nothing undoes it. Run 1 has two controller directories (one launch stopped at its probe), so the block takes the one report whose status is `completed`; with none or more than one it prints how many it found, sets no path, and the withdrawal refuses the empty checkpoint hash:

```sh
export RUN1_REPORT=$(python3 -c 'import json,pathlib,sys; found=[str(p) for p in sorted(pathlib.Path(sys.argv[1]).glob("*/supervisor/final/report.json")) if json.loads(p.read_text()).get("status") == "completed"]; print(found[0]) if len(found) == 1 else sys.exit(f"{len(found)} completed run 1 reports, expected exactly one")' "$CTRL_ROOT/1")
export RUN1_SFT_SHA256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sft"]["selected"]["sha256"])' "$RUN1_REPORT")
uv run --locked scratchcoder eval-withdraw --ledger "$RUN1_SHARED/seal/ledger.jsonl" --endpoint sft --checkpoint-sha256 "$RUN1_SFT_SHA256" --reason "formatting failure: run 1's SFT weights failed the SFT format gate (collinear special-token rows, empty completions); replaced by run 1b's SFT weights"
```

It prints the `withdrawn` event naming the two files. This holds whatever run 1b's own gate gave; when run 1b stopped at its gate, stop here and report.

**Re-check the gate on the CPU** (optional, free): the CPU command on the fetched weights gives the Pod's verdict on the same inputs:

```sh
uv run --locked scratchcoder sft-format-gate --checkpoint "$ART/sft-weights/checkpoint.pt" --tokenizer "$TOKENIZER" --rows "$SFT/code_dev.jsonl" --tasks "$EXPORTS/dev-export.jsonl" --benchmark humaneval="$RUN1_SHARED/benchmarks/HumanEval.jsonl.gz" --benchmark mbpp-plus="$RUN1_SHARED/benchmarks/MbppPlus-OriginFmt.jsonl.gz" --output "$SHARED/format-gate-cpu.json" --threads 4
```

Execute the two SFT dev completions files (base's dev results are run 1's), seal the two SFT test files in run 1's ledger, score the screening attempts, and score the three SFT benchmark files on GitHub Actions:

```sh
mkdir -p "$SHARED/results" "$SHARED/benchmarks-results"
for name in eval-sft-pool-dev-greedy eval-sft-pool-dev-n8; do
  uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder eval-execute --completions "$ART/$name/completions.jsonl" --tasks "$POOL/dev.jsonl" --results "$SHARED/results/$name.jsonl" --tokenizer "$TOKENIZER" --workers 8
done
for name in eval-sft-pool-test-greedy eval-sft-pool-test-n8; do
  uv run --locked scratchcoder eval-seal --completions "$ART/$name/completions.jsonl" --ledger "$RUN1_SHARED/seal/ledger.jsonl"
done
uv run --locked python -m tools.mac_window_check exec --image-id "$SANDBOX_IMAGE_ID" --dotenv .env -- uv run --locked scratchcoder rl-screen --tasks "$POOL/train.jsonl" --candidates "$SFT/rl_candidates.jsonl" --attempts "$ART/screen/attempts.jsonl" --output "$SHARED/screening.json" --timeout 5600 --workers 8 --require-header
for name in eval-sft-humaneval-greedy eval-sft-humaneval-plus-greedy eval-sft-mbpp-plus-greedy; do
  SUITE=$(printf '%s' "$name" | sed -e 's/^eval-sft-//' -e 's/-greedy$//')
  uv run --locked python -m tools.benchmark_dispatch --completions "$ART/$name/completions.jsonl" --suite "$SUITE" --key "posttrain-runs/1b/$POD_ID/$name/completions.jsonl" --repo "$PUBLIC_REPO" --output "$SHARED/benchmarks-results/$name"
done
```

The screening record's `outcome` is `kept` (run 2's RL task list) or `floor` (fewer than 320 kept: report to the owner). Run 2 reads run 1b's weights, results, gate record and screening from `$SHARED` ([run 2's runbook](POSTTRAIN_RUN2_RUNBOOK.md)).

## 9. Results

From run 1b's verified `report.json` and summary: the probe's rates, each quarter's SFT dev loss and the selected quarter, the special rows' redraw, the format gate's checks (first-token probability, the close, the separation, the smoke) and the recorded benchmark smoke, ARC-Easy for the SFT weights, the cost and the timings. From the Mac: the dev results, the ledger's `withdrawn` and `sealed` events, the screening record and the benchmark artifacts. Record the measured numbers as measured results in [docs/posttrain-runs.md](docs/posttrain-runs.md), apart from its projections.
