# Pilot runbook: single A40 bounded training pilot

Operational sequence for the first paid Runpod pilot of the 135,285,504-parameter S model. Every command below exists in this repository at the audited commit; nothing here is a placeholder script. Commands marked **Mac** run on the owner's Mac inside the repository checkout; commands marked **Pod** run inside the rented Pod over SSH. Shell variables are set once in step 2 and reused.

Nothing in steps 1 to 4 costs money. Step 5 creates the Pod and starts billing; do not execute it without the owner's explicit launch authorization. Read [PILOT_LAUNCH_READINESS.md](PILOT_LAUNCH_READINESS.md) first: it lists the blocking conditions, including that Runpod offers no provider-enforced Pod lifetime.

## 1. Prelaunch checks (Mac, free)

```sh
git fetch origin && git checkout pilot-readiness-audit
make setup && make check && make smoke
uv run --locked python -m tools.gpu_pilot smoke --output runs/pilot-smoke
uv run --locked python -m tools.pilot_preflight --output runs/pilot-preflight-cpu --device cpu
uv run --locked python -m tools.cpu_resume_check --output runs/pilot-resume-check --size tiny --seeds 0 7 --steps 4
uv run --locked python -m tools.pilot_feasibility --specification configs/pilot-a40-s.json --output runs/pilot-feasibility.json
uv run --locked pytest tests/test_pilot_specification.py tests/test_gpu_pilot.py tests/test_runpod_watchdog.py tests/test_pod_self_destruct.py -q
```

Expected: `make check` green; the smoke report ends `Pilot completed`; the preflight report shows `"status": "completed"` on `cpu`; the feasibility Markdown prints the line "All GPU numbers are analytical or CPU-proxy estimates; no A40 measurement exists yet".

## 2. Credentials and environment (Mac, free)

```sh
set -a; source .env; set +a            # .env is git-ignored; it holds RUNPOD_API_KEY=... only
test -n "$RUNPOD_API_KEY" && echo "RUNPOD_API_KEY is set"   # never echo the value
gh auth status                          # needed for the private record/log archive
ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
export PILOT_SSH_KEY=~/.ssh/id_ed25519
export CTRL=runs/pilot-controller && mkdir -p "$CTRL"
export PRIVATE_REPO=OWNER/PRIVATE_RECORDS_REPOSITORY   # must be private; the supervisor verifies this
```

Requirements: the Runpod API key must have Pod read and terminate permissions (the console shows key scopes); the SSH public key must be registered under Runpod account settings so the Pod injects it; the working directory must not sit under a symlinked path (macOS `/tmp` is a symlink; `tools/pilot_data.py` rejects symlinked components).

## 3. Final Runpod quote verification (Mac, free)

Open the Runpod console, Secure Cloud, NVIDIA A40, on-demand, 1 GPU, container disk 20 GB, no volume. Record the displayed hourly price. Then:

```sh
uv run --locked python - <<'PY'
import json; spec = json.load(open("configs/pilot-a40-s.json"))
print("specification GPU $/h:", spec["quoted_gpu_hourly"], "total $/h:", spec["limits"]["quoted_total_hourly"])
PY
```

If the console price exceeds `quoted_gpu_hourly` (0.59), edit both `quoted_gpu_hourly` and `limits.quoted_total_hourly` in `configs/pilot-a40-s.json` (total = GPU price + 20 GB x $0.10 / 730 h, rounded up to the cent), commit the change, and rerun `tests/test_pilot_specification.py`. The spending ceiling of $5 and the 1.5 x estimate cost stop ($1.35) stay as they are. Confirm the account balance covers at least one hour at the quoted rate (Runpod refuses to deploy otherwise) and that no network volume exists on the account.

## 4. Dataset staging and checksum verification (Mac, free)

The reviewed binding was built earlier with `tools/pilot_data.py`; do not rebuild it. Verify it, run the token-level leakage audit, and pin its hash:

```sh
export BINDING_DIR=runs/pilot-data
uv run --locked python -m tools.pilot_data_audit --binding "$BINDING_DIR/binding.json" --output "$CTRL/leakage-audit.json"
export BINDING_SHA256=$(shasum -a 256 "$BINDING_DIR/binding.json" | cut -d' ' -f1) && echo "$BINDING_SHA256"
uv run --locked python - <<'PY'
import json; b = json.load(open("runs/pilot-data/binding.json"))
print("train tokens incl. BOS:", b["train"]["tokens_including_bos"], "docs:", b["train"]["documents"])
print("validation docs:", b["validation"]["documents"], "tokenizer:", b["tokenizer"]["identity"][:16])
PY
```

Expected: the audit exits 0 with `leaks: 0`; the validation document count equals `validation_documents` (32) in the specification, otherwise the run command refuses to start; the tokenizer identity is `509ab5b810dee7ae…` (the frozen 65,536-entry tokenizer). Record `BINDING_SHA256`; it is bound into the receipt.

## 5. Pod creation and independent timeout controls (Mac, BILLABLE)

Only after explicit launch authorization. Capture the billing origin before the create request so every deadline is conservative:

```sh
export ORIGIN=$(date +%s)
curl -sS -X POST https://api.runpod.io/v2/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  --data @configs/runpod-pod-a40-s.json | tee "$CTRL/create-response.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'
export POD_ID=<id printed above>
export SELF_DESTRUCT_DEADLINE=$((ORIGIN + 5400 - 180))      # 3 minutes inside the 90-minute wall
```

The request body is `configs/runpod-pod-a40-s.json` (A40, Secure Cloud, template `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, 20 GB container disk, TCP 22 exposed, no volume). The console path is equivalent: deploy with exactly those settings and read the Pod ID from the Pods page. Then capture the identity the watchdog will enforce and start the watchdog immediately:

```sh
uv run --locked python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" \
  --identity "$CTRL/pod-identity.json" --connection "$CTRL/pod-connection.json"
cat > "$CTRL/policy.json" <<EOF
{"max_wall_seconds": 5400, "max_cost_usd": 1.35, "max_idle_seconds": 600,
 "quoted_total_hourly": 0.60, "max_transfer_seconds": 1500, "idle_baseline_unix": $(date +%s)}
EOF
uv run --locked python -m tools.runpod_watchdog --identity "$CTRL/pod-identity.json" \
  --heartbeat "$CTRL/heartbeat.json" --receipt "$CTRL/watchdog-receipt.json" \
  --max-wall-seconds 5400 --max-idle-seconds 600 --max-cost-usd 1.35 --quoted-total-hourly 0.60 \
  --billing-started-at-unix "$ORIGIN" --idle-baseline-unix "$(date +%s)" --max-transfer-seconds 1500 \
  > "$CTRL/watchdog.log" 2>&1 &
```

Wait until `pod-connection.json` shows `"status": "RUNNING"` and a non-null `ssh_direct` (re-run the capture command until it does). If `ssh_proxy_only` is true the Pod has no public IP and file transfer is impossible: terminate it (step 14) and deploy again in a data center with public IP support. Pin the host key before any transfer:

```sh
export POD_HOST=$(python3 -c 'import json; print(json.load(open("'"$CTRL"'/pod-connection.json"))["ssh_direct"]["host"])')
export POD_PORT=$(python3 -c 'import json; print(json.load(open("'"$CTRL"'/pod-connection.json"))["ssh_direct"]["port"])')
ssh-keyscan -p "$POD_PORT" "$POD_HOST" > "$CTRL/known_hosts" 2>/dev/null
ssh-keygen -lf "$CTRL/known_hosts"      # compare with the fingerprint shown in the Pod's web terminal
alias podssh="ssh -F /dev/null -p $POD_PORT -i $PILOT_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$CTRL/known_hosts -o BatchMode=yes root@$POD_HOST"
alias podscp="scp -F /dev/null -P $POD_PORT -i $PILOT_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$CTRL/known_hosts -o BatchMode=yes"
```

## 6. Container startup and dependency checks (Pod, billable)

Stage the exact audited commit and the bound data, then arm the self-destruct timer first:

```sh
git archive --format=tar.gz -o "$CTRL/repo.tar.gz" HEAD
podssh 'mkdir -p /workspace/repo /workspace/pilot /workspace/pilot-data'
podscp "$CTRL/repo.tar.gz" root@$POD_HOST:/workspace/repo.tar.gz
podscp "$BINDING_DIR"/binding.json "$BINDING_DIR"/train.bin "$BINDING_DIR"/train.bin.json \
       "$BINDING_DIR"/validation.bin "$BINDING_DIR"/validation.bin.json "$BINDING_DIR"/tokenizer.json \
       root@$POD_HOST:/workspace/pilot-data/
podssh "cd /workspace/repo && tar xzf /workspace/repo.tar.gz && bash tools/pilot_pod_bootstrap.sh --deadline-unix $SELF_DESTRUCT_DEADLINE --workspace /workspace/pilot --repo /workspace/repo" | tee "$CTRL/bootstrap.log"
podscp root@$POD_HOST:/workspace/pilot/self-destruct.json "$CTRL/self-destruct.json"
podscp root@$POD_HOST:/workspace/pilot/preflight-a40/report.json "$CTRL/preflight-a40-report.json"
podscp root@$POD_HOST:/workspace/pilot/environment.json "$CTRL/environment.json"
```

The bootstrap exits non-zero and prints the reason if the timer does not arm, if `python3` is not 3.12, if `torch.cuda.is_available()` is false, if more than one GPU is visible, or if the A40 precision/resume preflight fails. It also runs the CPU reference tests against the image's torch 2.9.1 build. Check `self-destruct.json` shows `"state": "armed"` and the expected `deadline_unix`; check `preflight-a40-report.json` shows `"status": "completed"` and `"hardware_precision_and_resume_checked": true`.

Measure the real transfer path with a dummy 1.5 GiB file so the transfer window is known before training (Mac-initiated pull, exactly as the supervisor does it):

```sh
podssh 'mkdir -p /workspace/transfer-test && head -c 1623502465 /dev/urandom > /workspace/transfer-test/checkpoint.pt && echo "{}" > /workspace/transfer-test/record.json && : > /workspace/transfer-test/training.jsonl'
time uv run --locked python -m tools.pilot_transfer --host root@$POD_HOST --port $POD_PORT \
  --identity-file "$PILOT_SSH_KEY" --known-hosts "$CTRL/known_hosts" --remote-dir /workspace/transfer-test \
  --destination "$CTRL/transfer-test" --bundle-name dummy --checkpoint checkpoint.pt --record --log training.jsonl --timeout 1800
podssh 'rm -rf /workspace/transfer-test'
```

If the pull takes longer than 20 minutes, the 15-minute finalization reserve is insufficient: stop here, terminate the Pod (step 14) and revisit the reserve. Record the measured rate; it is required for the `ssh_transfer_tested` confirmation.

Build the readiness receipt from the real artifacts. The confirmations file is the owner's explicit statement; `termination_guarantee` must be `redundant_best_effort` because no provider-enforced lifetime exists:

```sh
cat > "$CTRL/confirmations.json" <<'EOF'
{"owner_confirms_four_runpod_steps": true, "cloud_data_pre_staged": true, "ssh_transfer_tested": true,
 "external_watchdog_armed": true, "hardware_preflight_passed": true, "full_console_quote_confirmed": true,
 "staging_storage_cost_bound": true, "termination_guarantee": "redundant_best_effort",
 "owner_accepts_best_effort_termination": true}
EOF
uv run --locked python -m tools.pilot_readiness_receipt --specification configs/pilot-a40-s.json \
  --binding "$BINDING_DIR/binding.json" --preflight "$CTRL/preflight-a40-report.json" \
  --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --self-destruct-receipt "$CTRL/self-destruct.json" --confirmations "$CTRL/confirmations.json" \
  --billing-origin-unix "$ORIGIN" --output "$CTRL/readiness-receipt.json"
podscp "$CTRL/readiness-receipt.json" "$CTRL/preflight-a40-report.json" root@$POD_HOST:/workspace/pilot/
```

The four Runpod steps confirmed by the first flag are: funded balance verified, SSH public key registered, API key with Pod permissions created, console quote recorded. The receipt tool refuses to write if any gate is false, if the timer is not armed, or if the receipt would already violate a limit.

## 7. GPU microbatch probes (Pod, billable)

Start the worker detached from the SSH session. Probes (16, 32, 8, then the 4-sequence fallback) run first, 20 optimizer updates each with two accumulated microbatches, inside a 300-second window counted from the first probe:

```sh
podssh "cd /workspace/repo && setsid nohup python3 -m tools.gpu_pilot run \
  --specification configs/pilot-a40-s.json --readiness /workspace/pilot/readiness-receipt.json \
  --preflight /workspace/pilot/preflight-a40/report.json --binding /workspace/pilot-data/binding.json \
  --binding-sha256 $BINDING_SHA256 --output /workspace/pilot-run > /workspace/pilot/run.log 2>&1 < /dev/null &"
```

Start the Mac-side supervisor in a second terminal right away; it mirrors the heartbeat, guards through the API, pulls every export and archives records:

```sh
uv run --locked python -m tools.pilot_supervisor --identity "$CTRL/pod-identity.json" --policy "$CTRL/policy.json" \
  --identity-file "$PILOT_SSH_KEY" --known-hosts "$CTRL/known_hosts" --output "$CTRL/supervisor" \
  --host root@$POD_HOST --remote-root /workspace/pilot-run --repository "$PRIVATE_REPO" --port $POD_PORT \
  --transfer-timeout 1800 --poll-seconds 15 2>&1 | tee "$CTRL/supervisor.log"
```

Watch the probe table: `podssh 'python3 -c "import json; r=json.load(open(\"/workspace/pilot-run/report.json\")); print(r[\"probes\"], r.get(\"chosen_microbatch\"), r.get(\"accumulation_steps\"))"'`. Each row records status (`completed`, `oom`, `incomplete`, `skipped`), updates, tokens per second and peak allocated/reserved bytes. The worker picks the largest completed size whose reserved peak stays below 90% of device memory. `no measured microbatch fits` ends the pilot with status `stopped` and no training charge beyond the probes.

## 8. Learning-rate screens (Pod, automatic)

Three arms (0.0003, 0.0006, 0.0012) each run 100 updates of 131,072 tokens from identical initial weights, identical seeded data order and the same 50-step warmup plus cosine schedule over 500 steps. Periodic exports appear every 300 seconds under `/workspace/pilot-run/exports/lr-N-stepS-K`; the supervisor pulls each one. Monitor:

```sh
podssh 'tail -n 3 /workspace/pilot-run/updates.jsonl; cat /workspace/pilot-run/heartbeat.json'
```

Any screen that stops early (deadline, cost, idle, rising loss, throughput guard) yields status `stopped` with `incomplete_learning_rate_screen` or the stop reason; no winner is declared.

## 9. Winning-arm selection (Pod, automatic)

After all three arms complete, each is validated on the same 32 fixed held-out windows recorded as `validation_identity`; the lowest validation loss wins, ties broken by the lower learning rate. `report.json` shows `"winner"` and the `arms` table with each arm's validation loss, checkpoint SHA-256 and training tokens.

## 10. Winning-arm continuation (Pod, automatic)

The worker verifies the winner's step-100 checkpoint against its SHA-256 sidecar, resumes it into a run labelled `lr-N-continued`, and trains toward step 500 until the earlier of 3,600 seconds of training or the Pod wall minus the 900-second finalization reserve. The step-100 checkpoint and export are kept. Training stops for `training_time_limit` or `finalization_reserve` still allow the final checkpoint; `pod_wall_limit`, `cost_limit` and `idle_gpu_limit` do not.

## 11. Checkpoint save/restore verification (Pod, automatic; Mac, manual)

On the Pod, the final checkpoint is saved, then restored into a fresh trainer whose weights, optimizer moments, loader state, RNG and held-out outputs must match exactly (`last_checkpoint_resume.tested: true` in the report). On the Mac, once the supervisor has promoted the final bundle:

```sh
ls "$CTRL/supervisor/received/"
uv run --locked python - <<'PY'
import json, glob, torch
bundle = sorted(glob.glob("runs/pilot-controller/supervisor/received/lr-*-continued-step*"))[-1]
state = torch.load(f"{bundle}/checkpoint.pt", map_location="cpu", weights_only=True)
record = json.load(open(f"{bundle}/record.json"))
print(bundle, "step", state["step"], "run", state["run_id"], "status", record["status"], "tokens", record["training_tokens_processed"])
PY
```

## 12. Artifact transfer to the Mac

The supervisor pulls each export with a remote size/SHA-256 manifest, stages it, verifies every byte and promotes the bundle with `transfer-receipt.json`. If the supervisor is down, pull manually:

```sh
uv run --locked python -m tools.pilot_transfer --host root@$POD_HOST --port $POD_PORT \
  --identity-file "$PILOT_SSH_KEY" --known-hosts "$CTRL/known_hosts" \
  --remote-dir /workspace/pilot-run/exports/<bundle-name> --destination "$CTRL/manual" \
  --bundle-name <bundle-name> --checkpoint checkpoint.pt --record --log training.jsonl --timeout 1800
```

The Mac copy plus its receipt is the only durable checkpoint. The watchdog allows up to 1,500 seconds after the last GPU step while the Mac observes staged bytes growing; the 90-minute wall still applies. If the Mac goes offline the Pod is terminated at the wall or idle limit and the checkpoint is lost; the small `record.json` and `training.jsonl` from the last pulled export remain on the Mac.

## 13. GitHub record and log backup

The supervisor uploads only `record.json` and `training.jsonl` of each verified bundle to `pilot-runs/<pod-id>/<bundle>/` in `$PRIVATE_REPO`, confirms the repository is private, reads both files back and compares bytes (`github-receipt.json`). Checkpoints never go to GitHub. After shutdown it writes a `<bundle>-final` record containing the watchdog and transfer receipts and archives that too.

## 14. Pod termination

The supervisor marks `shutdown_ready` once the final bundle is on the Mac and archived, and the API guard then deletes the Pod and confirms a 404. The in-Pod timer deletes the Pod at `SELF_DESTRUCT_DEADLINE` regardless. Manual termination and confirmation:

```sh
curl -sS -o /dev/null -w '%{http_code}\n' -X DELETE "https://api.runpod.io/v2/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"   # expect 204
uv run --locked python -m tools.runpod_pod_identity status --pod-id "$POD_ID"    # expect "resource not found (HTTP 404)"
```

Do not terminate while the only checkpoint copy is still on the Pod unless a limit forces it; in that case report the lost checkpoint rather than pretending success.

## 15. Network-volume cleanup

This pilot creates no network volume and no volume disk. Verify in the console under Storage that the account holds no network volume; if an unexpected volume exists, copy anything needed first, then delete it, because it bills hourly after Pod termination.

## 16. Final billing verification

```sh
uv run --locked python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --output "$CTRL/billing.json"
uv run --locked python - <<'PY'
import glob, json
bundle = sorted(glob.glob("runs/pilot-controller/supervisor/received/*/record.json"))[-1]
record = json.load(open(bundle))
print("quoted accrued USD", record["quoted_accrued_cost_usd"], "pod seconds", record["pod_elapsed_seconds"])
print("provider totals", json.load(open("runs/pilot-controller/billing.json"))["totals"])
PY
```

Compare the provider totals with the quoted accrued cost in the record. Re-fetch billing after an hour; Runpod bills every five minutes and records may lag. Confirm the total stays below the $5 ceiling and note any disk charge.

## 17. Results analysis

From the final `record.json`: the probe table (tokens per second, peak allocated/reserved bytes per microbatch), `chosen_microbatch` and `accumulation_steps`, the three arms' validation losses and the winner, `continuation` (resumed step and checkpoint), `selected_step`, `training_tokens_processed`, `training_throughput`, `quoted_gpu_cost_per_billion_tokens`, `training_peak`, `data_exposure`, `leakage_audit`, `validation_identity` and `last_checkpoint_resume`. Loss curves per arm are in `training.jsonl` (one JSON row per update with loss, learning rate, gradient norm, tokens and step seconds). Re-run the feasibility calculator with the measured throughput and probe peaks to update the plan:

```sh
uv run --locked python -m tools.pilot_feasibility --specification configs/pilot-a40-s.json --throughputs <measured tokens/s>
```

Record the measured numbers in `docs/gpu-pilot-plan.md` and `docs/status.md` as measured A40 results, distinct from the analytical estimates that stand there today.
