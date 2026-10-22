# Bounded training pilot

This walkthrough describes the local tools for preparing, checking, running and retrieving a small training pilot. The worker runs on a machine that already exists; it does not create or rent one. A CPU test is useful for checking the code path, but it does not measure GPU speed or establish GPU precision, memory, or launch readiness. The Mac-side supervisor runs separately to mirror heartbeats, guard the Pod through the API, and retrieve verified artifacts.

## Check the local path

Start with the synthetic CPU smoke and the tiny precision/resume preflight:

```sh
python -m tools.gpu_pilot smoke --output runs/pilot-smoke
python -m tools.pilot_preflight --output runs/pilot-preflight-cpu --device cpu
```

The preflight compares FP32 and BF16 using the same initial weights, then checks an exact four-step interrupted/resumed run with gradient accumulation. It uses authored token IDs only. A `cuda:0` run must verify exactly one visible A40 through PyTorch and `nvidia-smi`; it checks the exact GPU identity, memory, Ampere capability, and BF16 support. The environment must have a CUDA-enabled PyTorch installation. The repository's locked Linux environment uses CPU PyTorch. This hardware preflight is required for a pilot but does not establish full launch readiness.

## Bind data before starting a worker

The data tool reads a local, reviewed JSONL source and an existing frozen tokenizer. It verifies both files against supplied SHA-256 identities, selects the reviewed training and held-out rows, builds shards locally, and writes `binding.json` containing the selected identities and output hashes. It does not download data or fit a tokenizer.

```sh
python -m tools.pilot_data \
  --source /path/to/reviewed.jsonl \
  --source-sha256 REVIEWED_SOURCE_SHA256 \
  --tokenizer /path/to/tokenizer.json \
  --tokenizer-identity FROZEN_TOKENIZER_IDENTITY \
  --tokenizer-file-sha256 TOKENIZER_FILE_SHA256 \
  --output runs/pilot-data
```

Review and preserve the resulting binding. Copy the already-built data to the worker before training; do not use rented GPU time to download or tokenize it. The worker verifies the binding hash and shard identities before model allocation.

Distinct document IDs do not prove distinct content, so audit the bound shards at the token level as well: `python -m tools.pilot_data_audit --binding runs/pilot-data/binding.json --output runs/pilot-data/leakage-audit.json` re-verifies the binding, then checks whether any validation document is an exact token-for-token repeat of a training document and whether its validation window (the first `--sequence-length` tokens after BOS, 1024 by default, matching the training-loop validation windows) appears verbatim anywhere in the training token stream, ignoring document boundaries. It also counts exact duplicate training documents and writes the report whether or not a leak is found; a leak exits with status 1, and `verify_binding` raises for the same condition. This is exact containment only: it does not detect near-duplicates, paraphrases, partial n-gram overlap, or text that tokenizes differently in another context, and it says nothing about validation content beyond the audited window.

## Arm the Pod before any paid work

Runpod has no provider-enforced Pod lifetime. The strongest available
termination is layered and best-effort: an in-Pod self-destruct timer, the
Mac-side API watchdog, and the worker's own deadlines. The bootstrap script
arms the timer first and refuses to continue if it does not arm:

```sh
bash tools/pilot_pod_bootstrap.sh --deadline-unix "$SELF_DESTRUCT_DEADLINE_UNIX" \
  --workspace /workspace/pilot --repo /workspace/repo
```

`tools/pod_self_destruct.py` is standard-library only, runs detached from the
SSH session, uses the Pod-scoped `RUNPOD_API_KEY` that Runpod injects, fires
on whichever of the wall clock or monotonic clock reaches the deadline first,
retries the delete until the API reports the Pod absent, and writes a
secret-free receipt. It cannot fire during a Runpod control-plane outage, and
the Pod-scoped key's permission to delete its own Pod must be verified on the
real Pod; the readiness receipt therefore records `termination_guarantee` as
`redundant_best_effort` and requires explicit owner acceptance of that risk.

Capture the Pod identity for the watchdog from the live API instead of typing
it: `python -m tools.runpod_pod_identity capture --pod-id "$POD_ID" --identity
pod-identity.json --connection pod-connection.json`. The connection file holds
the direct SSH host/port and never the Pod's environment block.

## Probe and train

The worker first probes candidate microbatches. Probe updates measure whether a size fits and how it runs; their weights and loader progress are discarded. Each probe runs with two accumulated microbatches so the fp32 gradients are resident during a forward pass, which the accumulated training phases also require; a probe is selected only if its reserved peak stays inside the specification's memory headroom fraction. The probe window counts from the first probe, not from Pod creation: when it expires, the running probe is recorded as `incomplete`, the remaining sizes as `skipped`, and the pilot continues with the largest completed probe. The worker then uses gradient accumulation to hold the configured effective token batch fixed throughout the pilot. If a stop arrives partway through an accumulated update, the trainer restores the loader and RNG state and drops partial gradients, so no half update is checkpointed. A failed or incomplete screen does not produce a winner.

Each learning-rate arm uses the same warmup and decay schedule over the configured total updates. The worker compares arms on the same fixed held-out windows (their count, target tokens and SHA-256 are recorded as `validation_identity`) and continues only the arm with the lowest validation loss, using the specified tie rule. Do not choose a winner from unequal or incomplete arms. The continuation resumes from the winner's screen checkpoint after verifying its recorded SHA-256 and saves under `<arm>-continued`, so the step-100 checkpoint and its export survive as the fair-comparison artifact. The speed reference is either an explicit prelaunch estimate or the selected batch's measured probe rate, recorded before accumulated training starts; the throughput guard uses that fixed reference.

Training stops at the earlier of the training budget and the Pod wall minus `finalization_reserve_seconds`; both permit the final checkpoint, a fresh-trainer resume test and the export. Wall, cost and idle limits never permit extra work. Every checkpoint gets a `last.pt.sha256.json` sidecar, and any later load compares the bytes against it. The report records training tokens processed, the deadline actually used and the data exposure (tokens per arm, epochs for the winner).

The run command requires a reviewed specification (`configs/pilot-a40-s.json` is the checked-in A40 specification), readiness receipt, exact data-binding SHA-256, and the completed A40 preflight report. The receipt binds the specification, data binding, and preflight report by SHA-256; it also records the exact provider Pod identity and creation time, conservative billing origin captured before the create request, watchdog policy, the armed self-destruct deadline and the termination guarantee classification. The self-destruct deadline must be within the 5,400-second wall-time ceiling. Prepare the receipt only after the four owner run confirmations, machine and data staging, transfer path, precision/resume preflight, independent watchdog, and self-destruct timer have been checked. These gates do not create or authorize a Pod.

```sh
python -m tools.gpu_pilot run \
  --specification configs/pilot-a40-s.json \
  --readiness PATH_TO_READINESS_RECEIPT.json \
  --preflight runs/pilot-preflight-a40/report.json \
  --binding runs/pilot-data/binding.json \
  --binding-sha256 VERIFIED_BINDING_SHA256 \
  --output runs/pilot-run
```

The worker records losses, learning rate, gradient norm, token counts, step duration and checkpoints. It publishes export snapshots containing `checkpoint.pt`, `record.json`, and `training.jsonl`. A later snapshot may replace an earlier snapshot for the same save label, so retrieve the appropriate completed export promptly. This record is evidence about one bounded run, not a benchmark or a capability claim.

## Watch and stop

Run the external watchdog independently from the worker, with a verified Pod identity, the worker heartbeat, the approved limits, and an API key in the process environment:

```sh
python -m tools.runpod_watchdog \
  --identity pod-identity.json \
  --heartbeat runs/pilot-run/heartbeat.json \
  --receipt watchdog-receipt.json \
  --max-wall-seconds "$APPROVED_WALL_SECONDS" \
  --max-idle-seconds "$APPROVED_IDLE_SECONDS" \
  --max-cost-usd "$APPROVED_COST_LIMIT" \
  --quoted-total-hourly "$VERIFIED_TOTAL_HOURLY_RATE" \
  --max-transfer-seconds 1500
```

The worker's timer, the in-Pod self-destruct and the API watchdog are safeguards, not a provider-enforced deadline. The watchdog reports termination only after the delete request returns HTTP 204 and a later status request returns HTTP 404; transport errors, other codes, or a still-visible machine leave shutdown unconfirmed. A 404 before the captured identity has ever been observed alive is reported as unknown, never as a completed shutdown, so a wrong Pod ID or key scope cannot masquerade as termination; the watchdog exits with status 3 if it never observes the Pod. Polling defaults to 15 seconds because the REST API is rate limited per minute. During `awaiting_transfer`, Mac-observed transfer progress (growing staged bytes) may extend the idle limit up to `--max-transfer-seconds`, bounded by the wall and cost limits; the Pod can never extend its own lease. Pod deletion can also leave separately billed network volumes; check those resources and their billing state separately. Stop at the approved wall-time or cost limit even if work is incomplete.

For an authorized run, the Mac-side supervisor is the controller. It keeps independent API-guard and SSH-heartbeat-mirror threads active while it copies checkpoints. A failed transfer is retried from the immutable export and only complete size/hash-verified bundles are promoted; incomplete staging is not treated as a checkpoint. It preserves failure diagnostics even when no checkpoint was created. The confirmed limits take precedence over transfer completion; copying does not extend a Pod lease beyond the transfer window. A stopped or interrupted copy may leave only diagnostic evidence, so do not interpret a missing checkpoint as a successful run.

```sh
python -m tools.pilot_supervisor \
  --identity pod-identity.json \
  --policy policy.json \
  --identity-file PATH_TO_SSH_IDENTITY \
  --known-hosts PATH_TO_VERIFIED_KNOWN_HOSTS \
  --output runs/controller \
  --host root@PUBLIC_HOST \
  --remote-root /path/to/pilot-run \
  --repository OWNER/PRIVATE_REPOSITORY \
  --port 22 \
  --transfer-timeout 1800
```

The supervisor archives only the verified `record.json` and `training.jsonl` to a repository it confirms is private, then downloads them back and checks exact bytes. It does not upload checkpoints to GitHub. Preserve the Mac-side checkpoint and transfer receipt as the durable model copy. After termination, `python -m tools.runpod_pod_identity billing --pod-id "$POD_ID" --output billing.json` records the provider's own billing for the Pod.

## Retrieve and verify

Use the prepared export directory and a verified SSH host key. The transfer utility requires an explicit identity file and port, uses strict known-host checking and batch mode, and accepts only selected checkpoint, record and approved log names. It needs the Pod's direct public-IP SSH port: Runpod's `ssh.runpod.io` proxy does not carry file transfers. It fetches a remote size/SHA-256 manifest, streams files into a staging directory, checks every size and hash, and atomically publishes the complete bundle with `transfer-receipt.json`. A failed or partial transfer is not promoted.

```sh
python -m tools.pilot_transfer \
  --host SSH_USER_AND_PUBLIC_HOST \
  --port 22 \
  --identity-file PATH_TO_SSH_IDENTITY \
  --known-hosts PATH_TO_VERIFIED_KNOWN_HOSTS \
  --remote-dir PATH_TO_PILOT_EXPORT \
  --destination runs/pilot-received \
  --bundle-name completed-step \
  --checkpoint checkpoint.pt \
  --record \
  --log training.jsonl
```

Check the receipt and load the copied checkpoint with the project's checkpoint-validation path before marking artifact transfer complete. Keep the destination machine available until the local copy and hashes are verified. Transfer receipts are required evidence for shutdown; they do not replace checks for residual provider resources.
