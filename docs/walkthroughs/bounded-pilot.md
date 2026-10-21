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

## Probe and train

The worker first probes candidate microbatches. Probe updates measure whether a size fits and how it runs; their weights and loader progress are discarded. The worker then selects the largest feasible microbatch and uses gradient accumulation to hold the configured effective token batch fixed throughout the pilot. If a stop arrives partway through an accumulated update, the trainer restores the loader and RNG state and drops partial gradients, so no half update is checkpointed. A failed or incomplete screen does not produce a winner.

Each learning-rate arm uses the same warmup and decay schedule over the configured total updates. The worker compares arms on the same fixed held-out windows and continues only the arm with the lowest validation loss, using the specified tie rule. Do not choose a winner from unequal or incomplete arms. The speed reference is either an explicit prelaunch estimate or the selected batch's measured probe rate, recorded before accumulated training starts; the throughput guard uses that fixed reference.

The run command requires a reviewed specification, readiness receipt, exact data-binding SHA-256, and the completed A40 preflight report. The receipt binds the specification, data binding, and preflight report by SHA-256; it also records the exact provider Pod identity and creation time, conservative billing origin captured before the create request, watchdog policy, and provider termination deadline. The provider deadline must be within the 5,400-second wall-time ceiling. Prepare the receipt only after the four owner run confirmations, machine and data staging, transfer path, precision/resume preflight, independent watchdog, and provider deadline controls have been checked. These gates do not create or authorize a Pod.

```sh
python -m tools.gpu_pilot run \
  --specification PATH_TO_SPECIFICATION.json \
  --readiness PATH_TO_READINESS_RECEIPT.json \
  --preflight runs/pilot-preflight-a40/report.json \
  --binding runs/pilot-data/binding.json \
  --binding-sha256 VERIFIED_BINDING_SHA256 \
  --output runs/pilot-run
```

The worker records losses, learning rate, gradient norm, token counts, step duration and checkpoints. It publishes export snapshots containing `checkpoint.pt`, `record.json`, and `training.jsonl`. A later snapshot may replace an earlier snapshot for the same training phase, so retrieve the appropriate completed export promptly. This record is evidence about one bounded run, not a benchmark or a capability claim.

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
  --quoted-total-hourly "$VERIFIED_TOTAL_HOURLY_RATE"
```

The worker's timer and an API watchdog are safeguards, not a provider-enforced deadline. The watchdog reports termination only after the delete request returns HTTP 204 and a later status request returns HTTP 404; transport errors, other codes, or a still-visible machine leave shutdown unconfirmed. Provider-enforced strict TTL support remains unresolved for this workflow, so do not start until a hard deadline has been independently confirmed. Pod deletion can also leave separately billed network volumes; check those resources and their billing state separately. Stop at the approved wall-time or cost limit even if work is incomplete.

For an authorized run, the Mac-side supervisor is the controller. It keeps independent API-guard and SSH-heartbeat-mirror threads active while it copies checkpoints. A failed transfer is retried from the immutable export and only complete size/hash-verified bundles are promoted; incomplete staging is not treated as a checkpoint. It preserves failure diagnostics even when no checkpoint was created. The 600-second idle stop and confirmed provider hard deadline take precedence over transfer completion; copying does not extend a Pod lease. A stopped or interrupted copy may leave only diagnostic evidence, so do not interpret a missing checkpoint as a successful run.

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
  --port 22
```

The supervisor archives only the verified `record.json` and `training.jsonl` to a repository it confirms is private, then downloads them back and checks exact bytes. It does not upload checkpoints to GitHub. Preserve the Mac-side checkpoint and transfer receipt as the durable model copy.

## Retrieve and verify

Use the prepared export directory and a verified SSH host key. The transfer utility requires an explicit identity file and port, uses strict known-host checking and batch mode, and accepts only selected checkpoint, record and approved log names. It fetches a remote size/SHA-256 manifest, streams files into a staging directory, checks every size and hash, and atomically publishes the complete bundle with `transfer-receipt.json`. A failed or partial transfer is not promoted.

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
