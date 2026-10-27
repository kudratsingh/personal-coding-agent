#!/usr/bin/env bash
# Bootstrap the rented Pod for the bounded pilot. Runs INSIDE the Pod over SSH.
#
# Order matters: the in-pod self-destruct timer is armed before anything else,
# so a later failure (install, preflight, SSH loss) cannot leave an unguarded
# Pod. This script never prints RUNPOD_API_KEY and never creates resources.
#
# Usage:
#   bash tools/pilot_pod_bootstrap.sh --deadline-unix <unix-seconds> \
#        --workspace /workspace/pilot [--repo /workspace/repo] [--skip-tests]
#
# Rehearsal mode (D-35; one Pod per run, see REHEARSAL_RUNBOOK.md) arms the timer
# with the 60,000-second lifetime ceiling first, then checks the shipped code archive
# against the Mac's revision.json and copies that record into the workspace (the worker
# refuses to start without it), loads the Pod's R2 key from a 0600 file the operator
# delivered over SSH stdin (never echoed), measures an R2 round trip (its upload rate is
# a readiness gate), fetches and re-verifies the run's pinned data and the shared
# initialization, runs the CPU tests including the worker's (none needs Docker or the
# network), and runs the A40 preflight with its compile check (--compile-check):
#   bash tools/pilot_pod_bootstrap.sh --mode rehearsal --run A|B --deadline-unix <unix> \
#        --workspace /workspace/rehearsal --repo /workspace/repo \
#        --revision /workspace/revision.json --archive /workspace/repo.tar.gz \
#        --r2-env /workspace/rehearsal/r2-pod.env --binding /workspace/rehearsal/binding.json \
#        --binding-sha256 <hex> --data-root /workspace/rehearsal-data \
#        --initialization-key <R2 key> --initialization-sha256 <hex> \
#        --initialization-size <bytes> [--skip-tests]
#
# Post-training mode (one Pod per run, see POSTTRAIN_RUNBOOK.md) arms the timer with the
# run's ceiling-derived lifetime (--lifetime-seconds, from `tools.posttrain_ops limits`; run 1:
# 18000) and the in-Pod idle self-destruct (--idle-seconds, 900, watching the worker's
# --heartbeat and the bootstrap's own setup-progress.json, which every step below stamps),
# then the same revision, key and environment steps, an R2 round trip under the run's
# posttrain-runs/ prefix, the pinned fetch and re-verification of every bound input
# (tools.posttrain_binding), the CPU tests and the A40 preflight with its compile check:
#   bash tools/pilot_pod_bootstrap.sh --mode posttrain --run 1 --deadline-unix <unix> \
#        --lifetime-seconds 18000 --idle-seconds 900 --workspace /workspace/posttrain \
#        --repo /workspace/repo --revision /workspace/revision.json \
#        --archive /workspace/repo.tar.gz --r2-env /workspace/posttrain/r2-pod.env \
#        --binding /workspace/posttrain/binding.json --binding-sha256 <hex> \
#        --data-root /workspace/posttrain-inputs \
#        --heartbeat /workspace/posttrain-run/heartbeat.json [--skip-tests]
#
# An existing self-destruct receipt counts as armed only while its timer lives: its pid
# answers kill -0 and it rewrote checked_at_unix within the last 30 s (the timer rewrites
# it every 5 s). Any other receipt is set aside and the timer is armed again.
set -euo pipefail

MODE="pilot"
DEADLINE=""
WORKSPACE=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TESTS=1
RUN=""
R2_ENV=""
BINDING=""
BINDING_SHA256=""
DATA_ROOT=""
INIT_KEY=""
INIT_SHA256=""
INIT_SIZE=""
REVISION=""
ARCHIVE=""
LIFETIME=""
IDLE=""
HEARTBEAT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --deadline-unix) DEADLINE="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    --run) RUN="$2"; shift 2 ;;
    --r2-env) R2_ENV="$2"; shift 2 ;;
    --binding) BINDING="$2"; shift 2 ;;
    --binding-sha256) BINDING_SHA256="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --initialization-key) INIT_KEY="$2"; shift 2 ;;
    --initialization-sha256) INIT_SHA256="$2"; shift 2 ;;
    --initialization-size) INIT_SIZE="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --lifetime-seconds) LIFETIME="$2"; shift 2 ;;
    --idle-seconds) IDLE="$2"; shift 2 ;;
    --heartbeat) HEARTBEAT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$DEADLINE" || -z "$WORKSPACE" ]]; then
  echo "usage: $0 --deadline-unix <unix> --workspace <dir> [--repo <dir>] [--skip-tests]" >&2
  exit 2
fi
REHEARSAL_ARGS="$RUN$R2_ENV$BINDING$BINDING_SHA256$DATA_ROOT$INIT_KEY$INIT_SHA256$INIT_SIZE"
REHEARSAL_ARGS="$REHEARSAL_ARGS$REVISION$ARCHIVE"
POSTTRAIN_ARGS="$LIFETIME$IDLE$HEARTBEAT"
case "$MODE" in
  pilot)
    if [[ -n "$REHEARSAL_ARGS$POSTTRAIN_ARGS" ]]; then
      echo "rehearsal arguments need --mode rehearsal (or --mode posttrain)" >&2
      exit 2
    fi
    ;;
  rehearsal)
    if [[ -n "$POSTTRAIN_ARGS" ]]; then
      echo "--lifetime-seconds, --idle-seconds and --heartbeat need --mode posttrain" >&2
      exit 2
    fi
    if [[ ! "$RUN" =~ ^[AB]$ || -z "$R2_ENV" || -z "$BINDING" || -z "$DATA_ROOT" \
          || -z "$INIT_KEY" || ! "$BINDING_SHA256" =~ ^[0-9a-f]{64}$ \
          || ! "$INIT_SHA256" =~ ^[0-9a-f]{64}$ || ! "$INIT_SIZE" =~ ^[1-9][0-9]*$ \
          || -z "$REVISION" || -z "$ARCHIVE" ]]; then
      echo "rehearsal mode needs --run A|B, --revision, --archive, --r2-env, --binding," \
        "--binding-sha256 <hex>, --data-root, --initialization-key," \
        "--initialization-sha256 <hex> and --initialization-size <bytes>" >&2
      exit 2
    fi
    ;;
  posttrain)
    if [[ ! "$RUN" =~ ^[1-9]$ || -z "$R2_ENV" || -z "$BINDING" || -z "$DATA_ROOT" \
          || ! "$BINDING_SHA256" =~ ^[0-9a-f]{64}$ || -z "$REVISION" || -z "$ARCHIVE" \
          || ! "$LIFETIME" =~ ^[1-9][0-9]*$ || ! "$IDLE" =~ ^[1-9][0-9]*$ \
          || -z "$HEARTBEAT" || -n "$INIT_KEY$INIT_SHA256$INIT_SIZE" ]]; then
      echo "posttrain mode needs --run <n>, --revision, --archive, --r2-env, --binding," \
        "--binding-sha256 <hex>, --data-root, --lifetime-seconds <s>, --idle-seconds <s>" \
        "and --heartbeat, and no initialization" >&2
      exit 2
    fi
    if (( LIFETIME > 60000 )); then
      echo "--lifetime-seconds is above the in-Pod timer's absolute 60000 s cap" >&2
      exit 2
    fi
    ;;
  *) echo "--mode must be pilot, rehearsal or posttrain" >&2; exit 2 ;;
esac
# A non-interactive SSH command does not inherit the container's injected variables;
# the Runpod image records them in /etc/rp_environment (observed 2026-10-24).
if [[ ( -z "${RUNPOD_POD_ID:-}" || -z "${RUNPOD_API_KEY:-}" ) && -r /etc/rp_environment ]]; then
  set -a; . /etc/rp_environment; set +a
fi
# The image's Ubuntu 24.04 system Python may be marked externally managed.
export PIP_BREAK_SYSTEM_PACKAGES=1
if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  echo "RUNPOD_POD_ID is not set; this script must run inside the Pod" >&2
  exit 2
fi
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY (Pod-scoped key) is not set; refusing to continue unguarded" >&2
  exit 2
fi
mkdir -p "$WORKSPACE"
cd "$REPO"

# 1. Arm the self-destruct timer first, detached from this SSH session.
RECEIPT="$WORKSPACE/self-destruct.json"
PROGRESS="$WORKSPACE/setup-progress.json"
LIFETIME_ARGS=()
if [[ "$MODE" == "rehearsal" ]]; then
  LIFETIME_ARGS=(--max-lifetime-seconds 60000)  # The D-35 ceiling; the pilot keeps 5400.
elif [[ "$MODE" == "posttrain" ]]; then
  # The run's own ceiling-derived lifetime, and the idle rule over the worker's heartbeat and
  # this bootstrap's progress stamps.
  LIFETIME_ARGS=(--max-lifetime-seconds "$LIFETIME" --idle-seconds "$IDLE"
    --activity "$HEARTBEAT" --activity "$PROGRESS")
fi
# Post-training only: each step below records progress, so the idle timer runs only while
# nothing happens. The stamp is standard-library Python, so it works before the install.
stamp() {
  if [[ "$MODE" == "posttrain" ]]; then
    python3 "$REPO/tools/pod_progress.py" --file "$PROGRESS" --source "bootstrap:$1" > /dev/null
  fi
}
FRESH_SECONDS=30
# Succeeds only for a receipt whose timer still runs: a live state, a pid that answers
# kill -0 and a checked_at_unix rewritten within FRESH_SECONDS. A receipt alone is not a timer.
timer_alive() {
  local pid
  [[ -f "$RECEIPT" ]] || return 1
  pid=$(python3 -c "
import json, sys, time
try:
    value = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(1)
pid, checked = value.get('pid'), value.get('checked_at_unix')
age = time.time() - checked if type(checked) in (int, float) else None
live = value.get('state') in ('armed', 'retrying', 'terminating')
if not (live and age is not None and -5 <= age <= float(sys.argv[2])):
    sys.exit(1)
if type(pid) is not int or pid <= 0:
    sys.exit(1)
print(pid)
" "$RECEIPT" "$FRESH_SECONDS") || return 1
  kill -0 "$pid" 2>/dev/null
}
if timer_alive; then
  echo "self-destruct already armed and alive: $RECEIPT"
else
  if [[ -e "$RECEIPT" ]]; then
    STALE="$RECEIPT.stale-$(date +%s)"
    mv "$RECEIPT" "$STALE"
    echo "set aside a self-destruct receipt with no live timer: $STALE"
  fi
  setsid nohup python3 "$REPO/tools/pod_self_destruct.py" \
    --pod-id "$RUNPOD_POD_ID" \
    --deadline-unix "$DEADLINE" \
    --receipt "$RECEIPT" \
    --stop-file "$WORKSPACE/TERMINATE_NOW" \
    ${LIFETIME_ARGS[@]+"${LIFETIME_ARGS[@]}"} \
    > "$WORKSPACE/self-destruct.log" 2>&1 < /dev/null &
  ARMED="import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['state']=='armed' else 1)"
  for _ in $(seq 1 20); do
    if timer_alive && python3 -c "$ARMED" "$RECEIPT"; then
      break
    fi
    sleep 0.5
  done
  if ! timer_alive || ! python3 -c "$ARMED" "$RECEIPT" 2>/dev/null; then
    echo "self-destruct timer failed to arm; see $WORKSPACE/self-destruct.log" >&2
    exit 3
  fi
  echo "self-destruct armed for unix deadline $DEADLINE (receipt: $RECEIPT)"
fi
stamp armed

# 1a. Rehearsal: the shipped archive must be the one the Mac's revision.json names (a
#     40-hex commit from a clean tree and the archive's SHA-256). The record is copied
#     byte for byte to $WORKSPACE/revision.json, where the worker compares it with its
#     readiness receipt; a different record already there stops the bootstrap.
if [[ "$MODE" != "pilot" ]]; then
  if ! python3 - "$REVISION" "$ARCHIVE" "$WORKSPACE/revision.json" <<'PY'
import hashlib, json, os, re, sys
source, archive, target = sys.argv[1:4]
data = open(source, "rb").read()
value = json.loads(data)
if not isinstance(value, dict) or set(value) != {"git_commit", "clean", "archive_sha256"}:
    sys.exit("revision.json must hold exactly git_commit, clean and archive_sha256")
commit = value["git_commit"]
if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
    sys.exit("revision.json's git_commit must be 40 lowercase hex digits")
if value["clean"] is not True:
    sys.exit("revision.json records a dirty tree")
digest = hashlib.sha256()
with open(archive, "rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
if digest.hexdigest() != value["archive_sha256"]:
    sys.exit("the shipped archive's SHA-256 differs from revision.json's archive_sha256")
if os.path.lexists(target):
    if os.path.islink(target) or open(target, "rb").read() != data:
        sys.exit(f"{target} already holds another revision")
else:
    with open(target + ".part", "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(target + ".part", target)
print(f"code revision {commit} (archive SHA-256 verified): {target}")
PY
  then
    echo "the code revision does not match the shipped archive; see the message above" >&2
    exit 6
  fi
  stamp revision
fi

# 1b. Rehearsal: load the Pod's own R2 key. The file arrived over SSH stdin; it must be
#     a regular 0600 file of this user. Its values are never printed; only names are.
if [[ "$MODE" != "pilot" ]]; then
  if ! python3 - "$R2_ENV" <<'PY'
import os, stat, sys
info = os.lstat(sys.argv[1])
ok = stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
sys.exit(0 if ok and info.st_uid == os.getuid() else 1)
PY
  then
    echo "the R2 environment file must be a regular 0600 file owned by this user" >&2
    exit 5
  fi
  set +x
  set -a; . "$R2_ENV"; set +a
  for name in R2_ENDPOINT R2_BUCKET R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY; do
    if [[ -z "${!name:-}" ]]; then
      echo "the R2 environment file does not set $name" >&2
      exit 5
    fi
  done
  for name in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
    if [[ -n "${!name:-}" ]]; then
      echo "$name is set on the Pod; only the Pod's own R2_POD_ key may be here" >&2
      exit 5
    fi
  done
  echo "R2 environment loaded for run $RUN (names only): R2_ENDPOINT R2_BUCKET R2_POD_ACCESS_KEY_ID R2_POD_SECRET_ACCESS_KEY"
  stamp r2-env
fi

# 2. Record the environment without secrets.
python3 - "$WORKSPACE" <<'PY'
import json, platform, subprocess, sys
root = sys.argv[1]
info = {"python": platform.python_version(), "platform": platform.platform()}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
except Exception as error:  # noqa: BLE001 - recorded, not hidden
    info["torch_error"] = type(error).__name__
try:
    info["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True).stdout.strip()
except Exception as error:  # noqa: BLE001
    info["nvidia_smi_error"] = type(error).__name__
json.dump(info, open(f"{root}/environment.json", "w"), indent=2, sort_keys=True)
print(json.dumps(info, indent=2, sort_keys=True))
if not info.get("cuda_available") or info.get("device_count") != 1 or not info.get("bf16_supported"):
    print("exactly one BF16-capable CUDA device is required", file=sys.stderr)
    sys.exit(4)
major, minor = (int(x) for x in platform.python_version_tuple()[:2])
if (major, minor) != (3, 12):
    print("Python 3.12 is required by pyproject (python_requires ==3.12.*)", file=sys.stderr)
    sys.exit(4)
PY

stamp environment

# 3. Install the project against the image's CUDA torch (never reinstall torch).
python3 -m pip install --no-deps -e "$REPO" >/dev/null
python3 -m pip install "pyyaml>=6,<7" "numpy>=2,<3" "pytest>=8,<10" >/dev/null
python3 -c "import scratchcoder, yaml, numpy; print('project import ok')"
stamp install

# 3b. Rehearsal: R2 round trip, then the run's pinned data and the shared initialization,
#     all with the Pod's key (R2_POD_). Each step stops the bootstrap on any mismatch.
if [[ "$MODE" == "rehearsal" ]]; then
  python3 -m tools.rehearsal_ops round-trip --run "$RUN" --pod-id "$RUNPOD_POD_ID" \
    --receipt "$WORKSPACE/r2-round-trip.json" --key-prefix R2_POD_
  # Printed only; the readiness receipt checks the upload rate against the transfer window.
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('R2 round trip bytes/s: upload', r.get('upload_bytes_per_second'), 'download', r.get('download_bytes_per_second'))" "$WORKSPACE/r2-round-trip.json"
  python3 -m tools.rehearsal_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" \
    --root "$DATA_ROOT" --run "$RUN" --receipt "$WORKSPACE/data-fetch.json" --key-prefix R2_POD_ \
    > "$WORKSPACE/data-fetch.log"
  python3 -m tools.rehearsal_binding verify --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" \
    --root "$DATA_ROOT" --run "$RUN" > "$WORKSPACE/data-verification.json"
  python3 -m tools.rehearsal_ops fetch-object --key "$INIT_KEY" --sha256 "$INIT_SHA256" \
    --size "$INIT_SIZE" --output "$WORKSPACE/initialization.pt" \
    --receipt "$WORKSPACE/initialization-fetch.json" --key-prefix R2_POD_
  echo "R2 round trip passed; run $RUN data and the initialization are pinned and verified"
fi

# 3c. Post-training: the R2 round trip under posttrain-runs/<run>/<pod id>/ (gated on the
#     weights-only upload rate), then every bound input fetched pinned and re-verified.
if [[ "$MODE" == "posttrain" ]]; then
  python3 -m tools.posttrain_ops round-trip --run "$RUN" --pod-id "$RUNPOD_POD_ID" \
    --receipt "$WORKSPACE/r2-round-trip.json" --key-prefix R2_POD_
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('R2 round trip bytes/s: upload', r.get('upload_bytes_per_second'), 'download', r.get('download_bytes_per_second'))" "$WORKSPACE/r2-round-trip.json"
  stamp round-trip
  python3 -m tools.posttrain_binding fetch --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" \
    --root "$DATA_ROOT" --receipt "$WORKSPACE/data-fetch.json" --key-prefix R2_POD_ \
    > "$WORKSPACE/data-fetch.log"
  stamp fetch
  python3 -m tools.posttrain_binding verify --binding "$BINDING" --expected-sha256 "$BINDING_SHA256" \
    --root "$DATA_ROOT" > "$WORKSPACE/data-verification.json"
  stamp verify
  echo "R2 round trip passed; run $RUN inputs are pinned and verified"
fi

# 4. CPU reference tests against the installed torch build (skips Docker-only cases).
if [[ "$RUN_TESTS" == "1" ]]; then
  TESTS=(tests/test_model.py tests/test_optimizer.py tests/test_bfloat16.py
    tests/test_training.py tests/test_training_control.py tests/test_gpu_pilot.py
    tests/test_pilot.py tests/test_pilot_data.py tests/test_pilot_preflight.py)
  if [[ "$MODE" == "rehearsal" ]]; then
    # The one-pass loader, validation, R2, the worker (tests/test_rehearsal_run.py), the
    # worker-to-Mac integration and the Mac-side tools the worker's records feed. None
    # needs Docker or the network; the data build's tests stay on the Mac.
    TESTS+=(tests/test_onepass_loader.py tests/test_onepass_order.py
      tests/test_validation_windows.py tests/test_validation.py
      tests/test_r2_store.py tests/test_r2_uploader.py
      tests/test_rehearsal_run.py tests/test_rehearsal_integration.py
      tests/test_rehearsal_binding.py tests/test_rehearsal_ops.py
      tests/test_rehearsal_readiness.py tests/test_rehearsal_supervisor.py
      tests/test_rehearsal_watchdog.py tests/test_rehearsal_feasibility.py
      tests/test_rehearsal_runbook.py tests/test_rehearsal_bootstrap.py)
  elif [[ "$MODE" == "posttrain" ]]; then
    # Generation, chat-v1 and SFT, the R2 machinery, the post-training worker and the
    # Mac-side tools its records feed. None needs Docker or the network.
    TESTS+=(tests/test_generation.py tests/test_chat_format.py tests/test_sft_loader.py
      tests/test_sft_training.py tests/test_validation.py tests/test_rl_checkpoint.py
      tests/test_r2_store.py tests/test_r2_uploader.py
      tests/test_posttrain_run.py tests/test_posttrain_guards.py
      tests/test_posttrain_binding.py tests/test_posttrain_integration.py
      tests/test_posttrain_readiness.py
      tests/test_posttrain_bootstrap.py tests/test_posttrain_feasibility.py)
  fi
  python3 -m pytest "${TESTS[@]}" -q -p no:cacheprovider 2>&1 | tail -5
  stamp tests
fi

# 5. Hardware precision/resume preflight on the single A40; the rehearsal worker also
#    requires the compiled-against-eager check, and its readiness receipt checks it passed.
PREFLIGHT_ARGS=()
if [[ "$MODE" != "pilot" ]]; then
  PREFLIGHT_ARGS=(--compile-check)
fi
if [[ ! -f "$WORKSPACE/preflight-a40/report.json" ]]; then
  python3 -m tools.pilot_preflight --output "$WORKSPACE/preflight-a40" --device cuda:0 \
    ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}
fi
stamp preflight
python3 -c "import hashlib,sys; print('preflight sha256', hashlib.sha256(open('$WORKSPACE/preflight-a40/report.json','rb').read()).hexdigest())"
echo "bootstrap complete; self-destruct receipt: $RECEIPT"
