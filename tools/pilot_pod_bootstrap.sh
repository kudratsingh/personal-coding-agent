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
set -euo pipefail

DEADLINE=""
WORKSPACE=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TESTS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deadline-unix) DEADLINE="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --skip-tests) RUN_TESTS=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$DEADLINE" || -z "$WORKSPACE" ]]; then
  echo "usage: $0 --deadline-unix <unix> --workspace <dir> [--repo <dir>] [--skip-tests]" >&2
  exit 2
fi
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
if [[ -f "$RECEIPT" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$RECEIPT'))['state'] in ('armed','retrying','terminating') else 1)"; then
  echo "self-destruct already armed: $RECEIPT"
else
  setsid nohup python3 "$REPO/tools/pod_self_destruct.py" \
    --pod-id "$RUNPOD_POD_ID" \
    --deadline-unix "$DEADLINE" \
    --receipt "$RECEIPT" \
    --stop-file "$WORKSPACE/TERMINATE_NOW" \
    > "$WORKSPACE/self-destruct.log" 2>&1 < /dev/null &
  for _ in $(seq 1 20); do
    if [[ -f "$RECEIPT" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$RECEIPT'))['state']=='armed' else 1)"; then
      break
    fi
    sleep 0.5
  done
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$RECEIPT'))['state']=='armed' else 1)" 2>/dev/null; then
    echo "self-destruct timer failed to arm; see $WORKSPACE/self-destruct.log" >&2
    exit 3
  fi
  echo "self-destruct armed for unix deadline $DEADLINE (receipt: $RECEIPT)"
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

# 3. Install the project against the image's CUDA torch (never reinstall torch).
python3 -m pip install --no-deps -e "$REPO" >/dev/null
python3 -m pip install "pyyaml>=6,<7" "numpy>=2,<3" "pytest>=8,<10" >/dev/null
python3 -c "import scratchcoder, yaml, numpy; print('project import ok')"

# 4. CPU reference tests against the installed torch build (skips Docker-only cases).
if [[ "$RUN_TESTS" == "1" ]]; then
  python3 -m pytest tests/test_model.py tests/test_optimizer.py tests/test_bfloat16.py \
    tests/test_training.py tests/test_training_control.py tests/test_gpu_pilot.py \
    tests/test_pilot.py tests/test_pilot_data.py tests/test_pilot_preflight.py \
    -q -p no:cacheprovider 2>&1 | tail -5
fi

# 5. Hardware precision/resume preflight on the single A40.
if [[ ! -f "$WORKSPACE/preflight-a40/report.json" ]]; then
  python3 -m tools.pilot_preflight --output "$WORKSPACE/preflight-a40" --device cuda:0
fi
python3 -c "import hashlib,sys; print('preflight sha256', hashlib.sha256(open('$WORKSPACE/preflight-a40/report.json','rb').read()).hexdigest())"
echo "bootstrap complete; self-destruct receipt: $RECEIPT"
