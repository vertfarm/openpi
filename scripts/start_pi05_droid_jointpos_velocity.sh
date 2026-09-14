#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_INDEX="${1:-3}"
PORT="${PORT:-8000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi05_droid_jointpos}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/pi05-droid-jointpos-velocity-team2}"
PID_FILE="$RUNTIME_ROOT/server.pid"
GPU_FILE="$RUNTIME_ROOT/physical-gpu"
PORT_FILE="$RUNTIME_ROOT/port"
MANIFEST_PATH="$RUNTIME_ROOT/checkpoint-manifest.json"

if [[ "$(id -un)" != "snu" ]]; then
  printf 'ERROR this launcher must run as the snu account (current=%s)\n' "$(id -un)" >&2
  exit 1
fi
if [[ "$GPU_INDEX" != "2" && "$GPU_INDEX" != "3" ]]; then
  printf 'ERROR SNU may use only physical GPU 2 or 3 (requested=%s)\n' "$GPU_INDEX" >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf 'ERROR invalid port: %s\n' "$PORT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'ERROR Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'ERROR nvidia-smi not found\n' >&2
  exit 1
fi
if ! command -v ss >/dev/null 2>&1; then
  printf 'ERROR ss not found\n' >&2
  exit 1
fi

mkdir -p "$RUNTIME_ROOT/logs" \
  "/data/keti/snu/home/.cache/openpi" \
  "/data/keti/snu/home/.cache/xdg" \
  "/data/keti/snu/home/.cache/jax" \
  "/data/keti/snu/tmp"

if [[ -f "$PID_FILE" ]]; then
  running_pid="$(<"$PID_FILE")"
  if [[ "$running_pid" =~ ^[0-9]+$ ]] && kill -0 "$running_pid" 2>/dev/null; then
    printf 'ERROR server already running pid=%s\n' "$running_pid" >&2
    exit 1
  fi
fi
if ss -H -ltn | awk -v suffix=":$PORT" '$4 ~ (suffix "$") { found=1 } END { exit !found }'; then
  printf 'ERROR port %s is already listening\n' "$PORT" >&2
  exit 1
fi

IFS=',' read -r gpu_uuid gpu_memory gpu_utilization <<<"$(
  nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX"
)"
gpu_uuid="${gpu_uuid//[[:space:]]/}"
gpu_memory="${gpu_memory//[[:space:]]/}"
gpu_utilization="${gpu_utilization//[[:space:]]/}"
if ! [[ "$gpu_memory" =~ ^[0-9]+$ && "$gpu_utilization" =~ ^[0-9]+$ ]]; then
  printf 'ERROR could not parse GPU state: memory=%s utilization=%s\n' "$gpu_memory" "$gpu_utilization" >&2
  exit 1
fi
if (( gpu_memory > 2048 || gpu_utilization > 10 )); then
  printf 'ERROR GPU %s is busy: memory=%sMiB utilization=%s%%\n' "$GPU_INDEX" "$gpu_memory" "$gpu_utilization" >&2
  exit 1
fi

root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
data_free_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
if (( root_free_kib < 15 * 1024 * 1024 )); then
  printf 'ERROR root filesystem has less than 15GiB free\n' >&2
  exit 1
fi
if (( data_free_kib < 50 * 1024 * 1024 )); then
  printf 'ERROR /data has less than 50GiB free\n' >&2
  exit 1
fi

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/keti/snu/home/.cache/openpi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/data/keti/snu/home/.cache/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/data/keti/snu/home/.cache/jax}"
export TMPDIR="${TMPDIR:-/data/keti/snu/tmp}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export DROID_PHYSICAL_GPU="$GPU_INDEX"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
"$PYTHON_BIN" scripts/prepare_pi05_droid_jointpos.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --manifest-path "$MANIFEST_PATH"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$RUNTIME_ROOT/logs/server-$timestamp.log"
commit="$(git rev-parse HEAD)"
printf '%s\n' "$GPU_INDEX" >"$GPU_FILE"
printf '%s\n' "$PORT" >"$PORT_FILE"

nohup "$PYTHON_BIN" scripts/serve_pi05_droid_jointpos_velocity.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --port "$PORT" \
  >"$log_file" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$PID_FILE"

sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  printf 'ERROR server exited during startup; log=%s\n' "$log_file" >&2
  tail -n 40 "$log_file" >&2 || true
  exit 1
fi

printf 'SERVER_START_STATUS=STARTING\n'
printf 'pid=%s\nphysical_gpu=%s\ngpu_uuid=%s\nport=%s\nrepo_commit=%s\ncheckpoint=%s\nlog=%s\n' \
  "$server_pid" "$GPU_INDEX" "$gpu_uuid" "$PORT" "$commit" "$CHECKPOINT_DIR" "$log_file"
printf 'NEXT=%s/scripts/status_pi05_droid_jointpos_velocity.sh\n' "$REPO"
