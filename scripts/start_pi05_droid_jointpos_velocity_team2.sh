#!/usr/bin/env bash
# Start the KETI DROID velocity server on the team2 checkpoint or on the official pi0.5 base.
#
# Usage:
#   scripts/start_pi05_droid_jointpos_velocity_team2.sh GPU_INDEX CHECKPOINT
#
#   GPU_INDEX   physical GPU 2 or 3
#   CHECKPOINT  required, one of
#                 ours  -> /data/keti/snu/workspace/ckpt/team2/q26_groupw_step18_openpi_pytorch (PyTorch safetensors)
#                 base  -> gs://openpi-assets/checkpoints/pi05_droid_jointpos (official JAX checkpoint)
#                 a bare name under /data/keti/snu/workspace/ckpt/team2, an explicit directory, or a gs:// URL
#               There is no default: the launcher refuses to start without it so the wrong model is never
#               served by accident. CHECKPOINT_DIR=... may be used instead of the positional argument.
#
# Examples:
#   scripts/start_pi05_droid_jointpos_velocity_team2.sh 3 ours
#   scripts/start_pi05_droid_jointpos_velocity_team2.sh 3 base
#   scripts/start_pi05_droid_jointpos_velocity_team2.sh 3 q24_gae099off50_step6_openpi_pytorch
#
# A directory containing model.safetensors is loaded through the upstream PyTorch branch of
# policy_config.create_trained_policy; a directory containing params/ (the gs:// base) is loaded through the
# JAX branch. Both go through the same pi05_droid_jointpos_velocity output pipeline, so the client contract
# (15 x 8 normalized joint velocities + gripper) is identical for every checkpoint.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_INDEX="${1:-}"
PORT="${PORT:-8000}"
TEAM2_CKPT_ROOT="${TEAM2_CKPT_ROOT:-/data/keti/snu/workspace/ckpt/team2}"
BASE_CHECKPOINT="gs://openpi-assets/checkpoints/pi05_droid_jointpos"
OURS_CHECKPOINT_NAME="q24_gae099off50_step6_openpi_pytorch"
CHECKPOINT_DIR="${2:-${CHECKPOINT_DIR:-}}"
SERVE_SEED="${SERVE_SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/pi05-droid-jointpos-velocity-team2}"
PID_FILE="$RUNTIME_ROOT/server.pid"
GPU_FILE="$RUNTIME_ROOT/physical-gpu"
PORT_FILE="$RUNTIME_ROOT/port"
CHECKPOINT_FILE="$RUNTIME_ROOT/checkpoint"
MANIFEST_PATH="$RUNTIME_ROOT/checkpoint-manifest.json"

if [[ -z "$GPU_INDEX" || -z "$CHECKPOINT_DIR" ]]; then
  printf 'ERROR usage: %s GPU_INDEX {ours|base|<checkpoint>}\n' "${BASH_SOURCE[0]}" >&2
  printf '  ours -> %s/%s\n  base -> %s\n' "$TEAM2_CKPT_ROOT" "$OURS_CHECKPOINT_NAME" "$BASE_CHECKPOINT" >&2
  exit 2
fi

# "ours" -> team2 q26; "base" -> official JAX checkpoint; a bare name -> team2 checkpoint root;
# gs:// URLs and explicit paths as-is.
if [[ "$CHECKPOINT_DIR" == "ours" ]]; then
  CHECKPOINT_DIR="$TEAM2_CKPT_ROOT/$OURS_CHECKPOINT_NAME"
elif [[ "$CHECKPOINT_DIR" == "base" ]]; then
  CHECKPOINT_DIR="$BASE_CHECKPOINT"
elif [[ "$CHECKPOINT_DIR" != *"://"* && "$CHECKPOINT_DIR" != */* ]]; then
  CHECKPOINT_DIR="$TEAM2_CKPT_ROOT/$CHECKPOINT_DIR"
fi

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
if ! [[ "$SERVE_SEED" =~ ^[0-9]+$ ]]; then
  printf 'ERROR invalid SERVE_SEED: %s\n' "$SERVE_SEED" >&2
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

# Local checkpoints must be complete before anything is started (remote gs:// paths are fetched by prepare).
weight_format="remote"
if [[ "$CHECKPOINT_DIR" != *"://"* ]]; then
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    printf 'ERROR checkpoint directory not found: %s\n' "$CHECKPOINT_DIR" >&2
    printf 'available under %s (or use "ours" / "base"):\n' "$TEAM2_CKPT_ROOT" >&2
    ls -1 "$TEAM2_CKPT_ROOT" >&2 || true
    exit 1
  fi
  if [[ -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
    weight_format="pytorch"
  elif [[ -d "$CHECKPOINT_DIR/params" ]]; then
    weight_format="jax"
  else
    printf 'ERROR %s has neither model.safetensors nor params/\n' "$CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$CHECKPOINT_DIR/assets/droid/norm_stats.json" ]]; then
    printf 'ERROR missing %s/assets/droid/norm_stats.json\n' "$CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ "$weight_format" == "pytorch" && -f "$CHECKPOINT_DIR/config.json" ]]; then
    exported_config="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("openpi_config",""))' "$CHECKPOINT_DIR/config.json")"
    if [[ -n "$exported_config" && "$exported_config" != "pi05_droid_jointpos" && "$exported_config" != "pi05_droid_jointpos_velocity" ]]; then
      printf 'ERROR checkpoint config.json openpi_config=%s is not a joint-position DROID checkpoint\n' "$exported_config" >&2
      exit 1
    fi
  fi
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
# JAX is still imported by the policy wrapper; keep it off the GPU so the PyTorch model owns the memory.
if [[ "$weight_format" == "pytorch" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi
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
printf '%s\n' "$CHECKPOINT_DIR" >"$CHECKPOINT_FILE"

nohup "$PYTHON_BIN" scripts/serve_pi05_droid_jointpos_velocity_safetensors.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --port "$PORT" \
  --seed "$SERVE_SEED" \
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
printf 'pid=%s\nphysical_gpu=%s\ngpu_uuid=%s\nport=%s\nrepo_commit=%s\ncheckpoint=%s\nweight_format=%s\nseed=%s\nlog=%s\n' \
  "$server_pid" "$GPU_INDEX" "$gpu_uuid" "$PORT" "$commit" "$CHECKPOINT_DIR" "$weight_format" "$SERVE_SEED" "$log_file"
printf 'NEXT=%s/scripts/status_pi05_droid_jointpos_velocity.sh\n' "$REPO"
