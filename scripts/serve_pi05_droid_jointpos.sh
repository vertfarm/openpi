#!/usr/bin/env bash
# Serve a pi05-DROID *joint position* checkpoint over the openpi websocket policy server.
#
# This is the counterpart to serving `pi0_fast_droid` / `pi05_droid`, which are joint *velocity*
# policies. The `pi05_droid_jointpos` train config (src/openpi/training/config.py) predicts joint
# position deltas and adds the current state back on the way out, so the server returns absolute
# joint position targets -- a chunk of shape [16, 8] (7 joint positions + 1 gripper position).
#
# The DROID client (examples/droid/main.py) connects to this server; it must be started with
# `RobotEnv(action_space="joint_position", ...)`, which is now the default there.
#
# Usage:
#   scripts/serve_pi05_droid_jointpos.sh [GPU_INDEX]
#
# Environment overrides:
#   PORT            websocket port to listen on          (default: 8000)
#   CHECKPOINT_DIR  checkpoint to load                   (default: the team1 jointpos export)
#   POLICY_CONFIG   openpi train config name             (default: pi05_droid_jointpos)
#   PYTHON_BIN      interpreter to run serve_policy.py   (default: <repo>/.venv/bin/python)

set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_INDEX="${1:-0}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/keti/snu/workspace/ckpt/team1/pi05_droid_jointpos_pytorch}"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"

if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  printf 'Invalid GPU index: %s\n' "$GPU_INDEX" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

# The PyTorch export is detected by policy_config.create_trained_policy via model.safetensors, and
# the DROID norm stats are read from <checkpoint>/assets/droid. Fail early with a clear message
# rather than deep inside model loading.
if [[ ! -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  printf 'Checkpoint is missing %s/model.safetensors (is the transfer complete?)\n' "$CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT_DIR/assets/droid" ]]; then
  printf 'Checkpoint is missing %s/assets/droid (norm stats)\n' "$CHECKPOINT_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

printf 'config=%s\ncheckpoint=%s\nport=%s\nphysical_gpu=%s\n' \
  "$POLICY_CONFIG" "$CHECKPOINT_DIR" "$PORT" "$GPU_INDEX"

cd "$REPO"
exec "$PYTHON_BIN" scripts/serve_policy.py \
  --port="$PORT" \
  policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$CHECKPOINT_DIR"
