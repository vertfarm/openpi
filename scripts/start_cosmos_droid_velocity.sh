#!/usr/bin/env bash
# Launch the Cosmos3 DROID policy behind the KETI joint-velocity contract.
#
# Counterpart of start_pi05_droid_jointpos_velocity.sh. Two processes instead of one:
#
#   [3090]  --8000-->  serve_cosmos_droid_velocity.py (proxy, no GPU)
#                          --8010-->  cosmos_framework action_policy_server_robolab (GPU)
#
# Usage
#   scripts/start_cosmos_droid_velocity.sh <GPU_INDEX: 2|3>
#   VARIANT=ours scripts/start_cosmos_droid_velocity.sh 3      # base checkpoint + LoRA overlay
#
# Tunables (environment): PORT (8000) UPSTREAM_PORT (8010) VARIANT (base|ours)
#   COSMOS_FRAMEWORK_DIR COSMOS_BASE_SNAPSHOT COSMOS_OVERLAY NUM_STEPS GUIDANCE SHIFT
#   SERVER_START_TIMEOUT_S (900) SNU_RUNTIME_ROOT
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_INDEX="${1:-3}"
PORT="${PORT:-8000}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8010}"
VARIANT="${VARIANT:-base}"

FRAMEWORK_DIR="${COSMOS_FRAMEWORK_DIR:-/data/keti/snu/workspace/cosmos-framework}"
SERVER_PYTHON="${SERVER_PYTHON:-$FRAMEWORK_DIR/.venv/bin/python}"
PROXY_PYTHON="${PROXY_PYTHON:-$REPO/.venv/bin/python}"

CKPT_ROOT="/data/keti/snu/workspace/ckpt/team2"
COSMOS_BASE_SNAPSHOT="${COSMOS_BASE_SNAPSHOT:-$CKPT_ROOT/cosmos3_base/snapshots/6706d7680581c255ff61e0f3bb49d90eac55c79e}"
COSMOS_OVERLAY="${COSMOS_OVERLAY:-$CKPT_ROOT/cosmos_fixedscene_r3/overlay.pt}"
HF_HOME_DIR="${HF_HOME:-/data/keti/snu/cache/huggingface}"
GUARDRAIL_SNAPSHOT="$HF_HOME_DIR/hub/models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf"
UV_BIN_DIR="${UV_BIN_DIR:-/data/keti/snu/tools/uv-latest}"

NUM_STEPS="${NUM_STEPS:-2}"   # A6000: ~1 s per step -> 4.1 s round trip (8 steps = 8.0 s)
GUIDANCE="${GUIDANCE:-3.0}"
SHIFT="${SHIFT:-3.0}"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-900}"
MIN_FREE_GPU_MIB=35000   # the server takes ~33.6 GiB on an A6000

RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/cosmos-droid-velocity}"
SERVER_PID_FILE="$RUNTIME_ROOT/cosmos.pid"
PROXY_PID_FILE="$RUNTIME_ROOT/proxy.pid"
GPU_FILE="$RUNTIME_ROOT/physical-gpu"
PORT_FILE="$RUNTIME_ROOT/port"
UPSTREAM_PORT_FILE="$RUNTIME_ROOT/upstream-port"
VARIANT_FILE="$RUNTIME_ROOT/variant"

fail() { printf 'ERROR %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
[[ "$(id -un)" == "snu" ]] || fail "this launcher must run as the snu account (current=$(id -un))"
[[ "$GPU_INDEX" == "2" || "$GPU_INDEX" == "3" ]] || fail "SNU may use only physical GPU 2 or 3 (requested=$GPU_INDEX)"
for p in "$PORT" "$UPSTREAM_PORT"; do
  [[ "$p" =~ ^[0-9]+$ ]] && (( p >= 1024 && p <= 65535 )) || fail "invalid port: $p"
done
[[ "$PORT" != "$UPSTREAM_PORT" ]] || fail "PORT and UPSTREAM_PORT must differ"
[[ "$VARIANT" == "base" || "$VARIANT" == "ours" ]] || fail "VARIANT must be base or ours (got $VARIANT)"
[[ -x "$SERVER_PYTHON" ]] || fail "cosmos-framework interpreter not found: $SERVER_PYTHON (run uv sync in $FRAMEWORK_DIR)"
[[ -x "$PROXY_PYTHON" ]] || fail "proxy interpreter not found: $PROXY_PYTHON (run uv sync in $REPO)"
[[ -f "$FRAMEWORK_DIR/cosmos_framework/scripts/action_policy_server_robolab.py" ]] || fail "server script missing under $FRAMEWORK_DIR"
[[ -f "$FRAMEWORK_DIR/cosmos_framework/rl/sde_sampler.py" ]] || fail "B300 patch missing: $FRAMEWORK_DIR/cosmos_framework/rl/ (see 0914_check.md 6-2)"
[[ -f "$COSMOS_BASE_SNAPSHOT/checkpoint.json" ]] || fail "base checkpoint snapshot missing: $COSMOS_BASE_SNAPSHOT"
if [[ "$VARIANT" == "ours" ]]; then
  [[ -f "$COSMOS_OVERLAY" ]] || fail "LoRA overlay missing: $COSMOS_OVERLAY"
fi
[[ -d "$GUARDRAIL_SNAPSHOT" ]] || fail "Cosmos-Guardrail1 not in HF cache: $GUARDRAIL_SNAPSHOT (gated repo; accept terms on HF and download once)"
[[ -x "$UV_BIN_DIR/uvx" ]] || fail "uvx not found in $UV_BIN_DIR (the server shells out to 'uvx hf download')"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
command -v ss >/dev/null 2>&1 || fail "ss not found"

mkdir -p "$RUNTIME_ROOT/logs" "/data/keti/snu/tmp"

for pid_file in "$SERVER_PID_FILE" "$PROXY_PID_FILE"; do
  if [[ -f "$pid_file" ]]; then
    running_pid="$(<"$pid_file")"
    if [[ "$running_pid" =~ ^[0-9]+$ ]] && kill -0 "$running_pid" 2>/dev/null; then
      fail "already running pid=$running_pid ($pid_file); use scripts/stop_cosmos_droid_velocity.sh"
    fi
  fi
done
for p in "$PORT" "$UPSTREAM_PORT"; do
  if ss -H -ltn | awk -v suffix=":$p" '$4 ~ (suffix "$") { found=1 } END { exit !found }'; then
    fail "port $p is already listening"
  fi
done

IFS=',' read -r gpu_uuid gpu_total gpu_used gpu_util <<<"$(
  nvidia-smi --query-gpu=uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU_INDEX"
)"
gpu_uuid="${gpu_uuid//[[:space:]]/}"; gpu_total="${gpu_total//[[:space:]]/}"
gpu_used="${gpu_used//[[:space:]]/}";  gpu_util="${gpu_util//[[:space:]]/}"
[[ "$gpu_total" =~ ^[0-9]+$ && "$gpu_used" =~ ^[0-9]+$ && "$gpu_util" =~ ^[0-9]+$ ]] \
  || fail "could not parse GPU state: total=$gpu_total used=$gpu_used util=$gpu_util"
gpu_free=$(( gpu_total - gpu_used ))
if (( gpu_free < MIN_FREE_GPU_MIB || gpu_util > 10 )); then
  fail "GPU $GPU_INDEX is busy: free=${gpu_free}MiB (need >= ${MIN_FREE_GPU_MIB}) utilization=${gpu_util}%"
fi

root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
data_free_kib="$(df -Pk /data | awk 'NR==2 {print $4}')"
(( root_free_kib >= 15 * 1024 * 1024 )) || fail "root filesystem has less than 15GiB free"
(( data_free_kib >= 50 * 1024 * 1024 )) || fail "/data has less than 50GiB free"

# ---------------------------------------------------------------- environment
export PATH="$UV_BIN_DIR:$PATH"
export HF_HOME="$HF_HOME_DIR"
export TMPDIR="${TMPDIR:-/data/keti/snu/tmp}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export DROID_PHYSICAL_GPU="$GPU_INDEX"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
server_log="$RUNTIME_ROOT/logs/cosmos-$timestamp.log"
proxy_log="$RUNTIME_ROOT/logs/proxy-$timestamp.log"
commit="$(git -C "$REPO" rev-parse HEAD)"

# ---------------------------------------------------------------- 1. cosmos server
server_args=(
  -m cosmos_framework.scripts.action_policy_server_robolab
  --checkpoint-path "$COSMOS_BASE_SNAPSHOT"
  --port "$UPSTREAM_PORT"
  --num-steps "$NUM_STEPS" --guidance "$GUIDANCE" --shift "$SHIFT"
  --stochastic-sampler --sampler-deterministic
)
if [[ "$VARIANT" == "ours" ]]; then
  # All four LoRA overrides MUST travel in ONE --experiment-overrides flag (0914_check.md risk 8).
  server_args+=(
    --param-overlay "$COSMOS_OVERLAY"
    --experiment-overrides
      model.config.lora_enabled=True
      model.config.lora_rank=16
      model.config.lora_alpha=32
      model.config.lora_target_modules=q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen
  )
fi

cd "$FRAMEWORK_DIR"
nohup "$SERVER_PYTHON" "${server_args[@]}" >"$server_log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$SERVER_PID_FILE"
printf '%s\n' "$GPU_INDEX" >"$GPU_FILE"
printf '%s\n' "$PORT" >"$PORT_FILE"
printf '%s\n' "$UPSTREAM_PORT" >"$UPSTREAM_PORT_FILE"
printf '%s\n' "$VARIANT" >"$VARIANT_FILE"
printf 'COSMOS_SERVER_STATUS=STARTING pid=%s port=%s variant=%s log=%s\n' "$server_pid" "$UPSTREAM_PORT" "$VARIANT" "$server_log"

# The proxy would wait on its own, but if the server dies during model load (checkpoint,
# Guardrail download, OOM) the proxy just sits there. Watch the pid here instead.
deadline=$(( SECONDS + SERVER_START_TIMEOUT_S ))
until ss -H -ltn | awk -v suffix=":$UPSTREAM_PORT" '$4 ~ (suffix "$") { found=1 } END { exit !found }'; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f -- "$SERVER_PID_FILE"
    printf 'ERROR cosmos server exited during startup; log=%s\n' "$server_log" >&2
    tail -n 40 "$server_log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    printf 'ERROR cosmos server did not open port %s within %ss; leaving it running for inspection (pid=%s log=%s)\n' \
      "$UPSTREAM_PORT" "$SERVER_START_TIMEOUT_S" "$server_pid" "$server_log" >&2
    exit 1
  fi
  sleep 5
done
printf 'COSMOS_SERVER_STATUS=LISTENING pid=%s port=%s elapsed_s=%s\n' "$server_pid" "$UPSTREAM_PORT" "$SECONDS"

# ---------------------------------------------------------------- 2. velocity proxy
export PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
nohup "$PROXY_PYTHON" scripts/serve_cosmos_droid_velocity.py \
  --upstream-host 127.0.0.1 --upstream-port "$UPSTREAM_PORT" \
  --port "$PORT" \
  >"$proxy_log" 2>&1 &
proxy_pid=$!
printf '%s\n' "$proxy_pid" >"$PROXY_PID_FILE"

sleep 1
if ! kill -0 "$proxy_pid" 2>/dev/null; then
  printf 'ERROR proxy exited during startup; log=%s\n' "$proxy_log" >&2
  tail -n 40 "$proxy_log" >&2 || true
  exit 1
fi

printf 'SERVER_START_STATUS=STARTING\n'
printf 'cosmos_pid=%s\nproxy_pid=%s\nphysical_gpu=%s\ngpu_uuid=%s\nport=%s\nupstream_port=%s\nvariant=%s\nrepo_commit=%s\ncheckpoint=%s\noverlay=%s\ncosmos_log=%s\nproxy_log=%s\n' \
  "$server_pid" "$proxy_pid" "$GPU_INDEX" "$gpu_uuid" "$PORT" "$UPSTREAM_PORT" "$VARIANT" "$commit" \
  "$COSMOS_BASE_SNAPSHOT" "$([[ "$VARIANT" == "ours" ]] && printf '%s' "$COSMOS_OVERLAY" || printf 'none')" \
  "$server_log" "$proxy_log"
printf 'NOTE proxy warmup runs 2 inferences before opening port %s; the first one on an A6000 can take minutes.\n' "$PORT"
printf 'NEXT=%s/scripts/status_cosmos_droid_velocity.sh\n' "$REPO"
