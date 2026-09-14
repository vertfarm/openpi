#!/usr/bin/env bash
# Report the state of the Cosmos3 policy server + velocity proxy pair started by
# start_cosmos_droid_velocity.sh.
#
#   SERVER_STATUS=READY     both processes alive, upstream port open, proxy answers /healthz
#   SERVER_STATUS=STARTING  processes alive but the proxy has not opened its port yet (warmup)
#   SERVER_STATUS=STOPPED   nothing running
#   SERVER_STATUS=ERROR     one half is dead or a pid points at something else
set -euo pipefail

RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/cosmos-droid-velocity}"
SERVER_PID_FILE="$RUNTIME_ROOT/cosmos.pid"
PROXY_PID_FILE="$RUNTIME_ROOT/proxy.pid"
GPU_FILE="$RUNTIME_ROOT/physical-gpu"
PORT_FILE="$RUNTIME_ROOT/port"
UPSTREAM_PORT_FILE="$RUNTIME_ROOT/upstream-port"
VARIANT_FILE="$RUNTIME_ROOT/variant"

# check_pid <label> <pid_file> <expected command substring>  -> prints state, returns 0 if alive
check_pid() {
  local label="$1" pid_file="$2" expect="$3" pid command_line
  if [[ ! -f "$pid_file" ]]; then
    printf '%s=STOPPED reason=no_pid_file\n' "$label"; return 1
  fi
  pid="$(<"$pid_file")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    printf '%s=STOPPED reason=stale_pid pid=%s\n' "$label" "$pid"; return 1
  fi
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  if [[ "$command_line" != *"$expect"* ]]; then
    printf '%s=ERROR reason=pid_identity_mismatch pid=%s command=%s\n' "$label" "$pid" "$command_line" >&2; return 2
  fi
  printf '%s=RUNNING pid=%s\n' "$label" "$pid"
  return 0
}

latest_log() {  # latest_log <prefix>
  find "$RUNTIME_ROOT/logs" -maxdepth 1 -type f -name "$1-*.log" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

server_state=0; proxy_state=0
check_pid COSMOS "$SERVER_PID_FILE" "action_policy_server_robolab"   || server_state=$?
check_pid PROXY  "$PROXY_PID_FILE"  "serve_cosmos_droid_velocity.py" || proxy_state=$?

if (( server_state == 1 && proxy_state == 1 )); then
  printf 'SERVER_STATUS=STOPPED\n'; exit 1
fi
if (( server_state != 0 || proxy_state != 0 )); then
  printf 'SERVER_STATUS=ERROR reason=half_running cosmos_state=%s proxy_state=%s\n' "$server_state" "$proxy_state" >&2
  for prefix in cosmos proxy; do
    log="$(latest_log "$prefix")"
    [[ -n "$log" ]] && { printf '%s_log=%s\n' "$prefix" "$log"; tail -n 15 "$log"; }
  done
  exit 1
fi

gpu="$(<"$GPU_FILE")"; port="$(<"$PORT_FILE")"; upstream_port="$(<"$UPSTREAM_PORT_FILE")"
variant="$(cat "$VARIANT_FILE" 2>/dev/null || printf 'unknown')"
printf 'physical_gpu=%s\nport=%s\nupstream_port=%s\nvariant=%s\n' "$gpu" "$port" "$upstream_port" "$variant"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader -i "$gpu"

if ss -H -ltn | awk -v suffix=":$upstream_port" '$4 ~ (suffix "$") { found=1 } END { exit !found }'; then
  printf 'upstream_port_%s=LISTENING\n' "$upstream_port"
else
  printf 'upstream_port_%s=CLOSED (cosmos server still loading)\n' "$upstream_port"
fi

if command -v curl >/dev/null 2>&1 && curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null; then
  printf 'SERVER_STATUS=READY\n'
  proxy_log="$(latest_log proxy)"
  [[ -n "$proxy_log" ]] && grep -E "Warmup|SERVER_READY|First request|saturation_fraction" "$proxy_log" | tail -n 4 || true
  exit 0
fi

printf 'SERVER_STATUS=STARTING\n'
for prefix in cosmos proxy; do
  log="$(latest_log "$prefix")"
  if [[ -n "$log" ]]; then
    printf '%s_log=%s\n' "$prefix" "$log"
    # Drop the tracebacks the proxy's plain-TCP liveness probe leaves in the cosmos log.
    grep -vE '^\s+|^\[rank0\]:\s+File|^\s*$|^(Traceback|EOFError|websockets\.|The above exception|INFO:websockets)' "$log" \
      | tail -n 8 | cut -c1-200
  fi
done
