#!/usr/bin/env bash
# Stop the Cosmos3 velocity proxy and the Cosmos3 policy server started by
# start_cosmos_droid_velocity.sh. Proxy first (it holds the robot-facing port), then the server.
set -euo pipefail

RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/cosmos-droid-velocity}"
SERVER_PID_FILE="$RUNTIME_ROOT/cosmos.pid"
PROXY_PID_FILE="$RUNTIME_ROOT/proxy.pid"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-30}"

status=0

# stop_one <label> <pid_file> <expected command substring>
stop_one() {
  local label="$1" pid_file="$2" expect="$3" pid command_line
  if [[ ! -f "$pid_file" ]]; then
    printf '%s_STOP_STATUS=ALREADY_STOPPED\n' "$label"
    return 0
  fi
  pid="$(<"$pid_file")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$pid_file"
    printf '%s_STOP_STATUS=ALREADY_STOPPED stale_pid=%s\n' "$label" "$pid"
    return 0
  fi
  if [[ "$(stat -c '%u' "/proc/$pid")" != "$(id -u)" ]]; then
    printf 'ERROR refusing to stop %s pid=%s owned by another account\n' "$label" "$pid" >&2
    return 1
  fi
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  if [[ "$command_line" != *"$expect"* ]]; then
    printf 'ERROR refusing to stop %s pid=%s command=%s\n' "$label" "$pid" "$command_line" >&2
    return 1
  fi

  kill -TERM "$pid"
  for _ in $(seq 1 "$STOP_TIMEOUT_S"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f -- "$pid_file"
      printf '%s_STOP_STATUS=PASS pid=%s\n' "$label" "$pid"
      return 0
    fi
    sleep 1
  done
  # A stuck server pins ~33 GiB of GPU memory; do not leave it behind.
  printf 'WARN %s pid=%s ignored SIGTERM for %ss; sending SIGKILL\n' "$label" "$pid" "$STOP_TIMEOUT_S" >&2
  kill -KILL "$pid" 2>/dev/null || true
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$pid_file"
    printf '%s_STOP_STATUS=KILLED pid=%s\n' "$label" "$pid"
    return 0
  fi
  printf '%s_STOP_STATUS=TIMEOUT pid=%s action=manual_review_required\n' "$label" "$pid" >&2
  return 1
}

stop_one PROXY  "$PROXY_PID_FILE"  "serve_cosmos_droid_velocity.py" || status=1
stop_one COSMOS "$SERVER_PID_FILE" "action_policy_server_robolab"   || status=1

if (( status == 0 )); then
  printf 'SERVER_STOP_STATUS=PASS\n'
else
  printf 'SERVER_STOP_STATUS=FAIL\n' >&2
fi
exit "$status"
