#!/usr/bin/env bash
set -euo pipefail

# Shared data root that holds home/, workspace/ and tmp/; defaults to the parent of $HOME.
SNU_DATA_ROOT="${SNU_DATA_ROOT:-$(dirname "$HOME")}"

RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-${SNU_DATA_ROOT}/home/runtime/pi05-droid-jointpos-velocity}"
PID_FILE="$RUNTIME_ROOT/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  printf 'SERVER_STOP_STATUS=ALREADY_STOPPED\n'
  exit 0
fi
pid="$(<"$PID_FILE")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f -- "$PID_FILE"
  printf 'SERVER_STOP_STATUS=ALREADY_STOPPED stale_pid=%s\n' "$pid"
  exit 0
fi
if [[ "$(stat -c '%u' "/proc/$pid")" != "$(id -u)" ]]; then
  printf 'ERROR refusing to stop pid=%s owned by another account\n' "$pid" >&2
  exit 1
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
if [[ "$command_line" != *"serve_pi05_droid_jointpos_velocity.py"* ]]; then
  printf 'ERROR refusing to stop pid=%s command=%s\n' "$pid" "$command_line" >&2
  exit 1
fi

kill -TERM "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$PID_FILE"
    printf 'SERVER_STOP_STATUS=PASS pid=%s\n' "$pid"
    exit 0
  fi
  sleep 1
done

printf 'SERVER_STOP_STATUS=TIMEOUT pid=%s action=manual_review_required\n' "$pid" >&2
exit 1
