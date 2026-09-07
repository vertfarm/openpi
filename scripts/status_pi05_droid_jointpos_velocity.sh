#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${SNU_RUNTIME_ROOT:-/data/keti/snu/home/runtime/pi05-droid-jointpos-velocity}"
PID_FILE="$RUNTIME_ROOT/server.pid"
GPU_FILE="$RUNTIME_ROOT/physical-gpu"
PORT_FILE="$RUNTIME_ROOT/port"

if [[ ! -f "$PID_FILE" ]]; then
  printf 'SERVER_STATUS=STOPPED reason=no_pid_file\n'
  exit 1
fi
pid="$(<"$PID_FILE")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  printf 'SERVER_STATUS=STOPPED reason=stale_pid pid=%s\n' "$pid"
  exit 1
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
if [[ "$command_line" != *"serve_pi05_droid_jointpos_velocity.py"* ]]; then
  printf 'SERVER_STATUS=ERROR reason=pid_identity_mismatch pid=%s command=%s\n' "$pid" "$command_line" >&2
  exit 1
fi

gpu="$(<"$GPU_FILE")"
port="$(<"$PORT_FILE")"
printf 'pid=%s\nphysical_gpu=%s\nport=%s\ncommand=%s\n' "$pid" "$gpu" "$port" "$command_line"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader -i "$gpu"

if command -v curl >/dev/null 2>&1 && curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:$port/healthz" >/dev/null; then
  printf 'SERVER_STATUS=READY\n'
  exit 0
fi

latest_log="$(find "$RUNTIME_ROOT/logs" -maxdepth 1 -type f -name 'server-*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
printf 'SERVER_STATUS=STARTING\n'
if [[ -n "$latest_log" ]]; then
  printf 'log=%s\n' "$latest_log"
  tail -n 20 "$latest_log"
fi
