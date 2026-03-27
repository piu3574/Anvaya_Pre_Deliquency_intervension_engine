#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/.service-pids"
LOG_DIR="$ROOT_DIR/.service-logs"

DASHBOARD_PID_FILE="$PID_DIR/dashboard_api.pid"
SCORING_PID_FILE="$PID_DIR/scoring_api.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

DASHBOARD_LOG_FILE="$LOG_DIR/dashboard_api.log"
SCORING_LOG_FILE="$LOG_DIR/scoring_api.log"
FRONTEND_LOG_FILE="$LOG_DIR/frontend.log"

ensure_dirs() {
  mkdir -p "$PID_DIR" "$LOG_DIR"
}

is_pid_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

read_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    tr -d '[:space:]' <"$pid_file"
  else
    printf ""
  fi
}

service_running_from_file() {
  local pid_file="$1"
  local pid
  pid="$(read_pid_file "$pid_file")"
  if is_pid_running "$pid"; then
    return 0
  fi
  return 1
}

start_service() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local cmd="$4"

  if service_running_from_file "$pid_file"; then
    local existing_pid
    existing_pid="$(read_pid_file "$pid_file")"
    printf "%s is already running (pid=%s)\n" "$name" "$existing_pid"
    return 0
  fi

  rm -f "$pid_file"
  nohup bash -lc "$cmd" >"$log_file" 2>&1 &
  local pid=$!
  printf "%s" "$pid" >"$pid_file"
  printf "Started %s (pid=%s)\n" "$name" "$pid"
}

stop_service() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    printf "%s is not running (no pid file)\n" "$name"
    return 0
  fi

  local pid
  pid="$(read_pid_file "$pid_file")"
  if ! is_pid_running "$pid"; then
    printf "%s is not running (stale pid file removed)\n" "$name"
    rm -f "$pid_file"
    return 0
  fi

  kill "$pid" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    if ! is_pid_running "$pid"; then
      break
    fi
    sleep 0.5
  done

  if is_pid_running "$pid"; then
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi

  rm -f "$pid_file"
  printf "Stopped %s\n" "$name"
}

print_service_status() {
  local name="$1"
  local pid_file="$2"
  local pid
  pid="$(read_pid_file "$pid_file")"
  if is_pid_running "$pid"; then
    printf "%-14s RUNNING (pid=%s)\n" "$name" "$pid"
  else
    printf "%-14s STOPPED\n" "$name"
  fi
}
