#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

ensure_dirs

start_service "dashboard-api" "$DASHBOARD_PID_FILE" "$DASHBOARD_LOG_FILE" "cd '$ROOT_DIR' && python api/dashboard_app.py"
start_service "scoring-api" "$SCORING_PID_FILE" "$SCORING_LOG_FILE" "cd '$ROOT_DIR' && uvicorn api.main:app --host 0.0.0.0 --port 8000"
start_service "frontend" "$FRONTEND_PID_FILE" "$FRONTEND_LOG_FILE" "cd '$ROOT_DIR/frontend' && npm run dev -- --host 0.0.0.0 --port 5173"

printf "\nService logs:\n"
printf -- "- dashboard: %s\n" "$DASHBOARD_LOG_FILE"
printf -- "- scoring:   %s\n" "$SCORING_LOG_FILE"
printf -- "- frontend:  %s\n" "$FRONTEND_LOG_FILE"
