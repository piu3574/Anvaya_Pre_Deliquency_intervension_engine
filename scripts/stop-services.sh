#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

ensure_dirs

stop_service "frontend" "$FRONTEND_PID_FILE"
stop_service "scoring-api" "$SCORING_PID_FILE"
stop_service "dashboard-api" "$DASHBOARD_PID_FILE"
