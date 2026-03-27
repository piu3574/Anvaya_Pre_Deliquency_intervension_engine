#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "$SCRIPT_DIR/common.sh"

ensure_dirs

print_service_status "dashboard-api" "$DASHBOARD_PID_FILE"
print_service_status "scoring-api" "$SCORING_PID_FILE"
print_service_status "frontend" "$FRONTEND_PID_FILE"
