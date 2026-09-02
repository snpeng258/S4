#!/usr/bin/env bash
# S4 reflected near-field (incident subtracted) + spatial / k-space plots.

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
NEAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${NEAR_DIR}/nearfield_reflected_workflow.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow script: $WF" >&2; exit 1; }

echo "=== run_s4_nearfield5: S4 reflected near-field ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
