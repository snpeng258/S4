#!/usr/bin/env bash
# S4 total near-field (GetEField, no incident subtraction) + raw-FFT k-space.

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
NEAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${NEAR_DIR}/nearfield_total_workflow.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow script: $WF" >&2; exit 1; }

export KSPACE_MODE="${KSPACE_MODE:-raw}"

echo "=== run_s4_nearfield7: total near-field, raw FFT k-space ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
