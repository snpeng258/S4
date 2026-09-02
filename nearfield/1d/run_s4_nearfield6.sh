#!/usr/bin/env bash
# S4 reflected near-field with raw-FFT k-space (no DSP demod).
# Defaults: psP_teP_cgR_kzM (carrier removed in TSV).

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
NEAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${NEAR_DIR}/nearfield_reflected_workflow2.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow script: $WF" >&2; exit 1; }

export KSPACE_MODE="${KSPACE_MODE:-raw}"
export S4_PHASE_SIGN="${S4_PHASE_SIGN:-plus}"
export S4_TE_SIGN="${S4_TE_SIGN:-1}"
export S4_CARRIER_GAUGE="${S4_CARRIER_GAUGE:-remove}"
export S4_Z_TERM_SIGN="${S4_Z_TERM_SIGN:-minus}"

echo "=== run_s4_nearfield6: reflected near-field, raw FFT k-space ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
