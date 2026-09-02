#!/usr/bin/env bash
# run3: finite-spot far-field | S4 angular-spectrum multi-angle RCWA + Floquet coherent sum.

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
FAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${FAR_DIR}/farfield_spot_spectrum_workflow.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow: $WF" >&2; exit 1; }

export ANGLE_DEG="${ANGLE_DEG:-60}"
export SPOT_W0_UM="${SPOT_W0_UM:-50}"
export AZIMUTH_DEG="${AZIMUTH_DEG:-90}"
export N_THETA="${N_THETA:-15}"
export THETA_SIGMA_SCALE="${THETA_SIGMA_SCALE:-3}"
export Z_MM="${Z_MM:-40}"
export ORDER_HALF_WIDTH_MM="${ORDER_HALF_WIDTH_MM:-8}"
export ORDER_R_THRESH="${ORDER_R_THRESH:-1e-5}"
export SCREEN_MODE="${SCREEN_MODE:-auto}"
export NX="${NX:-128}"
export NY="${NY:-32}"
export POLARIZATION="${POLARIZATION:-TE}"

echo "=== run_s4_farfield3: finite spot (S4 angular spectrum) @ z=${Z_MM} mm ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
