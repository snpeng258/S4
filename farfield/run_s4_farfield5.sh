#!/usr/bin/env bash
# run5: 800 nm HHG — all harmonics with λ∈[10,30] nm on one lab CCD overlay.

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
FAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${FAR_DIR}/farfield_hhg_workflow.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow: $WF" >&2; exit 1; }

export FUNDAMENTAL_NM="${FUNDAMENTAL_NM:-800}"
export WL_MIN="${WL_MIN:-13}"
export WL_MAX="${WL_MAX:-15}"
export HHG_ORDER_STEP="${HHG_ORDER_STEP:-1}"
export HHG_ODD_ONLY="${HHG_ODD_ONLY:-1}"
export DIFF_M_MAX="${DIFF_M_MAX:-2}"
export ANGLE_DEG="${ANGLE_DEG:-60}"
export AZIMUTH_DEG="${AZIMUTH_DEG:-90}"
export Z_MM="${Z_MM:-40}"
export APERTURE_U_UM="${APERTURE_U_UM:-15}"
export APERTURE_V_UM="${APERTURE_V_UM:-25}"
export OBS_N="${OBS_N:-4800}"
export LOCAL_N="${LOCAL_N:-512}"
export ASR_N_U="${ASR_N_U:-384}"
export ASR_N_V="${ASR_N_V:-384}"
export ORDER_R_THRESH="${ORDER_R_THRESH:-1e-5}"
export POLARIZATION="${POLARIZATION:-TE}"
export INTENSITY_SCALE="${INTENSITY_SCALE:-log}"
export ANNOTATE="${ANNOTATE:-1}"

echo "=== run_s4_farfield5: HHG overlay (fundamental=${FUNDAMENTAL_NM} nm, λ=${WL_MIN}–${WL_MAX} nm) ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
