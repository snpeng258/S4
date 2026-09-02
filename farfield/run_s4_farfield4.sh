#!/usr/bin/env bash
# run4: S4 Floquet + k-domain elliptical aperture + ASR far-field.

set -euo pipefail

_here="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
FAR_DIR="$(cd "$(dirname "$_here")" && pwd)"
WF="${FAR_DIR}/farfield_asr_workflow.sh"

[[ -f "$WF" ]] || { echo "error: missing workflow: $WF" >&2; exit 1; }

export ANGLE_DEG="${ANGLE_DEG:-60}"
export AZIMUTH_DEG="${AZIMUTH_DEG:-90}"
export Z_MM="${Z_MM:-40}"
export APERTURE_U_UM="${APERTURE_U_UM:-15}"
export APERTURE_V_UM="${APERTURE_V_UM:-25}"
export APERTURE_N="${APERTURE_N:-128}"
export OBS_N="${OBS_N:-4800}"
export FARFIELD_METHOD="${FARFIELD_METHOD:-asr}"
export OBS_SPAN_MM="${OBS_SPAN_MM:-}"
export LOCAL_HALF_MM="${LOCAL_HALF_MM:-0}"
export LOCAL_N="${LOCAL_N:-512}"
export PATCH_FEATHER_FRAC="${PATCH_FEATHER_FRAC:-0}"
export PASTE_METHOD="${PASTE_METHOD:-native}"
export STITCH_M_MAX="${STITCH_M_MAX:-1}"
export STITCH_PASTE_METHOD="${STITCH_PASTE_METHOD:-interp}"
export ASR_N_U="${ASR_N_U:-384}"
export ASR_N_V="${ASR_N_V:-384}"
export ORDER_R_THRESH="${ORDER_R_THRESH:-1e-5}"
export POLARIZATION="${POLARIZATION:-TE}"
export CCD_SIZE_MM="${CCD_SIZE_MM:-100}"
export INTENSITY_SCALE="${INTENSITY_SCALE:-log}"
export ORDER_PATCH_SCALE="${ORDER_PATCH_SCALE:-linear}"
export PLOT_ORDER_PATCHES="${PLOT_ORDER_PATCHES:-1}"

echo "=== run_s4_farfield4: method=${FARFIELD_METHOD} @ z=${Z_MM} mm ==="
echo "engine: ${WF}"

exec bash "$WF" "$@"
