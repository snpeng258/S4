#!/usr/bin/env bash
# run4: S4 Floquet + elliptical aperture + ASR far-field reconstruction.

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  _rp="$(readlink -f "$_SCRIPT" 2>/dev/null)" && [[ -n "$_rp" ]] && _SCRIPT="$_rp"
fi
FAR_DIR="$(cd "$(dirname "$_SCRIPT")" && pwd)"

find_s4_repo_root() {
  local d="$FAR_DIR"
  while [[ -n "$d" && "$d" != "/" ]]; do
    if [[ -f "$d/tools/s4_env.sh" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

REPO_ROOT="$(find_s4_repo_root)" || { echo "error: repo root not found (missing tools/s4_env.sh)" >&2; exit 1; }
# shellcheck source=tools/s4_env.sh
source "${REPO_ROOT}/tools/s4_env.sh"
OUT_DIR="${S4_RUNS}/farfield"
mkdir -p "$OUT_DIR"

WL_NM=${WL_NM:-13.5}
ANGLE_DEG=${ANGLE_DEG:-60}
AZIMUTH_DEG=${AZIMUTH_DEG:-90}
PERIOD_NM=${PERIOD_NM:-80}
DEPTH_NM=${DEPTH_NM:-40}
DUTY=${DUTY:-0.5}
NG=${NG:-31}
Z_MM=${Z_MM:-40}
APERTURE_U_UM=${APERTURE_U_UM:-15}
APERTURE_V_UM=${APERTURE_V_UM:-25}
APERTURE_N=${APERTURE_N:-128}
OBS_N=${OBS_N:-4800}
ASR_N_U=${ASR_N_U:-384}
ASR_N_V=${ASR_N_V:-384}
ORDER_R_THRESH=${ORDER_R_THRESH:-1e-5}
POLARIZATION=${POLARIZATION:-TE}
DPI=${DPI:-150}
INTENSITY_SCALE=${INTENSITY_SCALE:-log}
ORDER_PATCH_SCALE=${ORDER_PATCH_SCALE:-linear}
PLOT_ORDER_PATCHES=${PLOT_ORDER_PATCHES:-1}
CCD_SIZE_MM=${CCD_SIZE_MM:-100}
AXIS_HALF_MM=${AXIS_HALF_MM:-$((CCD_SIZE_MM / 2))}
OBS_THETA2_DEG=${OBS_THETA2_DEG:-}
OBS_PHI2_DEG=${OBS_PHI2_DEG:-}
# asr = per-order k-conv + ASR stitched on lab CCD; fraunhofer = analytic; asr-legacy = spatial patch
FARFIELD_METHOD=${FARFIELD_METHOD:-asr}
OBS_SPAN_MM=${OBS_SPAN_MM:-}
LOCAL_HALF_MM=${LOCAL_HALF_MM:-0}
LOCAL_N=${LOCAL_N:-512}
PATCH_FEATHER_FRAC=${PATCH_FEATHER_FRAC:-0}
PASTE_METHOD=${PASTE_METHOD:-native}
STITCH_M_MAX=${STITCH_M_MAX:-1}
STITCH_PASTE_METHOD=${STITCH_PASTE_METHOD:-interp}

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
AZIMUTH_TAG="${AZIMUTH_DEG//./p}"
Z_TAG="${Z_MM//./p}"
APU_TAG="${APERTURE_U_UM//./p}"
APV_TAG="${APERTURE_V_UM//./p}"
POL_TAG="$(printf '%s' "$POLARIZATION" | tr '[:lower:]' '[:upper:]')"

WAVES_TXT="${OUT_DIR}/s4_farfield_wl${WL_TAG}nm_th${ANGLE_TAG}deg_az${AZIMUTH_TAG}deg_L${PERIOD_NM//./p}nm_waves_air.txt"
OUT_PNG="${OUT_DIR}/${WL_TAG}nm_${ANGLE_TAG}deg_az${AZIMUTH_TAG}deg_${POL_TAG}_ap${APU_TAG}x${APV_TAG}um_z${Z_TAG}mm_farfield_asr.png"
OUT_STEM="${OUT_PNG%.png}"

ORDER_PATCH_ARGS=()
if [[ -n "${ORDER_PATCHES_DIR:-}" ]]; then
  ORDER_PATCH_ARGS+=(--order-patches-dir "$ORDER_PATCHES_DIR")
fi

S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};azimuth_deg=${AZIMUTH_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG}"

for req in \
  "${FAR_DIR}/au_grating_farfield.lua" \
  "${FAR_DIR}/s4_farfield_asr.py" \
  "${FAR_DIR}/asr_propagate.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

echo "=== Far-field run4: S4 Floquet + elliptical aperture + ASR ==="
echo "S4: ${S4_BIN}"
echo "θ=${ANGLE_DEG}°, φ=${AZIMUTH_DEG}°, aperture=${APERTURE_U_UM}×${APERTURE_V_UM} μm, z=${Z_MM} mm, method=${FARFIELD_METHOD}"

cd "$FAR_DIR"
if [[ ! -f "$WAVES_TXT" ]] || [[ "${FORCE_S4:-0}" == "1" ]]; then
  "$S4_BIN" au_grating_farfield.lua -a "$S4_ARG" > "$WAVES_TXT"
  echo "Wrote waves TSV: ${WAVES_TXT}"
else
  echo "Reuse waves TSV: ${WAVES_TXT} (set FORCE_S4=1 to rerun S4)"
fi

OBS_ARGS=()
if [[ -n "$OBS_THETA2_DEG" ]]; then
  OBS_ARGS+=(--obs-theta2-deg "$OBS_THETA2_DEG")
fi
if [[ -n "$OBS_PHI2_DEG" ]]; then
  OBS_ARGS+=(--obs-phi2-deg "$OBS_PHI2_DEG")
fi

SPAN_ARGS=()
if [[ -n "$OBS_SPAN_MM" ]]; then
  SPAN_ARGS+=(--obs-span-mm "$OBS_SPAN_MM")
fi

python3 "${FAR_DIR}/s4_farfield_asr.py" \
  "$WAVES_TXT" \
  --out "$OUT_PNG" \
  --method "$FARFIELD_METHOD" \
  --z-mm "$Z_MM" \
  --aperture-u-um "$APERTURE_U_UM" \
  --aperture-v-um "$APERTURE_V_UM" \
  --aperture-n "$APERTURE_N" \
  --obs-n "$OBS_N" \
  --local-half-mm "$LOCAL_HALF_MM" \
  --local-n "$LOCAL_N" \
  --patch-feather-frac "$PATCH_FEATHER_FRAC" \
  --paste-method "$PASTE_METHOD" \
  --stitch-m-max "$STITCH_M_MAX" \
  --stitch-paste-method "$STITCH_PASTE_METHOD" \
  --asr-n-u "$ASR_N_U" \
  --asr-n-v "$ASR_N_V" \
  --order-r-thresh "$ORDER_R_THRESH" \
  --polarization "$POLARIZATION" \
  --dpi "$DPI" \
  --intensity-scale "$INTENSITY_SCALE" \
  --axis-half-mm "$AXIS_HALF_MM" \
  --order-patch-scale "$ORDER_PATCH_SCALE" \
  $( [[ "${PLOT_ORDER_PATCHES}" == "1" ]] && echo --plot-order-patches || echo --no-plot-order-patches ) \
  ${ORDER_PATCH_ARGS[@]+"${ORDER_PATCH_ARGS[@]}"} \
  ${SPAN_ARGS[@]+"${SPAN_ARGS[@]}"} \
  ${OBS_ARGS[@]+"${OBS_ARGS[@]}"}

echo ""
echo "=== Outputs (${OUT_DIR}) ==="
echo "  Stitched: ${OUT_PNG}"
if [[ "${PLOT_ORDER_PATCHES}" == "1" ]]; then
  echo "  Per-order spot PNGs (same directory, linear scale):"
  shopt -s nullglob
  for f in "${OUT_STEM}_order_"*.png "${OUT_STEM}_orders_panel.png"; do
    echo "    ${f}"
  done
  shopt -u nullglob
fi
