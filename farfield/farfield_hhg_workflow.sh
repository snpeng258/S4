#!/usr/bin/env bash
# run5: 800 nm HHG — harmonics with λ in [10,30] nm on one lab CCD.

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

FUNDAMENTAL_NM=${FUNDAMENTAL_NM:-800}
WL_MIN=${WL_MIN:-10}
WL_MAX=${WL_MAX:-30}
HHG_ORDER_LIST=${HHG_ORDER_LIST:-}
HHG_ORDER_STEP=${HHG_ORDER_STEP:-1}
HHG_ODD_ONLY=${HHG_ODD_ONLY:-1}

ANGLE_DEG=${ANGLE_DEG:-60}
AZIMUTH_DEG=${AZIMUTH_DEG:-90}
PERIOD_NM=${PERIOD_NM:-80}
DEPTH_NM=${DEPTH_NM:-40}
DUTY=${DUTY:-0.5}
NG=${NG:-31}
Z_MM=${Z_MM:-40}
APERTURE_U_UM=${APERTURE_U_UM:-15}
APERTURE_V_UM=${APERTURE_V_UM:-25}
OBS_N=${OBS_N:-4800}
ASR_N_U=${ASR_N_U:-384}
ASR_N_V=${ASR_N_V:-384}
LOCAL_HALF_MM=${LOCAL_HALF_MM:-0}
LOCAL_N=${LOCAL_N:-512}
ORDER_R_THRESH=${ORDER_R_THRESH:-1e-5}
DIFF_M_MAX=${DIFF_M_MAX:-2}
POLARIZATION=${POLARIZATION:-TE}
DPI=${DPI:-150}
INTENSITY_SCALE=${INTENSITY_SCALE:-log}
ANNOTATE=${ANNOTATE:-1}
ANNOTATE_MIN_R=${ANNOTATE_MIN_R:-1e-4}
FORCE_S4=${FORCE_S4:-0}

# q range implied by λ bounds: q ∈ [ceil(800/30), floor(800/10)] = [27, 80]
Q_LO=$(python3 - <<PY
import math
print(int(math.ceil(${FUNDAMENTAL_NM} / ${WL_MAX})))
PY
)
Q_HI=$(python3 - <<PY
import math
print(int(math.floor(${FUNDAMENTAL_NM} / ${WL_MIN})))
PY
)

ANGLE_TAG="${ANGLE_DEG//./p}"
AZIMUTH_TAG="${AZIMUTH_DEG//./p}"
Z_TAG="${Z_MM//./p}"
APU_TAG="${APERTURE_U_UM//./p}"
APV_TAG="${APERTURE_V_UM//./p}"
POL_TAG="$(printf '%s' "$POLARIZATION" | tr '[:lower:]' '[:upper:]')"
FUND_TAG="${FUNDAMENTAL_NM//./p}"
WLMIN_TAG="${WL_MIN//./p}"
WLMAX_TAG="${WL_MAX//./p}"

OUT_PNG="${OUT_DIR}/hhg${FUND_TAG}nm_H${Q_LO}-H${Q_HI}_wl${WLMIN_TAG}-${WLMAX_TAG}nm_mpm${DIFF_M_MAX}_${ANGLE_TAG}deg_az${AZIMUTH_TAG}deg_${POL_TAG}_ap${APU_TAG}x${APV_TAG}um_z${Z_TAG}mm_overlay.png"

for req in \
  "${FAR_DIR}/au_grating_farfield.lua" \
  "${FAR_DIR}/s4_farfield_hhg_panel.py" \
  "${FAR_DIR}/s4_farfield_asr.py" \
  "${FAR_DIR}/asr_propagate.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

echo "=== Far-field run5: HHG harmonic overlay (fundamental=${FUNDAMENTAL_NM} nm) ==="
echo "S4: ${S4_BIN}"
echo "λ in [${WL_MIN}, ${WL_MAX}] nm  =>  harmonic orders H${Q_LO}–H${Q_HI} (odd_only=${HHG_ODD_ONLY}, step=${HHG_ORDER_STEP})"
echo "diffraction orders |m| <= ${DIFF_M_MAX}"
echo "θ=${ANGLE_DEG}°, φ=${AZIMUTH_DEG}°, aperture=${APERTURE_U_UM}×${APERTURE_V_UM} μm, z=${Z_MM} mm"

cd "$FAR_DIR"

PANEL_ARGS=(
  --far-dir "$OUT_DIR"
  --s4-bin "$S4_BIN"
  --out "$OUT_PNG"
  --fundamental-nm "$FUNDAMENTAL_NM"
  --wl-min "$WL_MIN"
  --wl-max "$WL_MAX"
  --hhg-order-step "$HHG_ORDER_STEP"
  --angle-deg "$ANGLE_DEG"
  --azimuth-deg "$AZIMUTH_DEG"
  --period-nm "$PERIOD_NM"
  --depth-nm "$DEPTH_NM"
  --duty "$DUTY"
  --ng "$NG"
  --z-mm "$Z_MM"
  --aperture-u-um "$APERTURE_U_UM"
  --aperture-v-um "$APERTURE_V_UM"
  --obs-n "$OBS_N"
  --local-half-mm "$LOCAL_HALF_MM"
  --local-n "$LOCAL_N"
  --asr-n-u "$ASR_N_U"
  --asr-n-v "$ASR_N_V"
  --order-r-thresh "$ORDER_R_THRESH"
  --diff-m-max "$DIFF_M_MAX"
  --polarization "$POLARIZATION"
  --dpi "$DPI"
  --intensity-scale "$INTENSITY_SCALE"
  --annotate-min-r "$ANNOTATE_MIN_R"
)

if [[ -n "$HHG_ORDER_LIST" ]]; then
  PANEL_ARGS+=(--hhg-order-list "$HHG_ORDER_LIST")
fi
if [[ "$HHG_ODD_ONLY" == "0" ]]; then
  PANEL_ARGS+=(--no-odd-only)
fi
if [[ "$ANNOTATE" == "0" ]]; then
  PANEL_ARGS+=(--no-annotate)
fi
if [[ "$FORCE_S4" == "1" ]]; then
  PANEL_ARGS+=(--force-s4)
fi

python3 "${FAR_DIR}/s4_farfield_hhg_panel.py" ${PANEL_ARGS[@]+"${PANEL_ARGS[@]}"} "$@"

echo ""
echo "=== Outputs ==="
echo "  Overlay PNG: ${OUT_PNG}"
echo "  Spot table:  ${OUT_PNG%.png}.spots.json"
