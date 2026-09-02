#!/usr/bin/env bash
# Far-field workflow: GetWaves("Air") export -> Floquet reconstruct -> reflected |E|^2 PNG.

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
ANGLE_DEG=${ANGLE_DEG:-80}
PERIOD_NM=${PERIOD_NM:-80}
DEPTH_NM=${DEPTH_NM:-40}
DUTY=${DUTY:-0.5}
NG=${NG:-31}
Z_MM=${Z_MM:-40}
X_SPAN_MM=${X_SPAN_MM:-15}
Y_SPAN_MM=${Y_SPAN_MM:-15}
NX=${NX:-256}
NY=${NY:-256}
POLARIZATION=${POLARIZATION:-TE}
DPI=${DPI:-150}
SCREEN_MODE=${SCREEN_MODE:-auto}
ORDER_HALF_WIDTH_MM=${ORDER_HALF_WIDTH_MM:-8}
ORDER_R_THRESH=${ORDER_R_THRESH:-1e-5}

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
Z_TAG="${Z_MM//./p}"
POL_TAG="$(printf '%s' "$POLARIZATION" | tr '[:lower:]' '[:upper:]')"

WAVES_TXT="${OUT_DIR}/s4_farfield_wl${WL_TAG}nm_th${ANGLE_TAG}deg_L${PERIOD_NM//./p}nm_waves_air.txt"
OUT_PNG="${OUT_DIR}/${WL_TAG}nm_${ANGLE_TAG}deg_${POL_TAG}_z${Z_TAG}mm_farfield_reflected.png"
OVERVIEW_PNG="${OUT_DIR}/${WL_TAG}nm_${ANGLE_TAG}deg_${POL_TAG}_z${Z_TAG}mm_order_screens_map.png"

S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG}"

for req in \
  "${FAR_DIR}/au_grating_farfield.lua" \
  "${FAR_DIR}/s4_farfield_reconstruct.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

echo "=== S4 far-field workflow (GetWaves + Floquet) ==="
echo "S4: ${S4_BIN}"
echo "z_obs=${Z_MM} mm (air side, z<0); screen_mode=${SCREEN_MODE}"
echo "Reflected field: E_total(Floquet) - E_inc(S4 planewave)"

cd "$FAR_DIR"
"$S4_BIN" au_grating_farfield.lua -a "$S4_ARG" > "$WAVES_TXT"
echo "Wrote waves TSV: ${WAVES_TXT}"

python3 "${FAR_DIR}/s4_farfield_reconstruct.py" \
  "$WAVES_TXT" \
  --out "$OUT_PNG" \
  --z-mm "$Z_MM" \
  --x-span-mm "$X_SPAN_MM" \
  --y-span-mm "$Y_SPAN_MM" \
  --nx "$NX" \
  --ny "$NY" \
  --polarization "$POLARIZATION" \
  --dpi "$DPI" \
  --screen-mode "$SCREEN_MODE" \
  --order-half-width-mm "$ORDER_HALF_WIDTH_MM" \
  --order-r-thresh "$ORDER_R_THRESH" \
  --overview-out "$OVERVIEW_PNG"
echo "Wrote far-field PNG: ${OUT_PNG}"
echo "Wrote order map PNG: ${OVERVIEW_PNG}"
