#!/usr/bin/env bash
# Total near-field: S4 GetEField TSV plotted directly (no incident subtraction).
# k-space: raw FFT. Invoked by run_s4_nearfield7.sh.

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  _rp="$(readlink -f "$_SCRIPT" 2>/dev/null)" && [[ -n "$_rp" ]] && _SCRIPT="$_rp"
fi
NEAR_DIR="$(cd "$(dirname "$_SCRIPT")" && pwd)"

find_s4_repo_root() {
  local d="$NEAR_DIR"
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
OUT_DIR="${S4_RUNS}/nearfield/1d"
mkdir -p "$OUT_DIR"

WL_NM=${WL_NM:-13.5}
ANGLE_DEG=${ANGLE_DEG:-80}
PERIOD_NM=${PERIOD_NM:-80}
DEPTH_NM=${DEPTH_NM:-40}
DUTY=${DUTY:-0.5}
RCWA_ORDER=${RCWA_ORDER:-15}
S4_NEARFIELD_LEGACY_XZ=${S4_NEARFIELD_LEGACY_XZ:-0}
KSPACE_MODE=${KSPACE_MODE:-raw}
DPI=${DPI:-150}
SPATIAL_STYLE=${SPATIAL_STYLE:-line}
LINE_AT_Z_NM=${LINE_AT_Z_NM:--5}

NG=$((2 * RCWA_ORDER + 1))

if [[ "$S4_NEARFIELD_LEGACY_XZ" == "1" ]]; then
  NX=${NX:-32}
  NZ=${NZ:-24}
  Z_MIN_NORM=${Z_MIN_NORM:--0.2}
  Z_MAX_NORM=${Z_MAX_NORM:-}
  if [[ -z "${Z_MAX_NORM}" ]]; then
    Z_MAX_NORM=$(awk -v d="$DEPTH_NM" -v p="$PERIOD_NM" 'BEGIN{printf "%.12g", d/p+0.2}')
  fi
  S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG};use_legacy_xz=1;nx=${NX};nz=${NZ};z_min_norm=${Z_MIN_NORM};z_max_norm=${Z_MAX_NORM}"
else
  X_SPAN=${X_SPAN:-5}
  Z_SURFACE_NM=${Z_SURFACE_NM:--5}
  NX=${NX:-1024}
  NZ=${NZ:-1}
  S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG};x_span_pitch=${X_SPAN};z_surface_nm=${Z_SURFACE_NM};nx=${NX};nz=${NZ}"
fi

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
PERIOD_TAG="${PERIOD_NM//./p}"
DEPTH_TAG="${DEPTH_NM//./p}"
DUTY_TAG="${DUTY//./p}"
STEM="s4_nearfield_total_wl${WL_TAG}nm_th${ANGLE_TAG}deg_L${PERIOD_TAG}nm_d${DEPTH_TAG}nm_duty${DUTY_TAG}_NG${NG}"
DATA_TXT="${OUT_DIR}/${STEM}.txt"
OUT_PNG="${OUT_DIR}/${STEM}.png"
OUT_KSPACE="${OUT_DIR}/${STEM}_kspace_raw.png"

for req in \
  "${NEAR_DIR}/au_grating_nearfield.lua" \
  "${NEAR_DIR}/plot_s4_nearfield_viz.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

echo "=== S4 near-field: total-field workflow (raw FFT k-space, no subtract) ==="
echo "S4: ${S4_BIN}"
echo "k-space: ${KSPACE_MODE} FFT (no carrier removal)"
echo "Order mapping: RCWA_ORDER=${RCWA_ORDER} <-> NG=${NG} (=2*order+1)"
if [[ "$S4_NEARFIELD_LEGACY_XZ" == "1" ]]; then
  echo "Sampling: legacy x-z nx=${NX} nz=${NZ} z_norm=[${Z_MIN_NORM}, ${Z_MAX_NORM}]"
else
  echo "Sampling: line mode x_span_pitch=${X_SPAN}, z_surface_nm=${Z_SURFACE_NM}, nx=${NX}, nz=${NZ}"
fi

cd "$NEAR_DIR"
"$S4_BIN" au_grating_nearfield.lua -a "$S4_ARG" > "$DATA_TXT"
echo "Wrote S4 total field log: ${DATA_TXT}"

python3 "${NEAR_DIR}/plot_s4_nearfield_viz.py" "$DATA_TXT" \
  --out "$OUT_PNG" \
  --wl "$WL_NM" \
  --angle "$ANGLE_DEG" \
  --dpi "$DPI" \
  --spatial-style "$SPATIAL_STYLE" \
  --line-at-z-nm "$LINE_AT_Z_NM" \
  --out-kspace "$OUT_KSPACE" \
  --kspace-mode "$KSPACE_MODE"
echo "Wrote S4 total-field figures: ${OUT_PNG}, ${OUT_KSPACE}"
