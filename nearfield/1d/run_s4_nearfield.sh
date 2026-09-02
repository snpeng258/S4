#!/usr/bin/env bash
# Run S4 near-field (au_grating_nearfield.lua), save full text output, then build two PNGs
#（空域默认：固定 z 下 |E|/相位随 x 的曲线 4×2；频域为沿 x 的一维 k_x 谱）
#
# Usage:
#   ./run_s4_nearfield.sh
#   WL_NM=633 NX=48 ./run_s4_nearfield.sh
#
# Env — 默认取点：固定 z_surface、x 为 ±X_SPAN 倍周期、NX=64：
#   X_SPAN Z_SURFACE_NM NX NZ
# 旧版 x–z 扫描：S4_NEARFIELD_LEGACY_XZ=1，此时 NX NZ Z_MIN_NORM Z_MAX_NORM 生效
#   SPATIAL_STYLE=line|imshow  空域图：曲线 vs x（默认 line）或 x–z 伪彩
#   LINE_AT_Z_NM  多 z 时取最接近该 z(nm) 的一层做曲线（默认 -5）
#   KSPACE_N_REP KSPACE_PAD DPI

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
KSPACE_N_REP=${KSPACE_N_REP:-2}
KSPACE_PAD=${KSPACE_PAD:-256}
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
  NX=${NX:-64}
  NZ=${NZ:-1}
  S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG};x_span_pitch=${X_SPAN};z_surface_nm=${Z_SURFACE_NM};nx=${NX};nz=${NZ}"
fi

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
PERIOD_TAG="${PERIOD_NM//./p}"
DEPTH_TAG="${DEPTH_NM//./p}"
DUTY_TAG="${DUTY//./p}"
STEM="s4_nearfield_wl${WL_TAG}nm_th${ANGLE_TAG}deg_L${PERIOD_TAG}nm_d${DEPTH_TAG}nm_duty${DUTY_TAG}_NG${NG}"
DATA_TXT="${OUT_DIR}/${STEM}.txt"
OUT_PNG="${OUT_DIR}/${STEM}.png"
OUT_KSPACE="${OUT_DIR}/${STEM}_kspace.png"

if [[ ! -f "${NEAR_DIR}/au_grating_nearfield.lua" ]]; then
  echo "error: au_grating_nearfield.lua not in ${NEAR_DIR}" >&2
  exit 1
fi
if [[ ! -f "${NEAR_DIR}/plot_s4_nearfield_viz.py" ]]; then
  echo "error: plot_s4_nearfield_viz.py not found" >&2
  exit 1
fi

echo "=== S4 near-field + figures ==="
echo "S4: ${S4_BIN}"
echo "Raw log (TSV inside): ${DATA_TXT}"
echo "Spatial PNG: ${OUT_PNG}"
echo "k-space PNG: ${OUT_KSPACE}"

cd "$NEAR_DIR"
"$S4_BIN" au_grating_nearfield.lua -a "$S4_ARG" > "$DATA_TXT"
echo "Wrote raw log: ${DATA_TXT}"

python3 "${NEAR_DIR}/plot_s4_nearfield_viz.py" "$DATA_TXT" \
  --out "$OUT_PNG" \
  --out-kspace "$OUT_KSPACE" \
  --wl "$WL_NM" \
  --angle "$ANGLE_DEG" \
  --dpi "$DPI" \
  --spatial-style "$SPATIAL_STYLE" \
  --line-at-z-nm "$LINE_AT_Z_NM" \
  --kspace-n-rep "$KSPACE_N_REP" \
  --kspace-pad "$KSPACE_PAD"
