#!/usr/bin/env bash
# 调用 S4：二维正方周期阵列上的 Au 正方形突起近场（au_square_grating_nearfield_xy.lua），
# 输出 TSV + 空间 4x2 + k-space PNG。
#
# 与一维脚本同名参数的含义（由 1d 条带推广到 2d 方格）：
#   PERIOD_NM — x、y 方向周期相同，均为该值（正方形原胞，Lx = Ly）。
#   DUTY      — 每个原胞内一个中心正方形 Au 突起，边长 = DUTY × PERIOD_NM（x、y 等长）。
#   DEPTH_NM  — 突起/刻蚀深度（与一维相同）。
# 默认数值与 nearfield/1d/run_s4_nearfield.sh 对齐。
#
# Usage:
#   ./run_s4_nearfield_2d.sh
#   WL_NM=633 NX=48 NY=48 ./run_s4_nearfield_2d.sh
#
# Env:
#   WL_NM ANGLE_DEG PERIOD_NM DEPTH_NM DUTY RCWA_ORDER
#   NX NY Z_PLANE_NORM KSPACE_N_REP KSPACE_PAD DPI

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
OUT_DIR="${S4_RUNS}/nearfield/2d"
mkdir -p "$OUT_DIR"

WL_NM=${WL_NM:-13.5}
ANGLE_DEG=${ANGLE_DEG:-80}
PERIOD_NM=${PERIOD_NM:-80}
DEPTH_NM=${DEPTH_NM:-40}
DUTY=${DUTY:-0.5}
RCWA_ORDER=${RCWA_ORDER:-15}
NX=${NX:-32}
NY=${NY:-32}
Z_PLANE_NORM=${Z_PLANE_NORM:--0.05}
KSPACE_N_REP=${KSPACE_N_REP:-2}
KSPACE_PAD=${KSPACE_PAD:-256}
DPI=${DPI:-150}

S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};rcwa_order=${RCWA_ORDER};nx=${NX};ny=${NY};z_plane_norm=${Z_PLANE_NORM}"

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
PERIOD_TAG="${PERIOD_NM//./p}"
DEPTH_TAG="${DEPTH_NM//./p}"
DUTY_TAG="${DUTY//./p}"
STEM="s4_nearfield_2d_wl${WL_TAG}nm_th${ANGLE_TAG}deg_L${PERIOD_TAG}nm_d${DEPTH_TAG}nm_duty${DUTY_TAG}_ord${RCWA_ORDER}"
DATA_TXT="${OUT_DIR}/${STEM}.txt"
OUT_PNG="${OUT_DIR}/${STEM}.png"
OUT_KSPACE="${OUT_DIR}/${STEM}_kspace.png"

if [[ ! -f "${NEAR_DIR}/au_square_grating_nearfield_xy.lua" ]]; then
  echo "error: au_square_grating_nearfield_xy.lua not in ${NEAR_DIR}" >&2
  exit 1
fi
if [[ ! -f "${NEAR_DIR}/plot_s4_nearfield_xy_viz.py" ]]; then
  echo "error: plot_s4_nearfield_xy_viz.py not found" >&2
  exit 1
fi

AU_SIDE_NM=$(awk -v d="$DUTY" -v p="$PERIOD_NM" 'BEGIN{printf "%.12g", d * p}')
echo "=== S4 2D near-field (square lattice Lx=Ly=${PERIOD_NM} nm, Au square bump side=${AU_SIDE_NM} nm) ==="
echo "S4: ${S4_BIN}"
echo "Raw log (TSV inside): ${DATA_TXT}"
echo "Spatial PNG: ${OUT_PNG}"
echo "k-space PNG: ${OUT_KSPACE}"

cd "$NEAR_DIR"
"$S4_BIN" au_square_grating_nearfield_xy.lua -a "$S4_ARG" > "$DATA_TXT"
echo "Wrote raw log: ${DATA_TXT}"

python3 "${NEAR_DIR}/plot_s4_nearfield_xy_viz.py" "$DATA_TXT" \
  --out "$OUT_PNG" \
  --out-kspace "$OUT_KSPACE" \
  --wl "$WL_NM" \
  --angle "$ANGLE_DEG" \
  --dpi "$DPI" \
  --kspace-n-rep "$KSPACE_N_REP" \
  --kspace-pad "$KSPACE_PAD"
