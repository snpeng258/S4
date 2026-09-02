#!/usr/bin/env bash
# Same reflected-field chain as nearfield_reflected_workflow.sh, but k-space uses
# a direct FFT (no DSP demod/tile/window). Invoked by run_s4_nearfield6.sh.

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

N_INC_REAL=${N_INC_REAL:-1.0}

# Reflected-field conventions (scan winner: psP_teP_cgR_kzM)
S4_PHASE_SIGN=${S4_PHASE_SIGN:-plus}
S4_TE_SIGN=${S4_TE_SIGN:-1}
S4_CARRIER_GAUGE=${S4_CARRIER_GAUGE:-remove}
S4_Z_TERM_SIGN=${S4_Z_TERM_SIGN:-minus}
S4_KSPACE_DEMOD=${S4_KSPACE_DEMOD:-minus}
KSPACE_N_REP=${KSPACE_N_REP:-2}
KSPACE_PAD=${KSPACE_PAD:-256}

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
  NX=${NX:-512}
  NZ=${NZ:-1}
  S4_ARG="wl_nm=${WL_NM};angle_deg=${ANGLE_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG};x_span_pitch=${X_SPAN};z_surface_nm=${Z_SURFACE_NM};nx=${NX};nz=${NZ}"
fi

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
PERIOD_TAG="${PERIOD_NM//./p}"
DEPTH_TAG="${DEPTH_NM//./p}"
DUTY_TAG="${DUTY//./p}"
STEM="s4_nearfield2_wl${WL_TAG}nm_th${ANGLE_TAG}deg_L${PERIOD_TAG}nm_d${DEPTH_TAG}nm_duty${DUTY_TAG}_NG${NG}"
DATA_TXT="${OUT_DIR}/${STEM}.txt"
REFL_TXT="${OUT_DIR}/${STEM}_reflected.txt"
OUT_PNG="${OUT_DIR}/${STEM}.png"
if [[ "$KSPACE_MODE" == "raw" ]]; then
  OUT_KSPACE="${OUT_DIR}/${STEM}_kspace_raw.png"
else
  OUT_KSPACE="${OUT_DIR}/${STEM}_kspace.png"
fi

for req in \
  "${NEAR_DIR}/au_grating_nearfield.lua" \
  "${NEAR_DIR}/s4_total_to_reflected.py" \
  "${NEAR_DIR}/plot_s4_nearfield_viz.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

make_tag() {
  local ps="$1" ts="$2" cg="$3" zs="$4"
  local psn="P"
  [[ "$ps" == "minus" ]] && psn="M"
  local tsn="P"
  [[ "$ts" == "-1" ]] && tsn="M"
  local cgn
  case "$cg" in
    none) cgn="N" ;;
    remove) cgn="R" ;;
    add) cgn="A" ;;
    *) cgn="X" ;;
  esac
  local zsn="P"
  [[ "$zs" == "minus" ]] && zsn="M"
  echo "ps${psn}_te${tsn}_cg${cgn}_kz${zsn}"
}

PHYS_TAG="$(make_tag "$S4_PHASE_SIGN" "$S4_TE_SIGN" "$S4_CARRIER_GAUGE" "$S4_Z_TERM_SIGN")"

if [[ "$S4_CARRIER_GAUGE" == "remove" ]]; then
  KSPACE_DEMOD_SIGN="none"
elif [[ -n "${S4_KSPACE_DEMOD:-}" ]]; then
  KSPACE_DEMOD_SIGN="$S4_KSPACE_DEMOD"
else
  KSPACE_DEMOD_SIGN="minus"
fi

echo "=== S4 near-field: reflected workflow2 (raw FFT k-space) ==="
echo "S4: ${S4_BIN}"
echo "Reflected field: phase_sign=${S4_PHASE_SIGN} te_sign=${S4_TE_SIGN} carrier_gauge=${S4_CARRIER_GAUGE} z_term=${S4_Z_TERM_SIGN} -> tag ${PHYS_TAG}"
if [[ "$KSPACE_MODE" == "raw" ]]; then
  echo "k-space: raw FFT (carrier in TSV via carrier_gauge)"
else
  echo "k-space: dsp demod=${KSPACE_DEMOD_SIGN} n_rep=${KSPACE_N_REP} pad=${KSPACE_PAD}"
fi
echo "Order mapping: RCWA_ORDER=${RCWA_ORDER} <-> NG=${NG} (=2*order+1)"
if [[ "$S4_NEARFIELD_LEGACY_XZ" == "1" ]]; then
  echo "Sampling: legacy x-z nx=${NX} nz=${NZ} z_norm=[${Z_MIN_NORM}, ${Z_MAX_NORM}]"
else
  echo "Sampling: line mode x_span_pitch=${X_SPAN}, z_surface_nm=${Z_SURFACE_NM}, nx=${NX}, nz=${NZ}"
fi

cd "$NEAR_DIR"
"$S4_BIN" au_grating_nearfield.lua -a "$S4_ARG" > "$DATA_TXT"
echo "Wrote S4 total field log: ${DATA_TXT}"

python3 "${NEAR_DIR}/s4_total_to_reflected.py" "$DATA_TXT" \
  --out "$REFL_TXT" \
  --wl "$WL_NM" \
  --angle "$ANGLE_DEG" \
  --period-nm "$PERIOD_NM" \
  --n-inc-real "$N_INC_REAL" \
  --phase-sign "$S4_PHASE_SIGN" \
  --te-sign "$S4_TE_SIGN" \
  --carrier-gauge "$S4_CARRIER_GAUGE" \
  --z-term-sign "$S4_Z_TERM_SIGN" >/dev/null
echo "Wrote S4 reflected TSV: ${REFL_TXT}"

python3 "${NEAR_DIR}/plot_s4_nearfield_viz.py" "$REFL_TXT" \
  --out "$OUT_PNG" \
  --out-kspace "$OUT_KSPACE" \
  --wl "$WL_NM" \
  --angle "$ANGLE_DEG" \
  --n-inc-real "$N_INC_REAL" \
  --dpi "$DPI" \
  --spatial-style "$SPATIAL_STYLE" \
  --line-at-z-nm "$LINE_AT_Z_NM" \
  --kspace-mode "$KSPACE_MODE" \
  --kspace-n-rep "$KSPACE_N_REP" \
  --kspace-pad "$KSPACE_PAD" \
  --kspace-demod-sign "$KSPACE_DEMOD_SIGN" \
  --phys-tag "$PHYS_TAG"
echo "Wrote S4 reflected figures: ${OUT_PNG}, ${OUT_KSPACE}"
