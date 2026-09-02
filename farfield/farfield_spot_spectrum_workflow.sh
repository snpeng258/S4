#!/usr/bin/env bash
# run3: finite-spot far-field via angular-spectrum multi-angle RCWA + Floquet coherent sum.

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
SPOT_W0_UM=${SPOT_W0_UM:-50}
N_THETA=${N_THETA:-15}
THETA_SIGMA_SCALE=${THETA_SIGMA_SCALE:-3}
ORDER_HALF_WIDTH_MM=${ORDER_HALF_WIDTH_MM:-8}
ORDER_R_THRESH=${ORDER_R_THRESH:-1e-5}
Y_HALF_SPAN_MM=${Y_HALF_SPAN_MM:-2}
NX=${NX:-128}
NY=${NY:-32}
SCREEN_MODE=${SCREEN_MODE:-auto}
POLARIZATION=${POLARIZATION:-TE}
DPI=${DPI:-150}
INTENSITY_SCALE=${INTENSITY_SCALE:-log}
CCD_SIZE_MM=${CCD_SIZE_MM:-100}
AXIS_HALF_MM=${AXIS_HALF_MM:-$((CCD_SIZE_MM / 2))}

WL_TAG="${WL_NM//./p}"
ANGLE_TAG="${ANGLE_DEG//./p}"
AZIMUTH_TAG="${AZIMUTH_DEG//./p}"
Z_TAG="${Z_MM//./p}"
W0_TAG="${SPOT_W0_UM//./p}"
POL_TAG="$(printf '%s' "$POLARIZATION" | tr '[:lower:]' '[:upper:]')"

WAVES_DIR="${OUT_DIR}/spot_spectrum_wl${WL_TAG}nm_th${ANGLE_TAG}deg_az${AZIMUTH_TAG}deg_w${W0_TAG}um"
MANIFEST="${WAVES_DIR}/manifest.tsv"
OUT_PNG="${OUT_DIR}/${WL_TAG}nm_${ANGLE_TAG}deg_az${AZIMUTH_TAG}deg_${POL_TAG}_w${W0_TAG}um_z${Z_TAG}mm_farfield_spot_spectrum.png"

S4_BASE="wl_nm=${WL_NM};azimuth_deg=${AZIMUTH_DEG};period_nm=${PERIOD_NM};depth_nm=${DEPTH_NM};duty=${DUTY};NG=${NG}"

for req in \
  "${FAR_DIR}/au_grating_farfield.lua" \
  "${FAR_DIR}/s4_farfield_spot_spectrum.py"; do
  [[ -f "$req" ]] || { echo "error: missing $req" >&2; exit 1; }
done

mkdir -p "$WAVES_DIR"

echo "=== Far-field run3: S4 angular-spectrum finite spot ==="
echo "S4: ${S4_BIN}"
echo "θ₀=${ANGLE_DEG}°, φ=${AZIMUTH_DEG}°, w0=${SPOT_W0_UM} μm, Nθ=${N_THETA}, z=${Z_MM} mm"

mapfile -t ANGLE_ROWS < <(
  python3 -c "
import sys
sys.path.insert(0, '${FAR_DIR}')
from s4_farfield_reconstruct import gaussian_angle_weights
ths, ws = gaussian_angle_weights(
    ${ANGLE_DEG}, wl_nm=${WL_NM}, w0_um=${SPOT_W0_UM},
    n_theta=${N_THETA}, sigma_scale=${THETA_SIGMA_SCALE},
)
for t, w in zip(ths, ws):
    print(f'{t}\t{w}')
"
)

: > "$MANIFEST"
cd "$FAR_DIR"
for row in "${ANGLE_ROWS[@]}"; do
  theta="$(echo "$row" | awk '{print $1}')"
  weight="$(echo "$row" | awk '{print $2}')"
  th_tag="${theta//./p}"
  th_tag="${th_tag//-/m}"
  wav="${WAVES_DIR}/waves_th${th_tag}deg.txt"
  echo "  S4 angle_deg=${theta} (w=${weight})"
  "$S4_BIN" au_grating_farfield.lua -a "${S4_BASE};angle_deg=${theta}" > "$wav"
  printf '%s\t%s\t%s\n' "$wav" "$weight" "$theta" >> "$MANIFEST"
done

X_HALF_ARG=()
python3 "${FAR_DIR}/s4_farfield_spot_spectrum.py" \
  --manifest "$MANIFEST" \
  --out "$OUT_PNG" \
  --z-mm "$Z_MM" \
  --spot-w0-um "$SPOT_W0_UM" \
  --theta0-deg "$ANGLE_DEG" \
  --wl-nm "$WL_NM" \
  --order-half-width-mm "$ORDER_HALF_WIDTH_MM" \
  --y-half-span-mm "$Y_HALF_SPAN_MM" \
  --nx "$NX" \
  --ny "$NY" \
  --order-r-thresh "$ORDER_R_THRESH" \
  --screen-mode "$SCREEN_MODE" \
  --polarization "$POLARIZATION" \
  --dpi "$DPI" \
  --intensity-scale "$INTENSITY_SCALE" \
  --axis-half-mm "$AXIS_HALF_MM"

echo "Wrote manifest: ${MANIFEST}"
echo "Wrote PNG: ${OUT_PNG}"
