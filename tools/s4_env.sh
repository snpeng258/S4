# Usage (from any workflow):
#   source "${REPO_ROOT}/tools/s4_env.sh"
# If REPO_ROOT is unset, it is inferred from this file's location.
#
# Sets REPO_ROOT, S4_BIN, S4_MATERIALS_DB, S4_RUNS, S4_DATA.
# Existing env values win.

_s4_env_this="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  _rp="$(readlink -f "$_s4_env_this" 2>/dev/null)" && [[ -n "$_rp" ]] && _s4_env_this="$_rp"
fi
_s4_env_root="$(cd "$(dirname "$_s4_env_this")/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-${_s4_env_root}}"
unset _s4_env_this _s4_env_root _rp

if [[ -z "${S4_BIN:-}" ]]; then
  if [[ -f "${REPO_ROOT}/upstream/build/S4" ]]; then
    S4_BIN="${REPO_ROOT}/upstream/build/S4"
  elif [[ -f "${REPO_ROOT}/build/S4" ]]; then
    S4_BIN="${REPO_ROOT}/build/S4"
  else
    S4_BIN="${REPO_ROOT}/upstream/build/S4"
  fi
fi
export S4_BIN
export S4_MATERIALS_DB="${S4_MATERIALS_DB:-${REPO_ROOT}/shared}"
export S4_RUNS="${S4_RUNS:-${REPO_ROOT}/runs}"
export S4_DATA="${S4_DATA:-${REPO_ROOT}/data}"
