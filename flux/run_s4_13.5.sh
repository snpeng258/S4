#!/usr/bin/env bash
# 从任意目录运行：S4 @ 13.5 nm（脚本会切换到 flux/）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_s4_repo_root() {
  local d="$SCRIPT_DIR"
  while [[ -n "$d" && "$d" != "/" ]]; do
    if [[ -f "$d/tools/s4_env.sh" ]]; then
      echo "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  return 1
}

REPO_ROOT="$(find_s4_repo_root)" || { echo "错误: 未找到仓库根目录 (缺少 tools/s4_env.sh)" >&2; exit 1; }
# shellcheck source=tools/s4_env.sh
source "${REPO_ROOT}/tools/s4_env.sh"

cd "$SCRIPT_DIR"
exec "$S4_BIN" au_grating_reflection.lua -a "wl_nm=13.5" "$@"
