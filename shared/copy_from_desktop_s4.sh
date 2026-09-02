#!/usr/bin/env bash
# 从 Windows 桌面 s4 目录拷贝材料数据到当前 shared/ 目录
# 在 WSL 中运行: bash copy_from_desktop_s4.sh

WIN_S4="/mnt/c/Users/PSN/Desktop/s4"
DB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$WIN_S4" ]]; then
  echo "未找到: $WIN_S4"
  echo "请确认 C:\\Users\\PSN\\Desktop\\s4 存在，或在脚本中修改 WIN_S4 路径"
  exit 1
fi

echo "拷贝: $WIN_S4 -> $DB_DIR"
cp -rv "$WIN_S4"/* "$DB_DIR"/ 2>/dev/null || cp -v "$WIN_S4"/* "$DB_DIR"/
echo "完成。请将拷贝后的表格数据按 materials.lua 格式整理到 materials.lua 中。"
