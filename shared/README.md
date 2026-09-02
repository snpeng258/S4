# 材料介电常数（共用）

远场、近场、通量、逆问题都读这一份，不要再复制到功能目录里。

## 内容

- `Au.txt` `SiO2.txt` `si.txt` `Mo.txt` `TaN.txt` `Si3N4.txt` — CXRO 光学常数表
- `materials.lua` / `load_materials.lua` — S4 Lua 插值接口（ε(λ)）

## 约定

表内存的是介电常数 ε，不是折射率 n。CXRO 用 n = 1 − δ − iβ，ε = n²。
S4 的 `AddMaterial` 要求 Im(ε) > 0；加载时若表中虚部为负会取共轭。

## 怎么被找到

1. 环境变量 `S4_MATERIALS_DB` 指向本目录（工作流脚本会自动设置）
2. 否则 Lua 从当前工作目录向上找 `shared/load_materials.lua`
3. 逆问题配置：`paths.materials_db: ../shared`

补材料：在本目录加表，并写入 `materials.lua`。
