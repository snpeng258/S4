-- S4 脚本: Au 光栅 on SiO2 基底, 1D, 各级次反射率
-- 材料必须来自 shared/（无默认值，缺数据则报错）。
-- 命令行 -a "wl_nm=13.5" 等会在此解析并覆盖下方默认值。
if S4 and S4.arg and type(S4.arg) == "string" then
	local fn, err = loadstring(S4.arg)
	if fn then fn() else error(err or "S4.arg parse error") end
end

wl_nm     = wl_nm     or 13.5   -- 波长 (nm)
angle_deg = angle_deg or 80    -- 入射角 (度, 相对法线)
period_nm = period_nm or 80    -- 周期 (nm)
depth_nm  = depth_nm  or 40    -- 槽深 (nm)
duty      = duty       or 0.5  -- 占空比 (线宽/周期)

-- 数据库路径: 优先环境变量，否则相对当前工作目录尝试常见位置
local function find_database_dir()
	local env = os.getenv("S4_MATERIALS_DB")
	if env and env ~= "" then
		return env
	end
	local function probe(path)
		local f = io.open(path .. "/load_materials.lua", "r")
		if f then f:close(); return true end
		return false
	end
	local dir = "."
	for _ = 1, 10 do
		if probe(dir .. "/shared") then
			return dir .. "/shared"
		end
		dir = dir .. "/.."
	end
	return nil
end

local get_epsilon_from_db
do
	local db_dir = find_database_dir()
	if not db_dir then
		error("未找到材料数据库: 请在 shared/ 下提供 materials.lua，或设置环境变量 S4_MATERIALS_DB 指向该目录")
	end
	DATABASE_DIR = (db_dir:sub(-1) == "/") and db_dir or (db_dir .. "/")
	local ok, mod = pcall(dofile, DATABASE_DIR .. "load_materials.lua")
	if not ok or not mod or (not mod.get_epsilon and not mod.get_epsilon_interp) then
		error("材料数据库加载失败: " .. tostring(mod or "load_materials.lua 未返回 get_epsilon/get_epsilon_interp"))
	end
	-- 默认改为线性插值；若旧版本无插值接口则回退到最近邻
	get_epsilon_from_db = mod.get_epsilon_interp or mod.get_epsilon
end

-- 归一化
L = 1
depth_norm = depth_nm / period_nm
halfwidth_x = 0.5 * duty
freq = (period_nm / wl_nm)

-- Fourier 截断 (奇数)。可通过 -a "NG=31" 覆盖。
NG = NG or 31
S = S4.NewSimulation()
S:SetLattice({L, 0}, {0, 0})
S:SetNumG(NG)

-- 材料: 仅从 database 读取，无则报错（不使用默认介电常数）
-- S4 要求: 吸收材料 Im(ε) > 0（见 S4 文档 AddMaterial）。数据库若为 CXRO 约定（EUV 下 Im(ε)<0），
-- 则需取共轭再传入 S4，即传入 {eps_r, -eps_i}，使 S4 得到 Im(ε)>0。
local function add_material(name)
	local eps, err = get_epsilon_from_db(name, wl_nm)
	if not eps then
		error(string.format("材料 '%s' 在波长 %g nm 下无数据: %s。请在 database/materials.lua 中补充，可参考 CXRO (henke.lbl.gov) 的 n=1-δ-iβ 换算 ε=n²。", name, wl_nm, tostring(err)))
	end
	local eps_r, eps_i = eps[1], eps[2]
	if eps_i < 0 then
		eps_i = -eps_i
	end
	S:AddMaterial(name, {eps_r, eps_i})
end

add_material("Au")
add_material("SiO2")
S:AddMaterial("Vacuum", {1, 0})

S:AddLayer("Air", 0, "Vacuum")
S:AddLayer("Grating", depth_norm, "Vacuum")
S:SetLayerPatternRectangle("Grating", "Au", {0, 0}, 0, {halfwidth_x, 0.5/L})
S:AddLayer("Substrate", 0, "SiO2")

S:SetExcitationPlanewave({angle_deg, 0}, {1, 0}, {0, 0})
S:SetFrequency(freq)

local forw_r, back_r, forw_i, back_i = S:GetPowerFlux("Air", 0)
local incident = forw_r
if incident <= 0 then incident = 1e-20 end

local P = S:GetPowerFluxByOrder("Air", 0)
local G = S:GetGList()

--[[
  级次与符号约定:
  - S4 内部 1D G 矢量顺序 (S4.cpp SetBases): Gx = 0, +1, -1, +2, -2, ... (仅存储顺序)。
  - 常用 RCWA 约定 k_x,m = k0*n*sinθ - m*(2π/Λ)。同一束物理衍射光满足 m = -Gx。
    故输出列 m := -Gx，按 m 升序排列。
  - 反射率: S4 反射侧 z 向 Poynting 常为负 (透射为正)。R_abs = |back_r|/入射；
    各阶 back_r 带符号求和仍用于能量闭合。
]]
print(string.format("# 波长=%g nm  入射角=%g deg  周期=%g nm  槽深=%g nm  占空比=%g  NG=%d", wl_nm, angle_deg, period_nm, depth_nm, duty, NG))
print(string.format("# 入射 (forw_r): %g  总反射 back_r(有符号): %g", incident, back_r))
print("# m=-Gx; Gx=S4 GetGList; R_abs=|back_r|/入射")
print("m\tGx\tR_abs\tback_r\tback_i")

local rows = {}
local sum_back_r = 0
for i = 1, #P do
	local ord = P[i]
	local back_r_o, back_i_o = ord[2], ord[4]
	local gx = (G[i] and G[i][1]) or 0
	local m = -gx
	sum_back_r = sum_back_r + back_r_o
	table.insert(rows, { m = m, gx = gx, br = back_r_o, bi = back_i_o })
end
table.sort(rows, function(a, b) return a.m < b.m end)

for _, r in ipairs(rows) do
	local R_abs = math.abs(r.br) / incident
	print(string.format("%d\t%d\t%.6g\t%.6g\t%.6g", r.m, r.gx, R_abs, r.br, r.bi))
end

local R_total_physical = math.abs(back_r) / incident
print(string.format("# 各阶 back_r 之和(有符号, 用于能量): %g", sum_back_r))
print(string.format("# 总反射率(绝对值): %.6g", R_total_physical))
if R_total_physical > 1 then
	print("# 注意: 总反射率>1 可能为材料/数值问题")
end

-- 机器可读块: m<TAB>R_abs（inverse/s4_runner.py 等解析）
print("#BEGIN_COMPARE_TSV")
for _, r in ipairs(rows) do
	local R_abs = math.abs(r.br) / incident
	print(string.format("%d\t%.10f", r.m, R_abs))
end
print("#END_COMPARE_TSV")
