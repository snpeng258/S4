-- S4: 二维方格周期 — SiO2 上 Au 正方形突起阵列（与 1d au_grating 相同材料与归一化），输出固定 z 上 x–y 电场。
-- 参数与一维脚本同源，但含义为「二维正方格」而非一维条带：
--   period_nm  — x、y 方向周期相同，均为该值（原胞边长 Lx = Ly = period_nm）。
--   duty       — 归一化后边长：每个原胞中心一个 Au 正方形，边长 = duty × period_nm（x、y 等长），
--                即一维中「沿周期方向的条带宽度 = duty×周期」在二维的推广为「正方形边长」。
--   depth_nm   — 刻蚀/突起深度（与一维相同）；depth_norm = depth_nm/period_nm。
-- 无限周期阵列在 x–y 上表现为多个相同正方形突起重复排列（每原胞一个）。
if S4 and S4.arg and type(S4.arg) == "string" then
	local fn, err = loadstring(S4.arg)
	if fn then fn() else error(err or "S4.arg parse error") end
end

wl_nm      = wl_nm      or 13.5
angle_deg  = angle_deg  or 80
period_nm  = period_nm  or 80
depth_nm   = depth_nm   or 40
duty       = duty       or 0.5
rcwa_order = rcwa_order or 15
nx         = nx         or 32
ny         = ny         or 32
z_plane_norm = z_plane_norm or -0.05

local n1 = 2 * rcwa_order + 1
NG = NG or (n1 * n1)

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
		error("未找到材料数据库: 设置 S4_MATERIALS_DB 或在 shared/ 提供 load_materials.lua")
	end
	DATABASE_DIR = (db_dir:sub(-1) == "/") and db_dir or (db_dir .. "/")
	local ok, mod = pcall(dofile, DATABASE_DIR .. "load_materials.lua")
	if not ok or not mod or (not mod.get_epsilon and not mod.get_epsilon_interp) then
		error("load_materials.lua 加载失败")
	end
	get_epsilon_from_db = mod.get_epsilon_interp or mod.get_epsilon
end

L = 1
depth_norm = depth_nm / period_nm
-- 正方形突起：半宽 (half_x, half_y) = (duty/2, duty/2)，边长 = duty（归一化）= duty*period_nm（nm）
half_xy = 0.5 * duty
freq = (period_nm / wl_nm)

local S = S4.NewSimulation()
S:SetLattice({L, 0}, {0, L})
S:SetNumG(NG)

local function add_material(name)
	local eps, err = get_epsilon_from_db(name, wl_nm)
	if not eps then error("材料 " .. name .. ": " .. tostring(err)) end
	local er, ei = eps[1], eps[2]
	if ei < 0 then ei = -ei end
	S:AddMaterial(name, {er, ei})
end

add_material("Au")
add_material("SiO2")
S:AddMaterial("Vacuum", {1, 0})

S:AddLayer("Air", 0, "Vacuum")
S:AddLayer("Grating", depth_norm, "SiO2")
S:SetLayerPatternRectangle("Grating", "Au", {0, 0}, 0, {half_xy, half_xy})
S:AddLayer("Substrate", 0, "SiO2")

S:SetExcitationPlanewave({angle_deg, 0}, {1, 0}, {0, 0})
S:SetFrequency(freq)

print(string.format("# S4 2D 近场 (x–y, z_norm=%.6g) | Lx=Ly=%g nm 正方形Au边长=%g nm | wl=%g nm angle=%g deg depth=%g nm duty=%g rcwa_order=%d NG=%d",
	z_plane_norm, period_nm, duty * period_nm, wl_nm, angle_deg, depth_nm, duty, rcwa_order, NG))
print(string.format("# 网格 nx=%d ny=%d  z_plane_norm=%.6g  (x_nm=x*Lx, y_nm=y*Ly, Lx=Ly=%g nm)",
	nx, ny, z_plane_norm, period_nm))
print("#BEGIN_NEARFIELD_TSV")
print("x_norm\ty_norm\tx_nm\ty_nm\tExr\tExi\tEyr\tEyi\tEzr\tEzi\tE_mag")

local dx = (nx > 1) and (1.0 / nx) or 0
local dy = (ny > 1) and (1.0 / ny) or 0

for iy = 1, ny do
	local y = (iy - 1) * dy
	for ix = 1, nx do
		local x = (ix - 1) * dx
		local Exr, Eyr, Ezr, Exi, Eyi, Ezi = S:GetEField({x, y, z_plane_norm})
		local x_nm = x * period_nm
		local y_nm = y * period_nm
		local emag = math.sqrt(Exr * Exr + Exi * Exi + Eyr * Eyr + Eyi * Eyi + Ezr * Ezr + Ezi * Ezi)
		print(string.format("%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g",
			x, y, x_nm, y_nm, Exr, Exi, Eyr, Eyi, Ezr, Ezi, emag))
	end
end
print("#END_NEARFIELD_TSV")
