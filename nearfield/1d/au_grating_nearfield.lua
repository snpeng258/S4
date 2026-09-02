-- S4: Au 光栅 on SiO2 衬底；光栅层槽区为 Vacuum（空气），条带为 Au；本脚本输出空间电场采样。
-- 几何与 ../flux/au_grating_reflection.lua 一致。
--
-- 两种取点模式（由 -a 中 use_legacy_xz 切换）：
--   默认（表面线切割）：
--     x 为周期倍数坐标 linspace(-x_span_pitch, +x_span_pitch, nx)
--     （物理 x_nm = x * period_nm）；z 固定为 z_surface_nm（默认 -5 nm），nz=1；y=0。
--   use_legacy_xz=1（旧版 x–z 扫描）：
--     x 在单周期内 [0,1) 上 nx 点；z 在 [z_min_norm, z_max_norm] 上 nz 点。
--
-- 命令行 -a "wl_nm=13.5;nx=64;..." 可覆盖参数。
if S4 and S4.arg and type(S4.arg) == "string" then
	local fn, err = loadstring(S4.arg)
	if fn then fn() else error(err or "S4.arg parse error") end
end

local function truthy(v)
	if v == true or v == 1 then return true end
	if type(v) == "string" then
		local s = v:lower()
		if s == "1" or s == "true" or s == "yes" then return true end
	end
	return false
end

wl_nm     = wl_nm     or 13.5
angle_deg = angle_deg or 80
period_nm = period_nm or 80
depth_nm  = depth_nm  or 40
duty      = duty       or 0.5
NG        = NG or 31

local legacy_xz = truthy(use_legacy_xz)

if legacy_xz then
	nx = nx or 32
	nz = nz or 24
	z_min_norm = z_min_norm or -0.2
	z_max_norm = z_max_norm or (depth_nm / period_nm + 0.2)
else
	x_span_pitch = x_span_pitch or 5
	nx = nx or 64
	nz = nz or 1
	z_surface_nm = z_surface_nm or -5
	if nz ~= 1 then
		error("非 legacy 模式仅支持 nz=1；多 z 扫描请设 use_legacy_xz=1")
	end
end

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
halfwidth_x = 0.5 * duty
freq = (period_nm / wl_nm)

local S = S4.NewSimulation()
S:SetLattice({L, 0}, {0, 0})
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
S:AddLayer("Grating", depth_norm, "Vacuum")
S:SetLayerPatternRectangle("Grating", "Au", {0, 0}, 0, {halfwidth_x, 0.5 / L})
S:AddLayer("Substrate", 0, "SiO2")

S:SetExcitationPlanewave({angle_deg, 0}, {1, 0}, {0, 0})
S:SetFrequency(freq)

print(string.format("# S4 近场 | wl=%g nm angle=%g deg period=%g nm depth=%g nm duty=%g NG=%d",
	wl_nm, angle_deg, period_nm, depth_nm, duty, NG))
if legacy_xz then
	print(string.format("# 模式 legacy_xz | nx=%d nz=%d z_norm in [%.6g, %.6g] | x in [0,1) 周期内",
		nx, nz, z_min_norm, z_max_norm))
	print(string.format("# 网格 (x_nm=x_norm*%g, z_nm=z_norm*%g)", period_nm, period_nm))
else
	print(string.format("# 模式 surface_line | nx=%d nz=%d x_span_pitch=%g z_surface_nm=%g",
		nx, nz, x_span_pitch, z_surface_nm))
	print(string.format("# x_norm: linspace(-span,+span); z_norm=z_surface_nm/period=%.6g",
		z_surface_nm / period_nm))
end
print("#BEGIN_NEARFIELD_TSV")
print("x_norm\tz_norm\tx_nm\tz_nm\tExr\tExi\tEyr\tEyi\tEzr\tEzi\tE_mag")

if legacy_xz then
	local dx = (nx > 1) and (1.0 / nx) or 0
	local dz = (nz > 1) and ((z_max_norm - z_min_norm) / (nz - 1)) or 0
	for iz = 1, nz do
		local z = z_min_norm + (iz - 1) * dz
		for ix = 1, nx do
			local x = (ix - 1) * dx
			local Exr, Eyr, Ezr, Exi, Eyi, Ezi = S:GetEField({x, 0, z})
			local x_nm = x * period_nm
			local z_nm = z * period_nm
			local emag = math.sqrt(Exr * Exr + Exi * Exi + Eyr * Eyr + Eyi * Eyi + Ezr * Ezr + Ezi * Ezi)
			print(string.format("%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g",
				x, z, x_nm, z_nm, Exr, Exi, Eyr, Eyi, Ezr, Ezi, emag))
		end
	end
else
	local x_lo, x_hi = -x_span_pitch, x_span_pitch
	local dx = (nx > 1) and ((x_hi - x_lo) / (nx - 1)) or 0
	local z = z_surface_nm / period_nm
	for ix = 1, nx do
		local x = x_lo + (ix - 1) * dx
		local Exr, Eyr, Ezr, Exi, Eyi, Ezi = S:GetEField({x, 0, z})
		local x_nm = x * period_nm
		local z_nm = z * period_nm
		local emag = math.sqrt(Exr * Exr + Exi * Exi + Eyr * Eyr + Eyi * Eyi + Ezr * Ezr + Ezi * Ezi)
		print(string.format("%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g",
			x, z, x_nm, z_nm, Exr, Exi, Eyr, Eyi, Ezr, Ezi, emag))
	end
end
print("#END_NEARFIELD_TSV")
