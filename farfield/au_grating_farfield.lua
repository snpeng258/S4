-- S4 far-field export: same Au grating model as ../flux/au_grating_reflection.lua
-- Exports GetWaves("Air") TSV + flux R_m for validation.
-- Usage: S4 au_grating_farfield.lua -a "wl_nm=13.5;angle_deg=80;azimuth_deg=90;NG=31"

if S4 and S4.arg and type(S4.arg) == "string" then
	local fn, err = loadstring(S4.arg)
	if fn then fn() else error(err or "S4.arg parse error") end
end

wl_nm     = wl_nm     or 13.5
angle_deg   = angle_deg   or 60
azimuth_deg = azimuth_deg or 90
period_nm = period_nm or 80
depth_nm  = depth_nm  or 40
duty      = duty       or 0.5
NG        = NG or 31

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
		error("materials database not found: set S4_MATERIALS_DB or use shared/")
	end
	DATABASE_DIR = (db_dir:sub(-1) == "/") and db_dir or (db_dir .. "/")
	local ok, mod = pcall(dofile, DATABASE_DIR .. "load_materials.lua")
	if not ok or not mod or (not mod.get_epsilon and not mod.get_epsilon_interp) then
		error("load_materials.lua failed: " .. tostring(mod))
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
	if not eps then
		error(string.format("material '%s' missing at wl=%g nm: %s", name, wl_nm, tostring(err)))
	end
	local eps_r, eps_i = eps[1], eps[2]
	if eps_i < 0 then eps_i = -eps_i end
	S:AddMaterial(name, {eps_r, eps_i})
end

add_material("Au")
add_material("SiO2")
S:AddMaterial("Vacuum", {1, 0})

S:AddLayer("Air", 0, "Vacuum")
S:AddLayer("Grating", depth_norm, "Vacuum")
S:SetLayerPatternRectangle("Grating", "Au", {0, 0}, 0, {halfwidth_x, 0.5 / L})
S:AddLayer("Substrate", 0, "SiO2")

S:SetExcitationPlanewave({angle_deg, azimuth_deg}, {1, 0}, {0, 0})
S:SetFrequency(freq)

if not S.GetWaves then
	error("GetWaves not bound in this S4 build; rebuild S4 or use build/S4 with main_lua.c GetWaves")
end

local forw_r, back_r = S:GetPowerFlux("Air", 0)
local incident = forw_r
if incident <= 0 then incident = 1e-20 end

local P = S:GetPowerFluxByOrder("Air", 0)
local G = S:GetGList()
local waves = S:GetWaves("Air")

print(string.format("# S4 far-field waves export | wl=%g nm angle=%g deg azimuth=%g deg period=%g nm depth=%g nm duty=%g NG=%d",
	wl_nm, angle_deg, azimuth_deg, period_nm, depth_nm, duty, NG))
print(string.format("# incident forw_r=%g total_reflection=%g", incident, back_r))

print("#BEGIN_FLUX_TSV")
print("m\tGx\tR_abs")
local rows = {}
for i = 1, #P do
	local ord = P[i]
	local back_r_o = ord[2]
	local gx = (G[i] and G[i][1]) or 0
	local m = -gx
	local R_abs = math.abs(back_r_o) / incident
	table.insert(rows, { m = m, gx = gx, R_abs = R_abs })
end
table.sort(rows, function(a, b) return a.m < b.m end)
for _, r in ipairs(rows) do
	print(string.format("%d\t%d\t%.10g", r.m, r.gx, r.R_abs))
end
print("#END_FLUX_TSV")

print("#BEGIN_WAVES_AIR")
print(string.format("# wl_nm=%g angle_deg=%g azimuth_deg=%g period_nm=%g depth_nm=%g duty=%g NG=%d freq=%g",
	wl_nm, angle_deg, azimuth_deg, period_nm, depth_nm, duty, NG, freq))
print("idx\tkx\tky\tkz_re\tkz_im\tux\tuy\tuz\tcu_re\tcu_im\tcv_re\tcv_im")
for i, w in ipairs(waves) do
	local k, u, cu, cv = w.k, w.u, w.cu, w.cv
	print(string.format(
		"%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g",
		i - 1,
		k[1], k[2], k[3], k[4],
		u[1], u[2], u[3],
		cu[1], cu[2],
		cv[1], cv[2]
	))
end
print("#END_WAVES_AIR")
