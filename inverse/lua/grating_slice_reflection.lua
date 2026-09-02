-- Multi-slice Au grating on SiO2 (layer order matches au_grating_reflection.lua: Air on top).
if S4 and S4.arg and type(S4.arg) == "string" then
	local fn, err = loadstring(S4.arg)
	if fn then fn() else error(err or "S4.arg parse error") end
end

wl_nm       = wl_nm       or 13.5
angle_deg   = angle_deg   or 70
azimuth_deg = azimuth_deg or 90
period_nm   = period_nm   or 80
NG          = NG          or 31
n_slices    = n_slices    or 1
duties      = duties      or "0.5"
offsets     = offsets     or "0"
slice_thicknesses = slice_thicknesses or "0.05"
grating_mat = grating_mat or "Au"
substrate_mat = substrate_mat or "SiO2"
pol_s_amp   = pol_s_amp   or 1
pol_s_phase = pol_s_phase or 0
pol_p_amp   = pol_p_amp   or 0
pol_p_phase = pol_p_phase or 0

local function split_csv(s)
	local t = {}
	for part in string.gmatch(s, "[^,]+") do
		table.insert(t, tonumber(part))
	end
	return t
end

local duty_list = split_csv(duties)
local offset_list = split_csv(offsets)
local thick_list = split_csv(slice_thicknesses)

if #duty_list ~= n_slices or #offset_list ~= n_slices or #thick_list ~= n_slices then
	error(string.format(
		"slice length mismatch: n_slices=%d duties=%d offsets=%d thicknesses=%d",
		n_slices, #duty_list, #offset_list, #thick_list))
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
	if not db_dir then error("materials database not found") end
	DATABASE_DIR = (db_dir:sub(-1) == "/") and db_dir or (db_dir .. "/")
	local ok, mod = pcall(dofile, DATABASE_DIR .. "load_materials.lua")
	if not ok or not mod then error("load_materials.lua failed") end
	get_epsilon_from_db = mod.get_epsilon_interp or mod.get_epsilon
end

local L = 1
local freq = period_nm / wl_nm
local S = S4.NewSimulation()
S:SetLattice({L, 0}, {0, 0})
S:SetNumG(NG)

local function add_material(name)
	local eps = get_epsilon_from_db(name, wl_nm)
	if not eps then error("material missing: " .. name) end
	local eps_r, eps_i = eps[1], eps[2]
	if eps_i < 0 then eps_i = -eps_i end
	S:AddMaterial(name, {eps_r, eps_i})
end

add_material(grating_mat)
add_material(substrate_mat)
S:AddMaterial("Vacuum", {1, 0})

S:AddLayer("Air", 0, "Vacuum")
for i = 1, n_slices do
	local lname = string.format("Slice%d", i)
	S:AddLayer(lname, thick_list[i], "Vacuum")
	local halfw = 0.5 * duty_list[i]
	S:SetLayerPatternRectangle(lname, grating_mat, {offset_list[i], 0}, 0, {halfw, 0.5 / L})
end
S:AddLayer("Substrate", 0, substrate_mat)

S:SetExcitationPlanewave({angle_deg, azimuth_deg}, {pol_s_amp, pol_s_phase}, {pol_p_amp, pol_p_phase})
S:SetFrequency(freq)

local forw_r, back_r = S:GetPowerFlux("Air", 0)
local incident = forw_r
if incident <= 0 then incident = 1e-20 end

local P = S:GetPowerFluxByOrder("Air", 0)
local G = S:GetGList()
local rows = {}
for i = 1, #P do
	local ord = P[i]
	local gx = (G[i] and G[i][1]) or 0
	table.insert(rows, { m = -gx, R_abs = math.abs(ord[2]) / incident })
end
table.sort(rows, function(a, b) return a.m < b.m end)

print(string.format("# wl=%g angle=%g az=%g period=%g n_slices=%d NG=%d incident=%g R_total=%g",
	wl_nm, angle_deg, azimuth_deg, period_nm, n_slices, NG, incident, math.abs(back_r)/incident))
print("#BEGIN_COMPARE_TSV")
print("m\tR_abs")
for _, r in ipairs(rows) do
	print(string.format("%d\t%.10g", r.m, r.R_abs))
end
print("#END_COMPARE_TSV")
