-- 材料数据库加载与查询
-- 数据表为介电常数 ε = eps_r + i*eps_i，与 CXRO (henke.lbl.gov) 的 n = 1-δ-iβ 对应 ε = n²。

local materials

-- CXRO 约定: n = 1 - delta - i*beta (δ,β>0 表示吸收)，返回 S4 用 {eps_r, eps_i} = ε = n²
function cxro_to_epsilon(delta, beta)
	local n_r, n_i = 1 - delta, -beta
	local er = n_r*n_r - n_i*n_i
	local ei = 2 * n_r * n_i
	return { er, ei }
end

local function load_db()
	if materials then return materials end
	local dir = DATABASE_DIR or ""
	if dir ~= "" and dir:sub(-1) ~= "/" then dir = dir .. "/" end
	local ok, err = pcall(function()
		materials = dofile(dir .. "materials.lua")
	end)
	if not ok or not materials then
		return nil, err or "materials.lua 加载失败"
	end
	return materials
end

-- 取最接近 wl_nm 的数据行，返回 {eps_r, eps_i} 供 S4 AddMaterial 使用
function get_epsilon(name, wl_nm)
	local db, err = load_db()
	if not db or not db[name] or #db[name] == 0 then
		return nil, err or ("材料 '" .. tostring(name) .. "' 未在 database 中")
	end
	local list = db[name]
	local best = list[1]
	local best_d = math.abs(list[1].wl - wl_nm)
	for i = 2, #list do
		local d = math.abs(list[i].wl - wl_nm)
		if d < best_d then
			best_d = d
			best = list[i]
		end
	end
	return { best.eps_r, best.eps_i }
end

-- 可选: 线性插值
function get_epsilon_interp(name, wl_nm)
	local db, err = load_db()
	if not db or not db[name] or #db[name] == 0 then
		return nil, err or ("材料 '" .. tostring(name) .. "' 未在 database 中")
	end
	local list = db[name]
	if #list == 1 then return { list[1].eps_r, list[1].eps_i } end
	table.sort(list, function(a, b) return a.wl < b.wl end)
	if wl_nm <= list[1].wl then return { list[1].eps_r, list[1].eps_i } end
	if wl_nm >= list[#list].wl then return { list[#list].eps_r, list[#list].eps_i } end
	for i = 1, #list - 1 do
		if wl_nm >= list[i].wl and wl_nm <= list[i+1].wl then
			local t = (wl_nm - list[i].wl) / (list[i+1].wl - list[i].wl)
			local er = list[i].eps_r + t * (list[i+1].eps_r - list[i].eps_r)
			local ei = list[i].eps_i + t * (list[i+1].eps_i - list[i].eps_i)
			return { er, ei }
		end
	end
	return { list[1].eps_r, list[1].eps_i }
end

return { get_epsilon = get_epsilon, get_epsilon_interp = get_epsilon_interp, load_db = load_db, cxro_to_epsilon = cxro_to_epsilon }
