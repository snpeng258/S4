#!/usr/bin/env lua
-- Probe S4 Lua bindings: which methods exist on a Simulation object.
-- Run: ../../../upstream/build/S4 check_s4_lua_api.lua

local function check_api(name, fn)
    if fn then
        print(string.format("%-24s loaded  ->  %s", name, tostring(fn)))
        return true
    end
    print(string.format("%-24s MISSING", name))
    return false
end

print("=== S4 module (global) ===")
if not S4 then
    error("S4 module not loaded (run this script with the S4 executable, not system lua)")
end

check_api("S4.NewSimulation", S4.NewSimulation)
check_api("S4.New", S4.New)
check_api("S4.GetGList", S4.GetGList)

print("\n=== Simulation object ===")
local s = S4.NewSimulation()
if not s then
    error("S4.NewSimulation() returned nil")
end
print("simulation type:", type(s))

-- User's original check (GetBasis)
if s.GetBasis then
    print("GetBasis 接口已加载，函数地址为:", s.GetBasis)
else
    print("GetBasis: 当前 Lua 绑定中不存在（Python 侧有 GetBasisSet，Lua 用 GetGList / GetWaves）")
end

print("\n--- k / basis related ---")
check_api("GetBasis", s.GetBasis)
check_api("GetBasisSet", s.GetBasisSet)
check_api("GetGList", s.GetGList)
check_api("GetWaves", s.GetWaves)
check_api("GetReciprocalLattice", s.GetReciprocalLattice)
check_api("GetDiffractionOrder", s.GetDiffractionOrder)
check_api("GetAmplitudes", s.GetAmplitudes)

print("\n--- near field ---")
check_api("GetEField", s.GetEField)
check_api("GetFieldPlane", s.GetFieldPlane)

print("\nDone.")
