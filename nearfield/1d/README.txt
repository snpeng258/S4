近场脚本说明（与 ../flux 中光栅几何、材料一致）

1) au_grating_nearfield.lua
   - S4 GetEField，输出 x-z 网格上 Ex..Ezi 与 |E|（TSV 块 #BEGIN_NEARFIELD_TSV）。
   - 坐标 z 与层厚同单位：周期归一化 L=1，故 z_nm = z_norm * period_nm。
   - 默认：x 为周期倍数坐标 linspace(-x_span_pitch, +x_span_pitch, nx)，
     z 固定为 z_surface_nm（默认 -5 nm），nz=1。
   - use_legacy_xz=1：x 在单周期 [0,1) 上 nx 点，z 在 [z_min_norm, z_max_norm] 上 nz 点。

2) run_s4_nearfield.sh
   - 调用 S4 近场 Lua，将完整 stdout 写入 `runs/nearfield/1d/`：s4_nearfield_wl*..._NG*.txt。
   - 再调用 plot_s4_nearfield_viz.py 生成两张 PNG：同主文件名 .png（空域 4×2）与 _kspace.png。

3) plot_s4_nearfield_viz.py
   - 输入：上述 .txt；解析 #BEGIN_NEARFIELD_TSV … #END_NEARFIELD_TSV。
   - 列：x_norm, z_norm, x_nm, z_nm, Exr, Exi, Eyr, Eyi, Ezr, Ezi, E_mag。

4) s4_total_to_reflected.py + nearfield_reflected_workflow.sh / run_s4_nearfield5.sh
   - 总场减 S4 入射：MakeExcitationPlanewave 的 hx/hy × exp(+i phase)，E = (k×H)/omega0。
   - S4 空域相位为 exp(+i(kx·x_norm + ky·y_norm))（x_norm = x_nm/period_nm）；
     k 图默认再乘 exp(-i kx x_nm) 去载波。
   - carrier_gauge=remove 时在 TSV 中已乘 exp(-i kx x)；绘图须 S4_KSPACE_DEMOD=none。

5) nearfield_reflected_workflow2.sh / run_s4_nearfield6.sh
   - 与 workflow1 相同反射场流程；默认物理约定 psP_teP_cgR_kzM
     （carrier_gauge=remove, z_term_sign=minus）。
   - k 空间：raw FFT。

6) nearfield_total_workflow.sh / run_s4_nearfield7.sh
   - 总场：S4 GetEField TSV 直接绘图，不减入射。
   - 采样默认：x_span_pitch=5，z_surface_nm=-5，nx=1024，nz=1（线切割）。
   - 输出：s4_nearfield_total_wl*..._NG*.txt/.png、*_kspace_raw.png。
   - 斜入射 raw k 图主峰在 kx≈k0·sin θ 属正常（入射载波仍在谱中）。

7) 符号与 flux 读图（../flux/au_grating_reflection.lua）
   - 级次 m = -Gx；反射 R = |back_r|/入射。
   - k 图 demod 后 m 级峰位仍读 kx ≈ -m·2π/Λ（rad/nm），与 flux 的 m 列一致。

8) 环境变量（nearfield_reflected_workflow.sh）
   - S4_PHASE_SIGN：plus（默认，对齐 GetFieldAtPoint +i）
   - S4_CARRIER_GAUGE：none | remove | add（默认 none）
   - S4_KSPACE_DEMOD：minus | plus | none（默认 minus；carrier_gauge=remove 时自动 none）
   - S4_TE_SIGN、S4_Z_TERM_SIGN：见 s4_total_to_reflected.py --help

9) diag_reflected_carrier.py
   - 读 *_reflected.txt：输出 dφ/dx、无/有 demod 的 FFT 主峰 kx
     （60° 验收：demod=minus 时主峰 → 0）。

环境：材料库在仓库根目录 `shared/`（`load_materials.lua`）。工作流会设置 `S4_MATERIALS_DB`。
输出写入 `runs/nearfield/1d/`。
