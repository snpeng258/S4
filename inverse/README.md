# S4 Scatterometry 逆问题框架

基于 S4 RCWA 的散射测量（scatterometry）正向/逆向求解。

## 依赖

- Python 3.10+：`numpy`, `scipy`, `matplotlib`, `pyyaml`
- 已编译 `upstream/build/S4`

## 配置

**单一配置文件 [`config.yaml`](config.yaml)**：建库、扫描、逆问题、evaluate 共用。通过 `eval.task` / `eval.mode` 切换任务（见文件头注释）。

| 段 | 用途 |
|----|------|
| `structure` | pitch, cd, depth, lswa, rswa, n_slices |
| `optical` | 偏振、NG、级次；`recipe`（HHG 多波长/多角度） |
| `inverse` | `method`、`order_collection`、`decoupling`、GA/LM 参数 |
| `library` | 光谱库 grid、prior、`build_workers` |
| `scan` | 扫描轴、Jacobian；可选 `scan.recipe`（与逆问题独立） |
| `eval` | **`task`**（inverse / scan_sweep）、**`mode`**（批量评价） |

### eval.task 与 eval.mode

| 字段 | 入口 | 含义 |
|------|------|------|
| `eval.task: inverse` | `inverse_solver.py` / `evaluate.py` | 逆问题 GA+LM |
| `eval.task: scan_sweep` | 同上 | 多参数正向扫描 |
| `eval.mode` | 仅 `evaluate.py` 且 `task=inverse` | methods / noise / timing / ga_workers / all |

### 噪声模型（合成测量 + WLS 权重）

零级（`inverse.noise.zero`，DoBEAM2000-2）与 \(\pm 1\)（`inverse.noise.first`，XV4040BSI High Gain）分相机。合成测量对光电子做泊松抽样，暗电流泊松后减均值，再加读出高斯。WLS 用对应方差：

\[
\sigma_j^2=(a f_j)^2 + f_j/N_0 + b^2
\]

零级保留 \(a=1\%\)；\(\pm 1\) 的 \(a=0\)。反演权重 \(1/\sigma_j\)（再乘解耦角色因子）。`eval.mode: noise` 扫描 `eval.noise_n0_electrons`。无噪声：`inverse.noise.apply: false`。

### HHG recipe（当前默认已启用）

- 两 band 奇次谐波，方案 E：8 入射角 × 5 方位角 → **280 光学条件**
- 库文件：`../data/inverse/spectra_hhg.npz`（建库 recipe 须为超集；逆问题可用子集 λ/θ/φ，自动列切片）
- 建库存全 31 级次；逆问题默认 `order_collection: propagating`

### order_collection: decoupling（φ–m 解耦）

当 `inverse.order_collection: decoupling` 时，观测量按方位角–衍射级次**角色**筛选（非全部 propagating）：

| 角色 | 典型 (φ, m) | 主要参数 |
|------|-------------|----------|
| `depth_anchor` | φ≈90°, m=0 | depth_nm |
| `swa` | φ≈90°, m=±1 | lswa/rswa（若可传播） |
| `lateral` | 非 90° 多 φ, m=0 | cd_nm（LM 中按 J 动态细分） |

- **GA / 库匹配**：`static_weights` × 探测器 \(1/\sigma\) 权重（或旧 SNR 启发式）
- **LM**：外层周期性有限差分 Jacobian → 动态 `wsqrt` + CD 软正则（`cd_anchor: ga`）
- 诊断：`p_est.json` 的 `lm_weight_history`（耦合比 r、λ_cd 等）

配置见 [`config.yaml`](config.yaml) 的 `inverse.decoupling` 段。启用：`order_collection: decoupling`，并将 `optical.recipe.azimuths_deg` 扩展为含 90° 与若干非 90° 角。

单波长快速试跑：注释 `optical.recipe`，改 `library.file` → `../data/inverse/spectra.npz`（见 `config.yaml` 注释）。

### scan.recipe（扫描专用，可选）

`scan_sweep` **不读光谱库**。若配置 `scan.recipe`，扫描仅用该段；否则回退 `optical.recipe`。  
逆问题/建库始终用 `optical.recipe`，互不影响。

```yaml
scan:
  axes: [wavelength, angle]
  recipe:
    fundamental_nm: 800
    harmonic_order_ranges:
      - {min: 27, max: 31, odd_only: true}
      - {min: 55, max: 61, odd_only: true}
    angles_deg: [69, 70, 71]    # 可窄于 optical.recipe
    azimuths_deg: [0, 90]
```

`eval.task: scan_sweep` 后运行 `python3 evaluate.py`。

## 使用

```bash
cd inverse

python3 forward_model.py
python3 build_library.py          # 或 --resume
python3 inverse_solver.py         # 读 eval.task
python3 evaluate.py               # eval.mode=methods / noise / ...
```

## 输出

- `../runs/inverse/scan_sweep.json` — 扫描网格结果
- `../data/inverse/spectra_hhg.npz` — HHG 形貌–光谱库
- `../runs/inverse/p_est.json` / `eval_*.json` — 反演与评价
- `../runs/inverse/build_benchmark.json` — 建库并行测速结果
