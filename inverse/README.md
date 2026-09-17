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
| `eval` | **`task`**（inverse / scan_sweep / fim_study）、**`mode`**（批量评价） |
| `fim` | 一层雅可比 / FIM 的掩膜与 \(N_0\)/\(a\) 扫描（见 [`config_fim.yaml`](config_fim.yaml)） |

### eval.task 与 eval.mode

| 字段 | 入口 | 含义 |
|------|------|------|
| `eval.task: inverse` | `inverse_solver.py` / `evaluate.py` | 逆问题 GA+LM |
| `eval.task: scan_sweep` | 同上 | 多参数正向扫描 |
| `eval.task: fim_study` | `fim_study.py` / `evaluate.py` | 一张全级次 \(J\) + 行掩膜 FIM / CRLB |
| `eval.mode` | 仅 `evaluate.py` 且 `task=inverse` | methods / noise / timing / ga_workers / all |

### 噪声模型（合成测量 + WLS 权重）

零级（`inverse.noise.zero`，DoBEAM2000-2）与 \(\pm 1\)（`inverse.noise.first`，XV4040BSI High Gain）分相机。合成测量对光电子做泊松抽样，暗电流泊松后减均值，再加读出高斯。WLS 用对应方差：

\[
\sigma_j^2=(a f_j)^2 + f_j/N_0 + b^2
\]

零级与 \(\pm 1\) 共用入射光功率抖动 \(a=1\%\)。零级读出暂按 \(5\,\mathrm{e^-}\)。反演权重 \(1/\sigma_j\)（再乘解耦角色因子）。`eval.mode: noise` 扫描 `eval.noise_n0_electrons`。无噪声：`inverse.noise.apply: false`。

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

### 一层雅可比 / FIM（`fim_study`）

在真值结构上对**全存储级次**做一次有限差分 \(J\)（不读谱库、不跑 GA+LM），再用行掩膜比较布局。\(\Sigma\) 用与反演相同的分相机 `observation_variance`（**不含**解耦角色权重）。

```bash
cd inverse
python3 fim_study.py --config config_fim.yaml --layout-only   # 只打印 (φ,m) 传播表，不调用 S4
python3 fim_study.py --config config_fim.yaml                # ~100 次 S4
```

输出在 `../runs/inverse/fim_study/`（gitignore）。FIM 与出图默认只留可传播的 \(m\in\{-1,0,1\}\)（`fim.keep_orders`）：`fim_R_lambda_phi.png` 以及每个级次的 \(\partial R/\partial p\)、白化图（\((\lambda,\varphi)\) 平面）。掩膜：`prop`（可传播的 0/±1）、`decoupling`、`m0_all`、`only90`、`no90`、`near90`、`far`、`mid`、`two_cam`、`drop_phi45`。同一张 \(J\) 上再扫 \(N_0\) 与 \(a\)。也可 `python3 evaluate.py --config config_fim.yaml`。S4 的 `NG` 不变，屏蔽高级次只影响分析和图，不加快求解。

密采样（\(\lambda=10\)–\(30\,\mathrm{nm}\)、步长 1 nm；\(\varphi=0\)–\(90^\circ\)、步长 \(5^\circ\)；depth 40 nm、占空比 0.5）用 `run_fim_dense.py`：80 nm 组 `NG=31`，300 nm 组 `NG=61`。每组约 399 个条件、1995 次 S4。先 `--layout-only` 看传播表。

## 使用

```bash
cd inverse

python3 forward_model.py
python3 build_library.py          # 或 --resume
python3 inverse_solver.py         # 读 eval.task
python3 evaluate.py               # eval.mode=methods / noise / ...
python3 fim_study.py --config config_fim.yaml --layout-only
python3 fim_study.py --config config_fim.yaml
python3 run_fim_dense.py --layout-only
python3 run_fim_dense.py
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode layout
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask decoupling --mode A
```

紧凑 recipe 上把 FIM/CRLB 和带噪反演对上：见 [`docs/CRLB_MC.md`](docs/CRLB_MC.md)。服务器上按该文档开 `tmux` 即可，不必同步原对话。

## 输出

- `../runs/inverse/scan_sweep.json` — 扫描网格结果
- `../data/inverse/spectra_hhg.npz` — HHG 形貌–光谱库
- `../runs/inverse/p_est.json` / `eval_*.json` — 反演与评价
- `../runs/inverse/build_benchmark.json` — 建库并行测速结果
