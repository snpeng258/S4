# 紧凑 recipe 上的 χ² 全局景观

对应 S4-8（Gross 等 2009）Fig. 4：固定一份测量，把**反演优化目标**铺在两个形貌参数上。不跑 GA/LM。左右墙已统一为 `swa_deg`。

和 CRLB–MC **可以并行**，但必须写到不同目录，也不要抢光核。

## 在画什么

默认切片钉死真值侧壁角，扫 CD × depth（Fig. 4 钉死 \(p_7\)、扫 \(p_2\)–\(p_6\)）。`--slice cd_swa` 则钉死 depth，扫 CD × SWA。

测量先用真值无噪声 \(R_{\mathrm{meas}}\)。Z 轴就是求解器的数据项（GA `_loss` / LM `lm_history`）：

\[
L(p)=\sum_j w_j\bigl(R_j(p)-R_{\mathrm{meas},j}\bigr)^2,\qquad w_j\propto 1/\sigma_j^2,\ \sum_j w_j=1
\]

`wsqrt` 与 `InverseProblem` 相同：默认 `use_role_weights: false` 时走 `snr_weights` → `inverse_sigma_weights`（先 \(1/\sigma\) 再归一化）。把 yaml 里的角色权重打开且 mask 为 `decoupling` 时，改走 `static_role_weights`。不含 LM 动态 CD 软正则，也不用未加权的 `resnorm`。正演存全套级次，事后再套 `prop` / `decoupling` / `m0_all` / `only90`。

## 配置

| | 80 nm 方槽 | 300 nm 矮槽 | 300 nm 方槽 |
|---|---|---|---|
| 文件 | `config_chi2_p80.yaml` | `config_chi2_p300.yaml` | `config_chi2_p300_d150.yaml` |
| 真值 | CD 40 / depth 40 / SWA \(89.45^\circ\) | CD 150 / depth 40 / SWA \(89.45^\circ\) | CD 150 / depth 150 / SWA \(89.45^\circ\) |
| 深宽比 | 1 | 0.27 | 1 |
| `NG` | 31 | 61 | 61 |
| 默认网格 | CD、depth 各 32–48、步长 1 → \(17\times17=289\) | CD 134–166 步长 2、depth 32–48 步长 1 → \(17\times17=289\) | CD 134–166 步长 2、depth 142–158 步长 1 → \(17\times17=289\) |
| 建议 workers | 8 | 4 | 4 |
| 无噪声输出 | `../runs/inverse/chi2_landscape/p80/` | `../runs/inverse/chi2_landscape/p300/` | `../runs/inverse/chi2_landscape/p300_d150/` |
| 有噪声输出 | `.../p80_noisy/` | `.../p300_noisy/` | `.../p300_d150_noisy/` |
| S4 次数（约） | \(289\times20=5780\) | \(289\times20=5780\) | \(289\times20=5780\) |

`p300_d150` 只把槽深从 40 nm 升到 150 nm，周期、CD、占空比、`NG`、光学 recipe 与 `p300` 相同。用来看 `only90` 沿 CD 变扁是不是深宽比造成的，而不是周期本身。`n_slices` 仍是 10（每层更厚），不要和 80 nm 那组的切片厚度混谈。

光学与 CRLB–MC 相同：H55–61 奇次 × \(\theta=70^\circ\) × \(\varphi\in\{0,30,45,60,90\}\)。

## 怎么跑

不要和 `crlb_mc/p80`、`crlb_mc/p300` 写到一起。CRLB 的 A 若已经占了 4×8 核，80 nm 景观用 8 个形貌进程，300 nm 用 4 个。每个进程的 `condition_workers` 固定为 1。

主实验仍是 **2 周期 × 2 噪声 = 4 次**。方槽对照是另外 2 次（`p300_d150` 无噪声 / `--noisy`），不要和 `p300` 写进同一目录。线性主图仍是每张 `chi2_cd_depth_decoupling.png` 和 `chi2_cd_depth_only90.png`（左等高线、右曲面）。对数等高线是同目录的 `*_log.png`：圈距按 \(\log L\)，用来看 loss 掉到 \(10^{-6}\)～\(10^{-7}\) 那一截；线性那版不删、不改名。对照时先看 `only90`：若 300 nm 方槽的等高线从横沟收成扁椭圆，才支持「深宽比」而不是「周期」。

已有 `chi2_*.npz` 时不必再跑 S4：

```bash
python3 run_chi2_landscape.py --replot ../runs/inverse/chi2_landscape
```

```bash
cd inverse

python3 run_chi2_landscape.py --config config_chi2_p80.yaml --dry-run
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --dry-run
python3 run_chi2_landscape.py --config config_chi2_p300_d150.yaml --dry-run

python3 run_chi2_landscape.py --config config_chi2_p80.yaml --workers 8
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --workers 4
python3 run_chi2_landscape.py --config config_chi2_p300_d150.yaml --workers 4
python3 run_chi2_landscape.py --config config_chi2_p80.yaml --noisy --seed 0 --workers 8
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --noisy --seed 0 --workers 4
python3 run_chi2_landscape.py --config config_chi2_p300_d150.yaml --noisy --seed 0 --workers 4
```

`--noisy` 写到 `p80_noisy/` / `p300_noisy/` / `p300_d150_noisy/`，不会盖掉无噪声结果。中断后重跑会读**该目录**里已有的 `chi2_cd_depth.npz`。不要两个终端写同一个输出目录。

第二张切片（墙角）：

```bash
python3 run_chi2_landscape.py --config config_chi2_p80.yaml --slice cd_swa --workers 8
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --slice cd_swa --workers 4
python3 run_chi2_landscape.py --config config_chi2_p300_d150.yaml --slice cd_swa --workers 4
```

有噪声用同一套 \(\sigma\) 模型（\(N_0=10^6\)、\(a=1\%\)），默认 `--seed 0`。

## 输出

```
runs/inverse/chi2_landscape/p80/
  chi2_cd_depth.npz
  chi2_cd_depth.json
  chi2_cd_depth_decoupling.png
  chi2_cd_depth_decoupling_log.png
  chi2_cd_depth_m0_all.png
  chi2_cd_depth_m0_all_log.png
  ...
```

线性图左侧是均匀切的等高线（黑叉真值，青圈 loss 最小），右侧是曲面。`*_log.png` 左侧按对数取圈，右侧 Z 是 \(\log_{10} L\)，无噪声真值格点 \(L=0\) 落在最内圈里面。`json` 里每个 mask 有 `loss_min` 和相对真值的偏移。数值比未归一化的 \(\chi^2\) 小，因为 \(\sum w=1\)。

## 怎样读

- **只有一个盆、最小点就在真值**：全局没有第二谷，B 模式若仍远差于 CRLB，更像求解器问题。
- **两个（或更多）盆**：Fig. 4 那种多谷；看第二谷离真值多远、loss 高多少。
- **`m0_all` 沿 SWA / 某方向拉成长谷**：和该 mask 上墙角 CRLB 很大是同一件事。

## 单元测试（不调 S4）

```bash
cd inverse
python3 -m pytest test_chi2_landscape.py -q
```
