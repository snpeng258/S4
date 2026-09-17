# 紧凑 recipe 上的 χ² 全局景观

对应 S4-8（Gross 等 2009）Fig. 4：固定一份测量，把加权 \(\chi^2\) 铺在两个形貌参数上。不跑 GA/LM。左右墙已统一为 `swa_deg`。

和 CRLB–MC **可以并行**，但必须写到不同目录，也不要抢光核。

## 在画什么

默认切片钉死真值侧壁角，扫 CD × depth（Fig. 4 钉死 \(p_7\)、扫 \(p_2\)–\(p_6\)）。`--slice cd_swa` 则钉死 depth，扫 CD × SWA。

测量先用真值无噪声 \(R_{\mathrm{meas}}\)。权重是固定的探测器 \(1/\sigma\)（由这份 \(R_{\mathrm{meas}}\) 算出），不含解耦角色因子。正演存全套级次，事后再套 `prop` / `decoupling` / `m0_all` / `only90`，四个 mask 共用同一次 S4。

## 配置

| | 80 nm | 300 nm |
|---|---|---|
| 文件 | `config_chi2_p80.yaml` | `config_chi2_p300.yaml` |
| 真值 | CD 40 / depth 40 / SWA \(89.45^\circ\) | CD 150 / depth 40 / SWA \(89.45^\circ\) |
| `NG` | 31 | 61 |
| 默认网格 | CD、depth 各 32–48、步长 1 → \(17\times17=289\) | CD 134–166 步长 2、depth 32–48 步长 1 → \(17\times17=289\) |
| 建议 workers | 8 | 4 |
| 输出 | `../runs/inverse/chi2_landscape/p80/` | `../runs/inverse/chi2_landscape/p300/` |
| S4 次数（约） | \(289\times20=5780\) | \(289\times20=5780\) |

光学与 CRLB–MC 相同：H55–61 奇次 × \(\theta=70^\circ\) × \(\varphi\in\{0,30,45,60,90\}\)。

## 怎么跑

不要和 `crlb_mc/p80`、`crlb_mc/p300` 写到一起。CRLB 的 A 若已经占了 4×8 核，80 nm 景观用 8 个形貌进程，300 nm 用 4 个。每个进程的 `condition_workers` 固定为 1。

```bash
cd inverse

python3 run_chi2_landscape.py --config config_chi2_p80.yaml --dry-run
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --dry-run

python3 run_chi2_landscape.py --config config_chi2_p80.yaml --workers 8
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --workers 4
```

中断后重跑会读已有 `chi2_cd_depth.npz` 并跳过填过的点。不要两个终端写同一个 yaml。

第二张切片（墙角）：

```bash
python3 run_chi2_landscape.py --config config_chi2_p80.yaml --slice cd_swa --workers 8
python3 run_chi2_landscape.py --config config_chi2_p300.yaml --slice cd_swa --workers 4
```

加一次噪声实现（同一套 \(\sigma\) 模型，\(N_0=10^6\)、\(a=1\%\)）：

```bash
python3 run_chi2_landscape.py --config config_chi2_p80.yaml --noisy --seed 0 --no-resume
```

## 输出

```
runs/inverse/chi2_landscape/p80/
  chi2_cd_depth.npz
  chi2_cd_depth.json
  chi2_cd_depth_decoupling.png
  chi2_cd_depth_m0_all.png
  ...
```

图左侧是等高线（黑叉真值，青圈 \(\chi^2\) 最小），右侧是曲面。`json` 里每个 mask 有 `chi2_min` 和相对真值的偏移。无噪声时真值格点的 \(\chi^2\) 应接近 0。

## 怎样读

- **只有一个盆、最小点就在真值**：全局没有第二谷，B 模式若仍远差于 CRLB，更像求解器问题。
- **两个（或更多）盆**：Fig. 4 那种多谷；看第二谷离真值多远、\(\chi^2\) 高多少。
- **`m0_all` 沿 SWA / 某方向拉成长谷**：和该 mask 上墙角 CRLB 很大是同一件事。

## 单元测试（不调 S4）

```bash
cd inverse
python3 -m pytest test_chi2_landscape.py -q
```
