# 紧凑 recipe 上的 CRLB 与反演对照

这份说明足够在服务器上单独跑完实验，**不需要上传或打开原来的对话**。拉这个分支、读本文、按命令开终端即可。

密网格 FIM（`fim_study_dense_p80` / `p300`）是 399 个 \((\lambda,\varphi)\) 上的线性诊断。现用反演大约只有 20 个条件。两个 CRLB **不能直接比**。本实验在**同一套观测量**上重算 \(F^{-1}\)，再做带噪反演。

## 在比什么

| 模式 | 求解器 | 权重 | 初值 | 默认次数 | 问题 |
|---|---|---|---|---|---|
| A | 只用 LM | 原始 \(1/\sigma\)（与 FIM 相同，无角色因子） | 真值 | 40 | 下限的尺度和排序准不准 |
| B | 现用 GA+LM（80 nm 有库则 `lib_pop_rand_ga_lm`） | 解耦 mask 才开角色权重 | 库 / GA | 20 | 求解器能不能走到 CRLB 附近 |

几何与密网格真值一致：80 nm 用 CD 40 / depth 40；300 nm 用 CD 150 / depth 40。墙 \(89.63^\circ/89.26^\circ\)。光学是现成反演 recipe：H55–61 奇次 × \(\theta=70^\circ\) × \(\varphi\in\{0,30,45,60,90\}\)。噪声 \(N_0=10^6\)、\(a=1\%\)。

四个 mask：`prop`、`decoupling`、`m0_all`、`only90`。行集与 `fim_study.mask_rows` 对齐（可传播的 \(m\in\{-1,0,1\}\)）。

## 不要上传对话

服务器上的 Cursor 只要能读本仓库。做法：

1. `git fetch && git checkout cursor/fim-jacobian-study-7d74 && git pull`
2. 打开 `inverse/docs/CRLB_MC.md`（本文）
3. 按下面的命令跑

不必把本机 chat 同步到云端。同步的是 git，不是对话。

## 并行：一终端一格

S4 可以多进程同时跑。缓存在进程内存里，谱库只读。

**每个任务必须写到不同目录。** 脚本已经按 `{output_dir}/{mask}_{A|B}/` 拆开。同一份 yaml、不同 `--mask` 即可并跑。不要两个终端写同一个 `{mask}_{mode}`。

核数拆法（A 的 `ga_workers=1`）：

\[
\text{终端数}\times\texttt{--condition-workers}
\;\lesssim\;
\text{逻辑核数}
\]

56 核建议先 **4 个终端 × 8 workers**。用 `htop` 或 `pgrep -c S4` 看有没有超卖。300 nm、`NG=61` 更吃内存，不要一上来就 8 个窗口。

用 `tmux`，不要依赖会断的 SSH 窗口。

## 一周怎么跑

在 `inverse/` 下。先 80 nm（`NG=31`），再 300 nm。

### 第 0 步：对行、算 FIM、标定时间（串行）

```bash
cd inverse

python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode layout
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mode layout

python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode fim
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mode fim

python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode probe
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mode probe
```

`layout` 不调 S4，应打印四行 `OK`。`fim` 各算一张紧凑 \(J\)（约 \(20\times5\) 次 S4），写入：

- `../runs/inverse/crlb_mc/p80/fim_compact.json`
- `../runs/inverse/crlb_mc/p80/jacobian_compact.npz`

`probe` 会给出一次 A 的墙钟，并外推 40 次 / 四格的时间。按这个数改并行窗口，不要按文档里的估时锁死。

### 第 1–2 天：80 nm 的 A（可四开）

四个 `tmux` 窗口，每个一条：

```bash
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask decoupling --mode A
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask prop --mode A
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask m0_all --mode A
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask only90 --mode A
```

默认 40 次。中断后重跑会跳过 `trials.jsonl` 里已有的 trial。

### 第 3–5 天：300 nm 的 A

同样四条，换成 `config_crlb_mc_p300.yaml`。若 `probe` 显示单次 A 超过约 15 分钟，先保 `decoupling` 和 `m0_all` 的 40 次，另外两格加 `--n-trials 25`。

四条 A 会共用已经算好的 `jacobian_compact.npz`。若 `fim` 还没跑完就开了 A，第一个进程算 \(J\)，其余等锁文件，不要手动删 `.lock`。

### 第 6 天：只做两格 B

```bash
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mask decoupling --mode B --n-trials 20
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mask m0_all --mode B --n-trials 20
```

80 nm 若服务器上有 `library.file` 指向的 npz，可再加：

```bash
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask decoupling --mode B --n-trials 20
```

300 nm 不要为这一周建库。没有库时 B 自动退回 `ga_lm`。

### 第 7 天：汇总

```bash
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mode summarize
python3 run_crlb_mc.py --config config_crlb_mc_p300.yaml --mode summarize
```

## 输出

```
runs/inverse/crlb_mc/p80/
  fim_compact.json
  jacobian_compact.npz
  decoupling_A/trials.jsonl
  decoupling_A/summary.json
  decoupling_A/scatter.png
  m0_all_A/...
```

`summary.json` 里：`std`、`rmse`、`bias`、`efficiency_crlb_over_std`（CRLB / \(\hat\sigma\)，贴 1 表示 A 达到下限）、经验相关、`swap_rate`。

看图：`scatter.png` 的 \((\mathrm{lswa},\mathrm{rswa})\)。`m0_all` 应贴在 \(L+R\approx\mathrm{const}\) 上。

## 怎样才算对上

- **排序对、尺度差几倍**：FIM 能指导配方，绝对 CRLB 当不了误差条。
- **`m0_all` 单墙 RMSE 看起来还行**：看 \(\rho(L,R)\) 和 \(L-R\) 的散点，不要只看单墙。
- **A 贴 CRLB、B 差一个数量级**：局部信息够，全局会掉坑。
- **A 也比 CRLB 差很多**：先确认比的是紧凑 `fim_compact.json`，不是密网格那张表；A 必须关角色权重（脚本已关）。

## 改并行度

```bash
python3 run_crlb_mc.py --config config_crlb_mc_p80.yaml --mask decoupling --mode A \
  --condition-workers 8 --n-trials 40 --seed 0
```

不要把单个任务的 `ga_workers` 拉高再叠多终端。A 不跑 GA。

## 单元测试（不调 S4）

```bash
cd inverse
python3 -m pytest test_crlb_mc.py -q
```
