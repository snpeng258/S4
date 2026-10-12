# 90°/0° 零级对 CD / 槽深 / 侧壁角的灵敏度

对应 S4-4（Corazza 等 NC 2026）Fig. 1：同一套 80 nm Au/SiO₂ 方槽、θ=70°、TE，只取 **m=0** 的 \(R_0(\lambda)\)。左右两列是方位角 \(0^\circ\)（平面掠入射）和 \(90^\circ\)（圆锥衍射，刻线沿入射面）。三行分别扫槽深、CD、侧壁角，另外两个形貌钉在真值。SWA 那一行是文章 Fig. 1 没有的，先画出来再分析。

不是反演、不建库、不加噪声。每个格子一次单条件 S4。

## 在画什么

| 行 | 扫的量 | 钉死 |
|---|---|---|
| grating height | depth 24–56 nm，步长 2 | CD 40 nm，SWA \(89.45^\circ\) |
| CD | CD 32–48 nm，步长 2 | depth 40 nm，SWA \(89.45^\circ\) |
| SWA | SWA 85–94°，步长 0.5 | CD 40 nm，depth 40 nm |

\(\lambda=12\)–\(16\,\mathrm{nm}\)，步长 0.2 nm（21 点）。灰虚线标 H55–61 奇次（800 nm 基波）。图上白虚线是真值。两列共用该行的色标，方便看 \(0^\circ\) vs \(90^\circ\)。

## 暗纹位置和波长轴

图里的颜色是零级反射率（衍射效率），不是样品表面的空间像。「暗条纹」就是 \(R_0\) 低的地方。

台面金和槽底之间多走的真空光程是几何量 \(2h\cos\theta\)。反射率写成

\[
R_0 \approx \bigl|f\,r_{\mathrm{Au}}+(1-f)\,r_{\mathrm{SiO_2}}e^{i 4\pi h \cos\theta/\lambda}\bigr|^2,\qquad f=\mathrm{CD}/\mathrm{pitch}.
\]

相位 \(\psi=4\pi h\cos\theta/\lambda=2\pi\times(2h\cos\theta)/\lambda\)：固定 \(h,\theta\) 时，扫 \(\lambda\) 就是在扫「这段光程里装了多少个波长」。相消时 \(\psi+\varphi=(2m+1)\pi\)，暗纹波长

\[
\lambda_m=\frac{4h\cos\theta}{2m+1-\varphi/\pi}.
\]

\(f\) 只进振幅（条纹有多黑），不进 \(\lambda_m\)。所以 Fig. 1d（\(y=\)CD，\(x=\lambda\)，\(h\) 钉死）里暗带是**竖条**：谷的波长不随 CD 走。Fig. 1b（\(y=h\)）里同一条件变成斜带，因为 \(h/\lambda\) 为常才相消。Fig. 1c 在 \(\varphi=0^\circ\) 会弯，是掠入射把邻居阴影耦进有效光程，不是波长轴和几何对不上。

本配置 \(h=40\,\mathrm{nm}\)、\(\theta=70^\circ\)、忽略材料相位时，前几条暗纹大约在 \(55,\ 18,\ 11,\ 8\,\mathrm{nm}\)。\(\lambda=12\)–\(16\,\mathrm{nm}\) 落在 \(11\) 与 \(18\) 之间，CD 行的 \(90^\circ\) 列可能看不到一整条黑带，只表现为比 \(0^\circ\) 更不随 CD 走。json 的 `spectral_dark_fringes_nm` 按上式列出。

金在 14.5 nm 趋肤深度约 16 nm，光几乎不穿过金线；pitch/\(\lambda\sim 5\)–\(20\)，不是准静态 \(d\ll\lambda\)，Haggans–Li–Kostuk 那层有效介质薄膜不适用于这里的零级。

预期（先跑再下结论）：

- **槽深行**两边都应看到随 \(\lambda\)、\(h\) 走的干涉条纹，周期大约 \(\Delta h\sim\lambda/(2\cos\theta)\approx 18\)–\(23\,\mathrm{nm}\)。
- **CD 行**在 \(90^\circ\) 应明显比 \(0^\circ\) 更平：圆锥几何下零级条纹的相位由高度定，占空比只改对比度，不改条纹位置。\(0^\circ\) 掠入射会把邻居阴影耦进相位，CD 会进条纹。
- **SWA 行**先出图，不预先声称哪边更敏感。

同目录还有 `fig_two_beam_depth90.png`：在 \(\varphi=90^\circ\)、真值 CD 上，把 S4 的 \(R_0(h)\) 叠到双光束

\[
R_0 \approx \bigl|f\,r_{\mathrm{Au}}+(1-f)\,r_{\mathrm{SiO_2}}e^{i 4\pi h \cos\theta/\lambda}\bigr|^2,\qquad f=\mathrm{CD}/\mathrm{pitch}
\]

\(r\) 用 CXRO \(\varepsilon\) 的真空→介质 TE Fresnel。只用来核对条纹周期，不是 RCWA 的替代。json 里的 `sensitivity` 是 \(\mathrm{rms}(\partial R_0/\partial p)\)（沿形貌轴有限差分，再对 \(\lambda\) 和该轴取 rms），以及 \(90^\circ\) vs \(0^\circ\) 的 CD/depth 比。

## 配置

| | 80 nm 方槽 |
|---|---|
| 文件 | `config_phi90_cd_sensitivity_p80.yaml` |
| 真值 | CD 40 / depth 40 / SWA \(89.45^\circ\) |
| `NG` | 31 |
| 方位角 | \(0^\circ,\ 90^\circ\) |
| 网格 | λ 21 ×（CD 9 + depth 17 + SWA 19） |
| S4 次数 | \(2\times 21\times 45=1890\) |
| workers | 8 |
| 输出 | `../runs/inverse/phi90_cd_sensitivity/p80/` |

光学主参数与 χ² 景观相同（θ=70°、TE、Au/SiO₂），但这里**不**跑 H55–61×五方位那 20 个条件：每个点只有一个 \((\lambda,\varphi)\)。recipe 只用来在图上标谐波。

## 怎么跑

不要写进 `chi2_landscape/` 或 `crlb_mc/`。先 `--dry-run` 核对 `S4=1890`。

```bash
cd inverse

python3 run_phi90_cd_sensitivity.py --config config_phi90_cd_sensitivity_p80.yaml --dry-run
python3 -m pytest test_phi90_cd_sensitivity.py -q
python3 run_phi90_cd_sensitivity.py --config config_phi90_cd_sensitivity_p80.yaml --workers 8
```

中断后重跑会读同目录的 `phi90_cd_sensitivity.npz`。不要两个终端写同一个输出目录。已有 npz 时不必再跑 S4：

```bash
python3 run_phi90_cd_sensitivity.py --replot ../runs/inverse/phi90_cd_sensitivity/p80
```

## 输出

- `phi90_cd_sensitivity.npz` — \(R_0\) 三维数组（方位 × 形貌 × 波长）
- `phi90_cd_sensitivity.json` — 网格、谐波、高度条纹周期、灵敏度表
- `fig_phi90_cd_sensitivity.png` — 3×2（depth / CD / SWA × \(0^\circ\) / \(90^\circ\)）
- `fig_two_beam_depth90.png` — H55–61 上 S4 vs 双光束

看图时先看 CD 行的 \(90^\circ\) 列是否比 \(0^\circ\) 更平，再看 `sensitivity` 里 `cd_over_depth`。SWA 行留给后面分析。
