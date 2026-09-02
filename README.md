# S4 程序包（Sofia）

根目录只放自己的工作。上游 Stanford S4 源码在 [`upstream/`](upstream/)。

| 目录 | 内容 |
|------|------|
| `farfield/` | 远场 ASR / 光斑 / HHG |
| `nearfield/` | 近场 1D / 2D |
| `flux/` | 反射通量 |
| `inverse/` | 散射测量逆问题 |
| `shared/` | 共用材料介电常数 |
| `docs/` | 笔记与开题材料 |
| `tools/` | `s4_env.sh`、图片索引 |
| `data/` `runs/` | 本地数据与出图（不入库） |
| `upstream/` | 原版 S4（编译：`cd upstream && make`） |

二进制默认路径：`upstream/build/S4`。工作流从任意子目录向上找 `tools/s4_env.sh` 作为仓库根。
