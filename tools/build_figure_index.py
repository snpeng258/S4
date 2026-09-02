#!/usr/bin/env python3
"""为仓库内所有绘图产物建立「图片 ↔ 生成脚本 ↔ 绘图数据」索引。

在桌面生成一个按功能分类的软链接目录，供人工筛选；同时输出对应表
（TSV + Markdown），说明每张图由哪个脚本、读哪份数据生成。

软链接不占额外空间，删除软链接不会动到仓库里的原图。
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 上游 S4 自带的文档插图，不属于 Sofia 的工作，排除
EXCLUDE_DIRS = ("doc/", "docold/", ".git/")


@dataclass
class Rule:
    """一类绘图产物的归属规则。"""

    category: str  # 分类目录名
    pattern: re.Pattern  # 匹配 png 文件名
    script: str  # 生成脚本（相对仓库根）
    entry: str  # 推荐的运行入口
    data: str  # 绘图数据来源说明
    data_glob: str | None = None  # 用于在同目录定位实际数据文件
    note: str = ""
    # 从 png 文件名里提取参数（wl / th / az 等），用于把图和「同一次仿真」的
    # 数据文件精确配对，而不是把同目录所有数据都列出来
    param_re: re.Pattern | None = None
    # 用 param_re 提取到的命名组格式化出的 glob，可给多个（按顺序全部尝试）
    param_globs: tuple[str, ...] = ()


# 远场 png / 目录名开头统一是 {波长}_{入射角}deg_az{方位角}deg_...
_FAR_PARAMS = re.compile(r"^(?P<wl>[\dp]+nm)_(?P<th>\d+)deg_az(?P<az>\d+)deg")
_FAR_GLOBS = ("s4_farfield_wl{wl}_th{th}deg_az{az}deg_*_waves_air.txt",)

RULES: list[Rule] = [
    Rule(
        category="01-远场-ASR总图",
        pattern=re.compile(r"_farfield_asr\.png$"),
        script="farfield/s4_farfield_asr.py",
        entry="farfield/run_s4_farfield4.sh",
        data="S4 GetWaves 波幅表 s4_farfield_wl{WL}_th{TH}deg_az{AZ}deg_L80nm_waves_air.txt",
        note="Floquet + 椭圆孔径 + ASR 传播后拼接的实验室 CCD 总图",
        param_re=_FAR_PARAMS,
        param_globs=_FAR_GLOBS,
    ),
    Rule(
        category="02-远场-ASR单级次",
        pattern=re.compile(r"_farfield_asr_order_m[pm]\d+\.png$|^order_m[pm]\d+\.png$"),
        script="farfield/s4_farfield_asr.py",
        entry="farfield/run_s4_farfield4.sh",
        data="同总图，来自同一次 S4 波幅表",
        note="mp0=0 级, mp1=+1 级, mp2=+2 级, mm1=-1 级, mm2=-2 级",
        param_re=_FAR_PARAMS,
        param_globs=_FAR_GLOBS,
    ),
    Rule(
        category="03-远场-ASR级次面板",
        pattern=re.compile(r"_farfield_asr_orders_panel\.png$|^orders_panel\.png$"),
        script="farfield/s4_farfield_asr.py",
        entry="farfield/run_s4_farfield4.sh",
        data="同总图，来自同一次 S4 波幅表",
        note="各级次拼在一张图上的对照面板",
        param_re=_FAR_PARAMS,
        param_globs=_FAR_GLOBS,
    ),
    Rule(
        category="04-远场-ASR诊断",
        pattern=re.compile(r"^patch_m[pm]\d+_(I_raw_linear|I_norm_linear|profile)\.png$"),
        script="farfield/farfield_asr_diagnose.py",
        entry="python farfield/farfield_asr_diagnose.py",
        data="同目录 patch_{级次}.npz（该级次的 ASR 光斑复场）",
        note="I_raw=原始强度, I_norm=归一化强度, profile=一维剖线；"
        "指标汇总见同目录 diagnose_summary.json",
        param_re=re.compile(r"^patch_(?P<tag>m[pm]\d+)_"),
        param_globs=("patch_{tag}.npz",),
    ),
    Rule(
        category="04-远场-ASR诊断",
        pattern=re.compile(r"^canvas_m0_native_vs_interp\.png$"),
        script="farfield/farfield_asr_diagnose.py",
        entry="python farfield/farfield_asr_diagnose.py",
        data="同目录 canvas_stitched.npz（拼接后画布）+ canvas_m0_profiles.tsv（剖线数值）",
        data_glob="canvas_*",
        note="0 级光斑「原生贴图 vs 插值贴图」对比，用于定位拼接失真",
    ),
    Rule(
        category="05-远场-HHG谐波叠加",
        pattern=re.compile(r"^hhg800nm_.*_overlay\.png$"),
        script="farfield/s4_farfield_hhg_panel.py",
        entry="farfield/run_s4_farfield5.sh",
        data="H 范围内每个谐波级各一份波幅表 s4_farfield_hhg_q{q}_th60deg_az90deg_L80nm_waves_air.txt",
        note="文件名 H45-H61 为叠加的谐波级范围，wl13-18nm 为对应波长范围；"
        "各光斑落点另存为同名 .spots.json",
    ),
    Rule(
        category="06-近场-总场",
        pattern=re.compile(r"^s4_nearfield_total_.*\.png$"),
        script="nearfield/1d/plot_s4_nearfield_viz.py",
        entry="nearfield/1d/run_s4_nearfield7.sh",
        data="同名 .txt（S4 GetEField TSV，含 #BEGIN_NEARFIELD_TSV 块）",
        note="_kspace_raw 后缀为不去载波的原始 FFT 谱",
        param_re=re.compile(r"^(?P<stem>.+?)(?:_kspace_raw)?\.png$"),
        param_globs=("{stem}.txt",),
    ),
    Rule(
        category="07-近场-2D",
        pattern=re.compile(r"^s4_nearfield_2d_.*\.png$"),
        script="nearfield/2d/plot_s4_nearfield_xy_viz.py",
        entry="nearfield/2d/run_s4_nearfield_2d.sh",
        data="同名 .txt（去掉 _kspace 后缀后与 png 同名）",
        note="x-y 面近场；_kspace 为对应频谱",
        param_re=re.compile(r"^(?P<stem>.+?)(?:_kspace)?\.png$"),
        param_globs=("{stem}.txt",),
    ),
    Rule(
        category="08-演示素材",
        pattern=re.compile(r"^source-page-\d+\.png$"),
        script="docs/side-slide-preview/azimuth-decoupling-demo/build_slide.py",
        entry="python docs/side-slide-preview/azimuth-decoupling-demo/build_slide.py",
        data="docs/side-slide-preview/azimuth-decoupling-demo/ 下的 md 文稿",
        data_glob=None,
        note="方位角解耦演示幻灯片素材",
    ),
]


@dataclass
class Entry:
    png: Path
    rule: Rule
    data_files: list[Path] = field(default_factory=list)


def classify(png: Path) -> Rule | None:
    name = png.name
    for rule in RULES:
        if rule.pattern.search(name):
            return rule
    return None


def _globs_for(png: Path, rule: Rule) -> list[str]:
    """把 png（或其父目录）名里的参数代入 glob 模板。"""
    if not (rule.param_re and rule.param_globs):
        return [rule.data_glob] if rule.data_glob else []
    # 级次子目录里的图（order_mp0.png）自身不含参数，参数在父目录名上
    for source in (png.name, png.parent.name):
        m = rule.param_re.search(source)
        if m:
            return [g.format(**m.groupdict()) for g in rule.param_globs]
    return [rule.data_glob] if rule.data_glob else []


def _hhg_data(png: Path) -> list[Path]:
    """HHG 叠加图：按文件名里的 H 谐波级范围收集逐级波幅表 + spots.json。"""
    hits: list[Path] = []
    spots = png.with_suffix("").with_suffix(".spots.json")
    if spots.exists():
        hits.append(spots)
    m = re.search(r"H(\d+)-H(\d+)", png.name)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        for q in range(lo, hi + 1):
            hits.extend(sorted(png.parent.glob(f"s4_farfield_hhg_q{q}_*_waves_air.txt")))
    return hits


def find_data(png: Path, rule: Rule) -> list[Path]:
    """定位与该图同属一次仿真的数据文件。"""
    if rule.category.startswith("05-"):
        return _hhg_data(png)

    hits: list[Path] = []
    for glob in _globs_for(png, rule):
        # 数据可能在图所在目录，也可能在上一层（级次子目录的情况）
        for base in (png.parent, png.parent.parent):
            found = sorted(base.glob(glob))
            if found:
                hits.extend(found)
                break
    # 去重且保序
    return list(dict.fromkeys(hits))


def collect() -> tuple[list[Entry], list[Path]]:
    entries: list[Entry] = []
    unmatched: list[Path] = []
    for png in sorted(REPO.rglob("*.png")):
        rel = png.relative_to(REPO).as_posix()
        if any(rel.startswith(d) for d in EXCLUDE_DIRS):
            continue
        rule = classify(png)
        if rule is None:
            unmatched.append(png)
            continue
        entries.append(Entry(png=png, rule=rule, data_files=find_data(png, rule)))
    return entries, unmatched


def build_symlinks(entries: list[Entry], out_root: Path) -> None:
    """按分类建软链接。同名冲突时用父目录名做前缀区分。"""
    for e in entries:
        cat_dir = out_root / e.rule.category
        cat_dir.mkdir(parents=True, exist_ok=True)
        link = cat_dir / e.png.name
        if link.exists() or link.is_symlink():
            link = cat_dir / f"{e.png.parent.name}__{e.png.name}"
        if link.is_symlink():
            link.unlink()
        os.symlink(e.png, link)


def write_manifest(entries: list[Entry], unmatched: list[Path], out_root: Path) -> None:
    tsv = out_root / "对应表.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("分类\t图片\t生成脚本\t运行入口\t绘图数据\t配套数据文件\n")
        for e in entries:
            data = ";".join(p.relative_to(REPO).as_posix() for p in e.data_files) or "-"
            f.write(
                f"{e.rule.category}\t{e.png.relative_to(REPO).as_posix()}\t"
                f"{e.rule.script}\t{e.rule.entry}\t{e.rule.data}\t{data}\n"
            )

    md = out_root / "README.md"
    by_cat: dict[str, list[Entry]] = {}
    for e in entries:
        by_cat.setdefault(e.rule.category, []).append(e)

    with md.open("w", encoding="utf-8") as f:
        f.write("# S4 绘图产物筛选目录\n\n")
        f.write(
            "本目录下全部是**软链接**，指向仓库里的原图。\n"
            "- 直接双击可预览；\n"
            "- 删除这里的软链接**不会**删除仓库原图，只是从筛选清单里剔除；\n"
            "- 筛选完后回头告诉我保留哪些，我把它们移到 `docs/figures/` 并纳入 git。\n\n"
        )
        f.write(f"共 {len(entries)} 张图，分 {len(by_cat)} 类。\n\n")
        for cat in sorted(by_cat):
            items = by_cat[cat]
            r = items[0].rule
            f.write(f"## {cat}（{len(items)} 张）\n\n")
            f.write(f"- **生成脚本**：`{r.script}`\n")
            f.write(f"- **运行入口**：`{r.entry}`\n")
            f.write(f"- **绘图数据**：{r.data}\n")
            if r.note:
                f.write(f"- **说明**：{r.note}\n")
            f.write("\n| 图片 | 配套数据文件 |\n|---|---|\n")
            for e in sorted(items, key=lambda x: x.png.name):
                data = (
                    "<br>".join(
                        f"`{p.relative_to(REPO).as_posix()}`" for p in e.data_files[:3]
                    )
                    or "—（重跑脚本复现）"
                )
                if len(e.data_files) > 3:
                    data += f"<br>…另有 {len(e.data_files) - 3} 个"
                f.write(f"| `{e.png.relative_to(REPO).as_posix()}` | {data} |\n")
            f.write("\n")

        if unmatched:
            f.write("## 未归类\n\n")
            for p in unmatched:
                f.write(f"- `{p.relative_to(REPO).as_posix()}`\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path.home() / "Desktop" / "S4-图片筛选"),
        help="软链接输出目录",
    )
    args = ap.parse_args()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    entries, unmatched = collect()
    build_symlinks(entries, out_root)
    write_manifest(entries, unmatched, out_root)

    print(f"输出目录: {out_root}")
    print(f"已建软链接: {len(entries)} 张")
    if unmatched:
        print(f"未归类: {len(unmatched)} 张")
        for p in unmatched:
            print(f"  - {p.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
