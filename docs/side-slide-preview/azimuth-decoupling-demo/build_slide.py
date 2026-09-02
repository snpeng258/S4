from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(
    r"C:\Users\25819\.codex\visualizations\2026\08\08\019fe098-e03f-78a2-88f9-3a1409a4e3ac\azimuth-decoupling-demo"
)
REGULAR_FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD_FONT = r"C:\Windows\Fonts\msyhbd.ttc"


def get_font(size, weight="regular"):
    path = BOLD_FONT if weight == "bold" else REGULAR_FONT
    return ImageFont.truetype(path, size)


page = Image.open(OUT / "source-page-04-hires.png").convert("RGB")
figure = page.crop((940, 960, 2350, 1660))
figure.save(OUT / "source-figure-2cd.png", quality=95)

width, height = 1920, 1080
slide = Image.new("RGB", (width, height), "#F8F9FA")
draw = ImageDraw.Draw(slide)


def rounded_rect(box, radius, fill, outline=None, stroke_width=1):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=stroke_width,
    )


def fit_contain(source, box, padding=0):
    x0, y0, x1, y1 = box
    max_width = x1 - x0 - 2 * padding
    max_height = y1 - y0 - 2 * padding
    scale = min(max_width / source.width, max_height / source.height)
    size = (int(source.width * scale), int(source.height * scale))
    resized = source.resize(size, Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - size[0]) // 2
    y = y0 + (y1 - y0 - size[1]) // 2
    slide.paste(resized, (x, y))


# Header
draw.text(
    (80, 45),
    "SCATTERMETRY VISUAL NOTE",
    font=get_font(22, "bold"),
    fill="#2563EB",
)
draw.text(
    (80, 88),
    "旋转方位角，改变零级光的参数灵敏度",
    font=get_font(58, "bold"),
    fill="#1A1A1A",
)
draw.text(
    (82, 172),
    "同一维光栅，两种衍射几何提供互补信息",
    font=get_font(29),
    fill="#4A5568",
)
draw.line((80, 224, 1840, 224), fill="#D1D5DB", width=2)

# Authentic source figure
rounded_rect((80, 260, 980, 805), 16, "#FFFFFF", "#D1D5DB", 2)
draw.text(
    (110, 282),
    "论文原始几何示意",
    font=get_font(25, "bold"),
    fill="#1A1A1A",
)
draw.text(
    (110, 321),
    "PLANAR / CONICAL DIFFRACTION",
    font=get_font(18),
    fill="#718096",
)
fit_contain(figure, (105, 350, 955, 745), 4)
draw.text(
    (110, 760),
    "周期方向（绿）与入射光面内投影（紫）的夹角定义为 φ",
    font=get_font(20),
    fill="#4A5568",
)


def info_card(box, accent, title, lines):
    x0, y0, x1, y1 = box
    rounded_rect(box, 16, "#FFFFFF", "#D1D5DB", 2)
    draw.rounded_rectangle((x0, y0, x0 + 12, y1), radius=6, fill=accent)
    draw.text(
        (x0 + 36, y0 + 26),
        title,
        font=get_font(33, "bold"),
        fill="#1A1A1A",
    )
    y = y0 + 84
    for label, value, strong in lines:
        draw.ellipse((x0 + 38, y + 9, x0 + 48, y + 19), fill=accent)
        draw.text(
            (x0 + 63, y), label, font=get_font(23), fill="#4A5568"
        )
        draw.text(
            (x0 + 300, y),
            value,
            font=get_font(23, "bold" if strong else "regular"),
            fill=accent if strong else "#1A1A1A",
        )
        y += 47


info_card(
    (1020, 260, 1840, 505),
    "#F97316",
    "φ = 0°  平面衍射",
    [
        ("几何", "周期方向 ∥ 入射投影", False),
        ("机制", "出现相邻线条阴影", True),
        ("零级编码", "CD + 槽深", True),
    ],
)
info_card(
    (1020, 535, 1840, 780),
    "#0F766E",
    "φ = 90°  锥形衍射",
    [
        ("几何", "光栅线 ∥ 入射面", False),
        ("机制", "阴影效应消失", True),
        ("零级编码", "槽深响应占主导", True),
    ],
)

# Sequential decoupling path
rounded_rect((80, 835, 1840, 1005), 16, "#FFFFFF", "#D1D5DB", 2)
draw.text((110, 861), "解耦路径", font=get_font(27, "bold"), fill="#1A1A1A")
steps = [
    ((315, 860, 700, 975), "#E6FFFA", "#0F766E", "01", "90°零级", "估计槽深 GH"),
    ((780, 860, 1165, 975), "#EFF6FF", "#2563EB", "02", "传递深度先验", "h90 与 σh"),
    ((1245, 860, 1630, 975), "#FFF7ED", "#F97316", "03", "0°零级", "估计中部 CD"),
]
for box, fill, accent, number, line1, line2 in steps:
    rounded_rect(box, 12, fill)
    x0, y0, _, _ = box
    draw.text(
        (x0 + 22, y0 + 18), number, font=get_font(20, "bold"), fill=accent
    )
    draw.text(
        (x0 + 72, y0 + 15), line1, font=get_font(25, "bold"), fill="#1A1A1A"
    )
    draw.text(
        (x0 + 72, y0 + 59), line2, font=get_font(22), fill="#4A5568"
    )

for x in (735, 1200):
    draw.line((x, 918, x + 32, 918), fill="#94A3B8", width=5)
    draw.polygon([(x + 32, 908), (x + 48, 918), (x + 32, 928)], fill="#94A3B8")

source = (
    "来源：Corazza et al., Nature Communications 17, 6573 (2026) · "
    "DOI 10.1038/s41467-026-73052-w"
)
draw.text((80, 1030), source, font=get_font(19), fill="#64748B")

slide.save(OUT / "01-slide-azimuth-decoupling.png", optimize=True)
