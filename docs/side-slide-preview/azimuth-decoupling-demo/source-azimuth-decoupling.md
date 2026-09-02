# 方位角改变零级光的参数灵敏度

依据 Corazza 等 2026 年论文：对于一维周期光栅，平面衍射条件 `phi = 0 deg` 下，大入射角造成相邻光栅线的部分阴影，零级光谱同时受临界尺寸（CD）和槽深（GH）影响；锥形衍射条件 `phi = 90 deg` 下，阴影效应消失，零级光谱主要由槽深引起的相位积累决定，对 CD 相对不敏感。

适合视觉化的解耦路径：先用 90 deg 方位角估计槽深，再将槽深及其不确定度作为先验，用 0 deg 方位角估计 CD。

来源：Corazza et al., Broadband extreme ultraviolet zeroth order scatterometry for nanostructure metrology, Nature Communications, 2026. DOI: 10.1038/s41467-026-73052-w.
