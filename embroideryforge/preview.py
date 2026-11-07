"""针迹清洁预览：颜色与 stitch_plan / report / 内存 thread 对齐。

DST 本身不保存 RGB，回读后 threadlist 为空——必须用旁路色表。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from pyembroidery import COLOR_CHANGE, JUMP, STITCH, EmbPattern, read


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return (128, 128, 128)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def colors_from_layers(layers: Sequence[dict[str, Any]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for L in layers:
        if L.get("enabled", True) is False:
            continue
        hex_color = L.get("hex")
        if not hex_color:
            continue
        out.append(_hex_to_bgr(str(hex_color)))
    return out


def colors_from_pattern(pattern: EmbPattern) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for t in getattr(pattern, "threadlist", []) or []:
        colors.append((int(t.get_blue()), int(t.get_green()), int(t.get_red())))
    return colors


def resolve_colors(
    *,
    pattern: EmbPattern | None = None,
    layers: Sequence[dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
    plan_path: str | Path | None = None,
) -> list[tuple[int, int, int]]:
    """优先级：显式 layers → report → plan → pattern.threadlist。"""
    if layers:
        cols = colors_from_layers(layers)
        if cols:
            return cols
    for path in (report_path, plan_path):
        if path is None:
            continue
        p = Path(path)
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        cols = colors_from_layers(data.get("layers") or [])
        if cols:
            return cols
    if pattern is not None:
        cols = colors_from_pattern(pattern)
        if cols:
            return cols
    return [(80, 80, 80)]


# 预览底色：白底易放大行间空隙；深色绣布更接近机绣观感
PREVIEW_BG_WHITE = (255, 255, 255)
PREVIEW_BG_DARK = (22, 22, 26)


def _auto_stroke_width(seg_lens: list[float], *, thick: bool) -> int:
    """按中位针距估线宽；thick 用于缩略图成团（假厚）。"""
    if not seg_lens:
        return 3 if thick else 1
    med = float(np.median(np.asarray(seg_lens, dtype=np.float32)))
    if thick:
        # 假厚封顶宜保守；过粗会把色边界糊成马赛克墙
        return int(np.clip(round(med * 0.75), 2, 4))
    return int(np.clip(round(med * 0.35), 1, 3))


def render_clean_preview(
    pattern_or_dst: EmbPattern | str | Path,
    out_path: str | Path,
    *,
    colors: Sequence[tuple[int, int, int]] | None = None,
    layers: Sequence[dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
    plan_path: str | Path | None = None,
    pad: int = 40,
    background: tuple[int, int, int] = (255, 255, 255),
    stroke_width: int | None = None,
    thick: bool = False,
) -> dict[str, Any]:
    """只画 STITCH；JUMP / COLOR_CHANGE 断笔。颜色来自旁路色表。

    thick=True 或更大的 stroke_width：预览假厚（不改变 DST），缩略图更成团。
    """
    if isinstance(pattern_or_dst, EmbPattern):
        pattern = pattern_or_dst
    else:
        pattern = read(str(pattern_or_dst))
        dst = Path(pattern_or_dst)
        if report_path is None:
            cand = dst.with_name("auto_stitch_report.json")
            if cand.is_file():
                report_path = cand
        if plan_path is None:
            cand = dst.with_name("stitch_plan.json")
            if cand.is_file():
                plan_path = cand

    palette = list(colors) if colors else resolve_colors(
        pattern=pattern,
        layers=layers,
        report_path=report_path,
        plan_path=plan_path,
    )

    stitches = pattern.stitches
    xs = [s[0] for s in stitches if s[2] == STITCH]
    ys = [s[1] for s in stitches if s[2] == STITCH]
    if not xs:
        raise ValueError("no stitches to preview")

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = int(maxx - minx) + 2 * pad
    h = int(maxy - miny) + 2 * pad
    # 假厚预览 2× 超采样再缩小，减轻色块锯齿/马赛克感
    scale_up = 2 if thick else 1
    canvas = np.full((h * scale_up, w * scale_up, 3), background, dtype=np.uint8)

    # 先扫一遍估针距
    seg_lens: list[float] = []
    last_pt = None
    for x, y, cmd in stitches:
        if cmd in {COLOR_CHANGE, JUMP} or cmd != STITCH:
            last_pt = None
            continue
        if last_pt is not None:
            seg_lens.append(float(np.hypot(x - last_pt[0], y - last_pt[1])))
        last_pt = (x, y)
    width = stroke_width if stroke_width is not None else _auto_stroke_width(seg_lens, thick=thick)
    draw_w = max(1, width * scale_up)

    color_i = 0
    last = None
    stitch_n = 0
    for x, y, cmd in stitches:
        if cmd == COLOR_CHANGE:
            color_i = min(color_i + 1, max(len(palette) - 1, 0))
            last = None
            continue
        if cmd == JUMP:
            last = None
            continue
        if cmd != STITCH:
            last = None
            continue
        px = int((x - minx + pad) * scale_up)
        py = int((maxy - y + pad) * scale_up)
        color = palette[color_i % len(palette)]
        if last is not None:
            cv2.line(canvas, last, (px, py), color, draw_w, cv2.LINE_AA)
            if draw_w >= 3:
                cv2.circle(canvas, (px, py), max(1, draw_w // 2), color, -1, cv2.LINE_AA)
        last = (px, py)
        stitch_n += 1

    if scale_up > 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return {
        "stitches_drawn": stitch_n,
        "size": [w, h],
        "out": str(out_path),
        "colors_bgr": [list(c) for c in palette],
        "n_colors": len(palette),
        "stroke_width": width,
        "thick": bool(thick),
    }
