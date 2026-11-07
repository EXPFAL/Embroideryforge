"""Photo-Stitch v4：密度场短针（非色块 tatami）。

主张：亮度 → 针密（暗密亮疏）+ Bayer 落点 → 按线色串短跑针。
双轨预览：计划图 (plan) / 针迹 (stitch)。Wilcom Photo Flash 仅作上限参照。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyembroidery import EmbPattern, write_dst, write_png

from embroideryforge.autofill import (
    _hex_to_rgb_name,
    _running_polylines,
    _satin_polylines,
    _tatami_polylines,
    _to_emb_xy,
)
from embroideryforge.bg import prepare_subject
from embroideryforge.preview import PREVIEW_BG_DARK, PREVIEW_BG_WHITE, render_clean_preview
from embroideryforge.segment import segment_image

# 延迟导入 photo_stitch 工具，避免环依赖在模块加载期炸开
def _ps():
    from embroideryforge import photo_stitch as ps

    return ps


def density_map(gray: np.ndarray, mask: np.ndarray, *, gamma: float = 0.85) -> np.ndarray:
    inv = (255.0 - gray.astype(np.float32)) / 255.0
    dens = np.clip(inv**gamma, 0.0, 1.0)
    return np.where(mask > 0, dens, 0.0).astype(np.float32)


def render_density_preview(dens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = np.clip(dens * 255.0, 0, 255).astype(np.uint8)
    vis = np.where(mask > 0, vis, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def dither_keep_mask(dens: np.ndarray, mask: np.ndarray, *, floor: float = 0.08) -> np.ndarray:
    ps = _ps()
    h, w = dens.shape
    yy, xx = np.indices((h, w))
    thr = ps._BAYER8[yy % 8, xx % 8]
    keep = ((dens >= thr * 0.92) | (dens >= 0.78)) & (dens >= floor) & (mask > 0)
    # 高光软留针
    bright = (dens >= 0.05) & (dens < 0.28) & (mask > 0)
    soft = bright & (dens >= thr * 0.38)
    # 高光→中调过渡带：略加密，填中间层次
    mid_bright = (dens >= 0.18) & (dens < 0.45) & (mask > 0)
    bridge = mid_bright & (dens >= thr * 0.55)
    keep = keep | soft | bridge
    return keep.astype(np.uint8) * 255


def soften_highlight_centers(centers: list[np.ndarray]) -> list[np.ndarray]:
    """最亮色略拉向次亮/中肤，减少死白高光块。"""
    if len(centers) < 2:
        return centers
    lum = [float(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]) for c in centers]
    order = sorted(range(len(centers)), key=lambda i: lum[i])
    hi = order[-1]
    mid = order[-2] if len(order) >= 2 else order[0]
    # 仅当最亮明显偏亮时软化
    if lum[hi] - lum[mid] < 18:
        return centers
    out = [c.copy() for c in centers]
    a = out[hi].astype(np.float32)
    b = out[mid].astype(np.float32)
    out[hi] = np.clip(a * 0.62 + b * 0.38, 0, 255).astype(np.uint8)
    return out


def blend_hex_toward_bgr(hex_c: str, bridge_bgr: np.ndarray, *, t: float = 0.32) -> str:
    """发色向脸缘中位靠拢，减轻发际色差。"""
    ps = _ps()
    h = hex_c.lstrip("#")
    if len(h) != 6:
        return hex_c
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    # hex 是 RGB；bridge 是 BGR
    face = bridge_bgr.astype(np.float32)
    src = np.array([b, g, r], dtype=np.float32)  # to BGR
    mixed = np.clip(src * (1.0 - t) + face * t, 0, 255)
    return ps._bgr_to_hex(mixed)


def face_edge_bridge_bgr(work_r: np.ndarray, dens_mask: np.ndarray) -> np.ndarray | None:
    """脸罩边缘一圈的中位色，供发/帽色板衔接。"""
    if dens_mask is None or not np.any(dens_mask):
        return None
    er = cv2.erode(dens_mask, np.ones((5, 5), np.uint8), iterations=1)
    rim = cv2.bitwise_and(dens_mask, cv2.bitwise_not(er))
    if int(np.count_nonzero(rim)) < 40:
        rim = dens_mask
    pix = work_r[rim > 0]
    if len(pix) < 20:
        return None
    return np.median(pix, axis=0).astype(np.float32)


def soft_hairline_dens_mask(face_m: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
    """额-发过渡：脸罩上缘外扩一圈，密度场吃进发际，减少切色。"""
    if not np.any(face_m):
        return face_m.copy()
    rim = cv2.dilate(face_m, np.ones((9, 9), np.uint8), iterations=1)
    ys = np.where(face_m > 0)[0]
    y_cut = int(np.percentile(ys, 55)) if len(ys) else face_m.shape[0] // 2
    upper = np.zeros_like(face_m)
    upper[: max(1, y_cut + 1), :] = 255
    soft = cv2.bitwise_and(rim, upper)
    out = cv2.bitwise_or(face_m, soft)
    return cv2.bitwise_and(out, work_mask)


def face_skin_median_bgr(work_r: np.ndarray, face_m: np.ndarray) -> np.ndarray | None:
    fg = face_m > 0
    if int(fg.sum()) < 80:
        return None
    pix = work_r[fg]
    # 去掉过暗/过亮，取中位肤色
    lum = 0.114 * pix[:, 0] + 0.587 * pix[:, 1] + 0.299 * pix[:, 2]
    mid = (lum >= 55) & (lum <= 220)
    use = pix[mid] if int(mid.sum()) >= 40 else pix
    return np.median(use, axis=0).astype(np.float32)


def skin_exterior_mask(
    work_r: np.ndarray,
    face_m: np.ndarray,
    work_mask: np.ndarray,
    *,
    max_lab_dist: float = 28.0,
) -> np.ndarray:
    """脸外手足/颈肤：相对脸肤 Lab 近邻 + 非过暗，供密度场延伸。"""
    skin_ref = face_skin_median_bgr(work_r, face_m)
    if skin_ref is None:
        return np.zeros(work_mask.shape, dtype=np.uint8)
    exterior = (work_mask > 0) & (face_m == 0)
    if int(exterior.sum()) < 80:
        return np.zeros(work_mask.shape, dtype=np.uint8)
    lab = cv2.cvtColor(work_r, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(
        skin_ref.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - ref, axis=2)
    hsv = cv2.cvtColor(work_r, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 肤色带：偏橙黄；Lab 近邻为主，饱和只挡灰衣/荧光
    hue_ok = (h <= 40) | (h >= 160)
    cand = exterior & (dist <= max_lab_dist) & hue_ok & (s >= 8) & (s <= 170) & (v >= 60) & (v <= 240)
    out = (cand.astype(np.uint8) * 255)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 丢掉碎斑
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 60:
            cleaned[labels == i] = 255
    return cleaned


def chroma_accent_mask(
    work_r: np.ndarray,
    work_mask: np.ndarray,
    exclude: np.ndarray,
    *,
    min_s: int = 100,
) -> np.ndarray:
    """领章/徽章等高饱和色块（红为主）；丢掉手部碎红斑。"""
    hsv = cv2.cvtColor(work_r, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((h <= 12) | (h >= 168)) & (s >= min_s) & (v >= 55) & (v <= 230)
    out = ((work_mask > 0) & (exclude == 0) & red).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # 领章成块；滤掉 1–2px 噪点，保留小徽章
        # 领章通常小；上限放宽到画面 12%，避免测试/近景大红块被误杀
        if 18 <= area <= max(80, int(0.12 * work_mask.shape[0] * work_mask.shape[1])):
            cleaned[labels == i] = 255
    return cleaned


def dark_prop_mask(
    work_r: np.ndarray,
    work_mask: np.ndarray,
    exclude: np.ndarray,
    *,
    max_v: int = 95,
    max_s: int = 85,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    """枪械等深色低饱和道具；有手部 prior 时优先在先验内取暗块。"""
    hsv = cv2.cvtColor(work_r, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # 金属灰偏暗：V 可到 ~95；再扣掉过亮边缘
    dark = (v <= max_v) & (s <= max_s) & (v <= 110)
    base = (work_mask > 0) & (exclude == 0) & dark
    if prior is not None and np.any(prior):
        # 先验内放宽一点亮度，贴枪身高光边
        near = (prior > 0) & (work_mask > 0) & (exclude == 0) & (v <= max(max_v + 25, 120)) & (s <= max_s + 15)
        base = base | near
        # 先验外的大块深色衣影不抢 prop：仅保留与先验连通的成分
        out = (base.astype(np.uint8) * 255)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        min_area = max(40, int(0.0006 * work_mask.shape[0] * work_mask.shape[1]))
        for i in range(1, n):
            comp = labels == i
            if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
                continue
            if np.any(comp & (prior > 0)):
                cleaned[comp] = 255
        return cleaned
    out = (base.astype(np.uint8) * 255)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    cleaned = np.zeros_like(out)
    min_area = max(50, int(0.0008 * work_mask.shape[0] * work_mask.shape[1]))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def build_figure_region_masks(
    work_r: np.ndarray,
    face_m: np.ndarray,
    work_mask: np.ndarray,
    *,
    figure_geo: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """人物+服饰：脸外肤 / 点缀 / 道具；手罩优先 MediaPipe，失败回退色域。"""
    skin_color = skin_exterior_mask(work_r, face_m, work_mask)
    hand_m = np.zeros(work_mask.shape, dtype=np.uint8)
    prop_prior = None
    source = {"hands": "color", "pose": "none"}
    if figure_geo:
        hm = figure_geo.get("hand_mask")
        if hm is not None and hm.shape[:2] == work_mask.shape[:2] and np.any(hm):
            hand_m = cv2.bitwise_and(hm, work_mask)
            hand_m = cv2.bitwise_and(hand_m, cv2.bitwise_not(face_m))
            source["hands"] = str((figure_geo.get("source") or {}).get("hands") or "mediapipe")
        ps = figure_geo.get("prop_seed")
        if ps is not None and ps.shape[:2] == work_mask.shape[:2]:
            prop_prior = ps
        source["pose"] = str((figure_geo.get("source") or {}).get("pose") or "none")
    # MediaPipe 手罩 ∪ 色域肤（颈等手模型盖不到的）
    skin_x = cv2.bitwise_or(hand_m, skin_color)
    skin_x = cv2.bitwise_and(skin_x, work_mask)
    skin_x = cv2.bitwise_and(skin_x, cv2.bitwise_not(face_m))
    dens_seed = cv2.bitwise_or(face_m, skin_x)
    accent = chroma_accent_mask(work_r, work_mask, dens_seed)
    prop = dark_prop_mask(
        work_r, work_mask, cv2.bitwise_or(dens_seed, accent), prior=prop_prior
    )
    return {
        "skin_exterior": skin_x,
        "accent": accent,
        "prop": prop,
        "hand_mask": hand_m,
        "source": source,
    }


def sample_keep_grid(
    keep: np.ndarray,
    dens: np.ndarray,
    mask: np.ndarray,
    step: float,
    *,
    core_mask: np.ndarray | None = None,
    core_step_scale: float = 0.55,
) -> np.ndarray:
    h, w = keep.shape
    step_i = max(1, int(round(step)))
    grid = np.zeros_like(keep)
    grid[0:h:step_i, 0:w:step_i] = 255
    out = cv2.bitwise_and(keep, grid)
    dark = (dens >= 0.55) & (mask > 0)
    fine_step = max(1, step_i - 1)
    fine = np.zeros_like(keep)
    fine[0:h:fine_step, 0:w:fine_step] = 255
    out = np.where(dark, np.maximum(out, cv2.bitwise_and(fine, (mask > 0).astype(np.uint8) * 255)), out)
    # 五官核：半步相位网格 + 放宽 dens 门限（step≈1 时仍能加针）
    if core_mask is not None and np.any(core_mask):
        cstep = max(1, int(round(max(1.0, step) * core_step_scale)))
        cgrid = np.zeros_like(keep)
        cgrid[0:h:cstep, 0:w:cstep] = 255
        off = max(1, cstep // 2) if cstep >= 2 else 1
        if cstep >= 2:
            cgrid[off:h:cstep, off:w:cstep] = 255
        else:
            # 已是 1px 主网格：对角半相再采一趟
            cgrid[1:h:2, 1:w:2] = 255
        core = (core_mask > 0) & (mask > 0)
        relaxed = (keep > 0) | ((dens >= 0.32) & core)
        extra = (cgrid > 0) & relaxed
        out = np.where(core, np.maximum(out, extra.astype(np.uint8) * 255), out)
    return out.astype(np.uint8)


def feature_core_mask(
    gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    percentile: float = 16.0,
) -> np.ndarray:
    """眼/眉/嘴等暗核：分位暗部 + 局部对比，限制在脸窗中上部。"""
    fg = face_mask > 0
    if int(fg.sum()) < 80:
        return np.zeros_like(face_mask)
    vals = gray[fg]
    med = float(np.median(vals))
    p_lo = float(np.percentile(vals, percentile))
    # 相对中位下沉，避免平铺亮肤把分位抬到肤色导致整脸成核
    thr = min(p_lo, med - 18.0)
    dark = fg & (gray.astype(np.float32) <= thr)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
    local = fg & ((blur.astype(np.int16) - gray.astype(np.int16)) >= 12)
    u8 = ((dark | local).astype(np.uint8) * 255)
    # 只用脸窗上 70% 高度，少把下巴阴影当核
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    cut = y0 + int((y1 - y0) * 0.72)
    band = np.zeros_like(u8)
    band[: cut + 1, :] = 255
    u8 = cv2.bitwise_and(u8, band)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros_like(u8)
    face_area = int(fg.sum())
    max_comp = max(350, int(face_area * 0.05))
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 8 <= a <= max_comp:
            keep[markers == i] = 255
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return keep


def highlight_preserve_mask(
    gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    percentile: float = 90.0,
) -> np.ndarray:
    """眼白/牙/鼻尖等亮核：相对中位上浮的连通小块，用于抠空保高光。"""
    fg = face_mask > 0
    if int(fg.sum()) < 80:
        return np.zeros_like(face_mask)
    vals = gray[fg]
    med = float(np.median(vals))
    p_hi = float(np.percentile(vals, percentile))
    thr = max(p_hi, med + 22.0)
    bright = fg & (gray.astype(np.float32) >= thr)
    # 局部比周围亮
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    local = fg & ((gray.astype(np.int16) - blur.astype(np.int16)) >= 14)
    u8 = ((bright | local).astype(np.uint8) * 255)
    u8 = cv2.bitwise_and(u8, (face_mask > 0).astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros_like(u8)
    face_area = int(fg.sum())
    max_comp = max(280, int(face_area * 0.035))
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 6 <= a <= max_comp:
            keep[markers == i] = 255
    keep = cv2.dilate(keep, np.ones((2, 2), np.uint8), iterations=1)
    return keep


def structure_band_mask(core: np.ndarray) -> np.ndarray:
    """五官核边缘窄带：只勾边，不填核心（避免 satin 把眼窝涂成黑块）。"""
    if core is None or not np.any(core):
        return np.zeros((1, 1), dtype=np.uint8)
    edge = cv2.morphologyEx(core, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    # 略加粗一圈便于成绣可见，仍保持空心
    band = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    return band


def _dominant_edge_angle(gray: np.ndarray, mask: np.ndarray) -> float:
    """结构针主方向：沿边缘（梯度法向 + 90°）。"""
    m = mask > 0
    if int(m.sum()) < 20:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy[m], gx[m])  # 法向
    # 沿边 = 法向转 90°
    along = ang + np.pi / 2.0
    # 主方向：圆均值
    c = float(np.mean(np.cos(along * 2.0)))
    s = float(np.mean(np.sin(along * 2.0)))
    deg = float(np.degrees(0.5 * np.arctan2(s, c)))
    return deg


def points_to_row_runs(
    ys: np.ndarray, xs: np.ndarray, *, max_gap: int = 3
) -> list[list[tuple[float, float]]]:
    """兼容旧接口 → 蛇形串针。"""
    return points_to_serpentine_runs(ys, xs, max_gap=max_gap)


def points_to_serpentine_runs(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    max_gap: int = 3,
    max_row_jump: int = 2,
    max_connect_dx: int | None = None,
) -> list[list[tuple[float, float]]]:
    """同行串段 + 隔行反向衔接，减少 JUMP。"""
    if len(xs) == 0:
        return []
    from collections import defaultdict

    rows: dict[int, list[int]] = defaultdict(list)
    for y, x in zip(ys.tolist(), xs.tolist()):
        rows[int(y)].append(int(x))
    connect_dx = max_connect_dx if max_connect_dx is not None else max_gap * 4
    polys: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    direction = 1  # +1 L→R, -1 R→L

    def _flush() -> None:
        nonlocal current
        if len(current) >= 2:
            polys.append(current)
        elif len(current) == 1:
            x0, y0 = current[0]
            polys.append([(x0, y0), (x0 + 1.0, y0)])
        current = []

    for y in sorted(rows.keys()):
        xs_row = sorted(set(rows[y]))
        if direction < 0:
            xs_row = list(reversed(xs_row))
        # 行内按 gap 切段
        segs: list[list[int]] = []
        seg = [xs_row[0]]
        for x in xs_row[1:]:
            if abs(x - seg[-1]) <= max_gap:
                seg.append(x)
            else:
                segs.append(seg)
                seg = [x]
        segs.append(seg)
        for seg_xs in segs:
            pts = [(float(x), float(y)) for x in seg_xs]
            if len(pts) == 1:
                pts = [pts[0], (pts[0][0] + float(direction), pts[0][1])]
            if not current:
                current = pts
                continue
            lx, ly = current[-1]
            fx, fy = pts[0]
            if abs(int(fy) - int(ly)) <= max_row_jump and abs(fx - lx) <= connect_dx:
                current.extend(pts)
            else:
                _flush()
                current = pts
        direction *= -1
    _flush()
    return polys


def _emit_polys(
    pattern: EmbPattern,
    polys: list[list[tuple[float, float]]],
    *,
    hex_c: str,
    scale: float,
    height: int,
    first: bool,
) -> tuple[bool, int]:
    if not polys:
        return first, 0
    if not first:
        pattern.color_change()
    pattern.add_thread(_hex_to_rgb_name(hex_c))
    n_pts = 0
    for poly in polys:
        block = [_to_emb_xy(x, y, scale, height) for x, y in poly]
        if len(block) < 2:
            continue
        pattern.move_abs(block[0][0], block[0][1])
        for x, y in block:
            pattern.stitch_abs(x, y)
            n_pts += 1
    return False, n_pts


def emit_density_layers(
    pattern: EmbPattern,
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    n_colors: int,
    step: float,
    target_height_mm: float,
    first: bool,
    detail_boost: bool = False,
    hand_mask: np.ndarray | None = None,
    face_only_mask: np.ndarray | None = None,
) -> tuple[EmbPattern, list[dict[str, Any]], bool, dict[str, np.ndarray]]:
    ps = _ps()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 半身脸小：少平滑、多色阶，减轻糊感
    n_q = int(np.clip(int(n_colors) + (2 if detail_boost else 1), 8, 12))
    tone, label_map, centers = ps.quantize_tones(
        bgr, mask, n_colors=n_q, pre_smooth=not detail_boost
    )
    dens = density_map(gray, mask, gamma=0.70 if detail_boost else 0.78)
    core_pct = 14.0 if detail_boost else 22.0
    # 五官核优先在真脸罩上算，避免手区脏核
    core_src = face_only_mask if face_only_mask is not None and np.any(face_only_mask) else mask
    core = feature_core_mask(gray, core_src, percentile=core_pct)
    dens_use = dens.copy()
    face = mask > 0
    # 中调微抬
    mid = face & (dens >= 0.28) & (dens <= 0.82)
    boost_m = 1.18 if detail_boost else 1.1
    dens_use[mid] = np.minimum(1.0, dens_use[mid] * boost_m + (0.08 if detail_boost else 0.05))
    # 高光→中调：把极亮压到可留针区间，保留层次而非死白
    bright = face & (dens < 0.32)
    if np.any(bright):
        d = dens[bright]
        # 0..0.32 → 0.12..0.40
        dens_use[bright] = np.clip(0.12 + (d / 0.32) * 0.28, 0.12, 0.42).astype(np.float32)
    if np.any(core):
        boost = core > 0
        dens_use[boost] = np.minimum(1.0, dens_use[boost] * (1.45 if detail_boost else 1.3) + (0.18 if detail_boost else 0.14))
    centers = soften_highlight_centers(list(centers))
    # tone 同步最亮色软化
    for i, c in enumerate(centers):
        tone[label_map == i] = c
    step_scale = 0.72 if detail_boost else 0.85
    face_step = max(1.0, float(step) * step_scale)
    keep = sample_keep_grid(
        dither_keep_mask(dens_use, mask, floor=0.04 if detail_boost else 0.05),
        dens_use,
        mask,
        face_step,
        core_mask=core,
        core_step_scale=0.35 if detail_boost else 0.45,
    )
    # 半身：全脸 + 手区再叠半相网格，逼近 1px 密度
    if detail_boost:
        h0, w0 = keep.shape
        phase = np.zeros_like(keep)
        phase[0:h0:1, 0:w0:1] = 255
        phase[1:h0:2, 1:w0:2] = 255
        keep = np.maximum(
            keep,
            ((phase > 0) & (mask > 0) & (dither_keep_mask(dens_use, mask, floor=0.04) > 0)).astype(
                np.uint8
            )
            * 255,
        )
    if hand_mask is not None and np.any(hand_mask):
        hm = (hand_mask > 0) & (mask > 0)
        hkeep = dither_keep_mask(dens_use, (hm.astype(np.uint8) * 255), floor=0.03)
        keep = np.where(hm, np.maximum(keep, hkeep), keep)
    h = mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    step_i = max(1, int(round(face_step)))
    gap = max(2, step_i + 1)
    layers: list[dict[str, Any]] = []
    order = sorted(
        range(len(centers)),
        key=lambda i: float(0.114 * centers[i][0] + 0.587 * centers[i][1] + 0.299 * centers[i][2]),
    )
    for idx in order:
        ys, xs = np.where((label_map == idx) & (keep > 0))
        if len(xs) < 8:
            continue
        polys = points_to_serpentine_runs(
            ys,
            xs,
            max_gap=gap,
            max_row_jump=1,
            max_connect_dx=gap,
        )
        hex_c = ps._bgr_to_hex(centers[idx])
        first, n_pts = _emit_polys(
            pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
        )
        if n_pts >= 2:
            layers.append(
                {
                    "tag": f"dens_{idx}",
                    "hex": hex_c,
                    "bgr": [int(x) for x in centers[idx]],
                    "stitch": "density_serpentine",
                    "points": n_pts,
                    "segments": len(polys),
                    "enabled": True,
                }
            )
    maps = {"dens": dens, "keep": keep, "tone": tone, "core": core, "gray": gray}
    return pattern, layers, first, maps


def emit_feature_structure(
    pattern: EmbPattern,
    gray: np.ndarray,
    core: np.ndarray,
    face_mask: np.ndarray,
    *,
    target_height_mm: float,
    first: bool,
    hex_c: str = "#1a1210",
) -> tuple[EmbPattern, list[dict[str, Any]], bool, np.ndarray]:
    """五官结构针：核边缘窄 satin + 轮廓 running，沿局部梯度方向。"""
    ps = _ps()
    layers: list[dict[str, Any]] = []
    band = structure_band_mask(core)
    if band.shape != face_mask.shape:
        band = np.zeros_like(face_mask)
    band = cv2.bitwise_and(band, (face_mask > 0).astype(np.uint8) * 255)
    band = ps.clean_layer_mask(band, min_area=10, harden=False)
    if int(np.count_nonzero(band)) < 24:
        return pattern, layers, first, band
    h = face_mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    angle = _dominant_edge_angle(gray, band)
    # 以轮廓 running 为主勾形；窄 satin 轻扫一次，避免重复填实
    run = ps.filter_short_polylines(
        _running_polylines(band, stitch_len=0.95),
        min_span=1.8,
    )
    satin = ps.filter_short_polylines(
        _satin_polylines(band, angle_deg=angle, spacing=0.7),
        min_span=1.8,
    )
    ys, _xs = np.where(face_mask > 0)
    y_cut = int(ys.min() + (ys.max() - ys.min()) * 0.55) if len(ys) else band.shape[0] // 2
    eye_band = band.copy()
    eye_band[y_cut:, :] = 0
    if int(np.count_nonzero(eye_band)) >= 16:
        run += ps.filter_short_polylines(
            _running_polylines(eye_band, stitch_len=0.85),
            min_span=1.5,
        )
    mouth_band = band.copy()
    mouth_band[:y_cut, :] = 0
    if int(np.count_nonzero(mouth_band)) >= 16:
        run += ps.filter_short_polylines(
            _running_polylines(mouth_band, stitch_len=0.85),
            min_span=1.5,
        )
    polys = run + satin
    first, n_pts = _emit_polys(
        pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
    )
    if n_pts >= 2:
        layers.append(
            {
                "tag": "face_structure",
                "hex": hex_c,
                "stitch": "feature_satin_run",
                "angle_deg": round(angle, 1),
                "points": n_pts,
                "segments": len(polys),
                "enabled": True,
                "role": "feature_structure",
            }
        )
    return pattern, layers, first, band


def _emit_tone_band_tatami(
    pattern: EmbPattern,
    work_r: np.ndarray,
    region: np.ndarray,
    *,
    spacing: float,
    stitch_len: float,
    scale: float,
    height: int,
    first: bool,
    angle: float,
    tag_prefix: str,
    n_bands: int = 3,
    fold_edges: bool = False,
) -> tuple[EmbPattern, list[dict[str, Any]], bool, np.ndarray]:
    """衣褶：同区按亮度拆多带 tatami；可选暗褶 running 勾边。"""
    ps = _ps()
    layers: list[dict[str, Any]] = []
    covered = np.zeros(region.shape, dtype=np.uint8)
    fg = region > 0
    if int(fg.sum()) < 80:
        return pattern, layers, first, covered
    gray = cv2.cvtColor(work_r, cv2.COLOR_BGR2GRAY)
    vals = gray[fg]
    n_bands = int(np.clip(n_bands, 2, 4))
    edges = np.percentile(vals, np.linspace(0, 100, n_bands + 1))
    for bi in range(n_bands):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        if bi == n_bands - 1:
            band = fg & (gray >= lo) & (gray <= hi)
        else:
            band = fg & (gray >= lo) & (gray < hi)
        mask = (band.astype(np.uint8) * 255)
        mask = ps.clean_layer_mask(mask, min_area=30, harden=False)
        if int(np.count_nonzero(mask)) < 50:
            continue
        pix = work_r[mask > 0]
        hex_c = ps._bgr_to_hex(np.median(pix, axis=0))
        # 暗带略密、亮带略疏
        t = bi / max(n_bands - 1, 1)
        sp = spacing * (0.88 + 0.16 * t)
        sl = stitch_len * (1.02 + 0.06 * t)
        ang = angle + (bi - (n_bands - 1) / 2.0) * 5.0
        polys = ps.filter_short_polylines(
            _tatami_polylines(mask, angle_deg=ang, spacing=sp, stitch_len=sl),
            min_span=2.0,
        )
        first, n_pts = _emit_polys(
            pattern, polys, hex_c=hex_c, scale=scale, height=height, first=first
        )
        if n_pts >= 2:
            covered = cv2.bitwise_or(covered, mask)
            layers.append(
                {
                    "tag": f"{tag_prefix}_t{bi}",
                    "hex": hex_c,
                    "stitch": "tatami",
                    "angle_deg": ang,
                    "points": n_pts,
                    "segments": len(polys),
                    "enabled": True,
                    "role": "clothes_tone",
                }
            )
    if fold_edges and int(fg.sum()) >= 200:
        # 衣褶：梯度暗边 running，补动作褶皱感
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        thr = float(np.percentile(mag[fg], 72))
        edge = ((mag >= thr) & fg & (gray <= float(np.percentile(vals, 55)))).astype(np.uint8) * 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if int(np.count_nonzero(edge)) >= 40:
            pix = work_r[edge > 0]
            hex_c = ps._bgr_to_hex(np.median(pix, axis=0) * 0.75)
            polys = ps.filter_short_polylines(
                _running_polylines(edge, stitch_len=max(1.1, stitch_len * 0.7)),
                min_span=1.6,
            )
            first, n_pts = _emit_polys(
                pattern, polys, hex_c=hex_c, scale=scale, height=height, first=first
            )
            if n_pts >= 2:
                layers.append(
                    {
                        "tag": f"{tag_prefix}_fold",
                        "hex": hex_c,
                        "stitch": "running",
                        "angle_deg": 0.0,
                        "points": n_pts,
                        "segments": len(polys),
                        "enabled": True,
                        "role": "clothes_fold",
                    }
                )
    return pattern, layers, first, covered


def emit_body_tatami(
    pattern: EmbPattern,
    work_r: np.ndarray,
    work_mask: np.ndarray,
    face_dil: np.ndarray,
    subject_path: Path,
    *,
    spacing: float,
    stitch_len: float,
    target_height_mm: float,
    first: bool,
    n_colors: int,
    bridge_bgr: np.ndarray | None = None,
    region_masks: dict[str, np.ndarray] | None = None,
    detail_boost: bool = False,
    finger_polys: list[list[tuple[float, float]]] | None = None,
) -> tuple[EmbPattern, list[dict[str, Any]], bool]:
    """v4.1 衣帽 tatami（无 fold/指骨/道具轮廓叠针；半身请走 V6 unified）。"""
    _ = detail_boost, finger_polys, region_masks  # 叠针已回退，参数保留兼容
    ps = _ps()
    seg = segment_image(subject_path, mode="palette", k=max(8, n_colors), upscale=1, domain="portrait")
    label_img = seg["label_img"]
    label_tags = list(seg["label_tags"])
    lab_r = cv2.resize(
        label_img.astype(np.int32),
        (work_r.shape[1], work_r.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    hex_by_tag = {str(L["tag"]): str(L.get("hex") or "#808080") for L in seg["layers"]}
    h = work_r.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    layers: list[dict[str, Any]] = []
    covered = np.zeros(work_mask.shape, dtype=np.uint8)
    extra_excl = face_dil.copy()

    for tag in ps.BODY_TAGS:
        if tag not in label_tags:
            continue
        tid = label_tags.index(tag)
        body = ((lab_r == tid) & (work_mask > 0) & (extra_excl == 0) & (covered == 0)).astype(
            np.uint8
        ) * 255
        body = ps.clean_layer_mask(body, min_area=50, harden=False)
        if int(np.count_nonzero(body)) < 80:
            continue
        if tag == "clothes":
            pattern, tone_layers, first, cov = _emit_tone_band_tatami(
                pattern,
                work_r,
                body,
                spacing=spacing,
                stitch_len=stitch_len,
                scale=scale,
                height=h,
                first=first,
                angle=0.0,
                tag_prefix="clothes",
                n_bands=3,
                fold_edges=False,
            )
            layers.extend(tone_layers)
            covered = cv2.bitwise_or(covered, cov)
            continue
        hex_c = hex_by_tag.get(tag, "#808080")
        if bridge_bgr is not None and tag in ("hair", "hat"):
            hex_c = blend_hex_toward_bgr(hex_c, bridge_bgr, t=0.28 if tag == "hair" else 0.18)
        angle = {"clothes": 0.0, "hat": 32.0, "hair": 58.0}.get(tag, 0.0)
        sp = spacing * (0.95 if tag == "hair" else 1.1)
        sl = stitch_len * (1.05 if tag == "hair" else 1.15)
        polys = ps.filter_short_polylines(
            _tatami_polylines(body, angle_deg=angle, spacing=sp, stitch_len=sl),
            min_span=2.5,
        )
        first, n_pts = _emit_polys(
            pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
        )
        if n_pts >= 2:
            covered = cv2.bitwise_or(covered, body)
            layers.append(
                {
                    "tag": tag,
                    "hex": hex_c,
                    "stitch": "tatami",
                    "angle_deg": angle,
                    "points": n_pts,
                    "segments": len(polys),
                    "enabled": True,
                    "role": "body_fill",
                }
            )
    rest = ((work_mask > 0) & (extra_excl == 0) & (covered == 0)).astype(np.uint8) * 255
    rest = ps.clean_layer_mask(rest, min_area=80, harden=False)
    if int(np.count_nonzero(rest)) >= 120:
        pattern, rest_layers, first, cov = _emit_tone_band_tatami(
            pattern,
            work_r,
            rest,
            spacing=spacing * 1.05,
            stitch_len=stitch_len * 1.1,
            scale=scale,
            height=h,
            first=first,
            angle=12.0,
            tag_prefix="body_rest",
            n_bands=3,
            fold_edges=False,
        )
        if rest_layers:
            layers.extend(rest_layers)
        else:
            pix = work_r[rest > 0]
            med = np.median(pix, axis=0) if len(pix) else np.array([40, 40, 40])
            hex_c = ps._bgr_to_hex(med)
            polys = ps.filter_short_polylines(
                _tatami_polylines(
                    rest, angle_deg=12.0, spacing=spacing * 1.15, stitch_len=stitch_len * 1.2
                ),
                min_span=2.8,
            )
            first, n_pts = _emit_polys(
                pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
            )
            if n_pts >= 2:
                layers.append(
                    {
                        "tag": "body_rest",
                        "hex": hex_c,
                        "stitch": "tatami",
                        "angle_deg": 12.0,
                        "points": n_pts,
                        "segments": len(polys),
                        "enabled": True,
                        "role": "body_fill",
                    }
                )
    return pattern, layers, first


def run_density_v4(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    n_colors: int = 10,
    resolution: str = "high",
    target_height_mm: float = 70.0,
    bg_mode: str = "auto",
    hybrid_body: bool = True,
) -> dict[str, Any]:
    ps = _ps()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(image_path)

    subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    subject_path = out_dir / "subject.png"
    cv2.imwrite(str(subject_path), subject_bgr)
    cv2.imwrite(str(out_dir / "subject_mask.png"), subject_mask)

    cfg = dict(ps.RESOLUTION.get(resolution, ps.RESOLUTION["high"]))
    cfg["max_side"] = max(int(cfg["max_side"]), 400)
    spacing = float(cfg["spacing"])
    stitch_len = float(cfg["stitch_len"])
    step = max(1.15, spacing * 1.05)
    n_colors = int(np.clip(n_colors, 8, 12))

    geo = ps.detect_face_geometry(subject_bgr, subject_mask=subject_mask)
    roi_src = geo["roi"]
    face_landmarks_src = geo.get("landmarks")
    face_source = str(geo.get("source") or "unknown")
    boxed = subject_bgr.copy()
    x0, y0, w0, h0 = roi_src
    cv2.rectangle(boxed, (x0, y0), (x0 + w0, y0 + h0), (0, 180, 255), 2)
    if face_landmarks_src is not None:
        for px, py in face_landmarks_src[::8]:
            cv2.circle(boxed, (int(px), int(py)), 1, (80, 220, 80), -1)
    cv2.imwrite(str(out_dir / "roi_box.png"), boxed)

    sub_px = max(1, int(np.count_nonzero(subject_mask)))
    face_frac = float(w0 * h0) / float(sub_px)

    work_r, resize_scale = ps.resize_for_resolution(
        subject_bgr, resolution, max_side=int(cfg["max_side"])
    )
    work_mask_r = cv2.resize(
        subject_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    work_mask_r = ps._strip_fringe_white(work_r, work_mask_r)
    roi = ps._scale_roi(roi_src, subject_bgr.shape[:2], work_r.shape[:2])
    lm_r = ps._scale_landmarks(face_landmarks_src, subject_bgr.shape[:2], work_r.shape[:2])
    face_m = ps._face_mask(
        work_r.shape[:2], roi, subject_mask=work_mask_r, landmarks=lm_r
    )
    face_m = cv2.bitwise_and(face_m, work_mask_r)
    # v4.1.1：脸窗密度 + 软发际；衣帽 tatami（无叠针）
    dens_mask = soft_hairline_dens_mask(face_m, work_mask_r)
    dens_mask = cv2.bitwise_and(dens_mask, work_mask_r)
    bridge = face_edge_bridge_bgr(work_r, dens_mask)

    pattern = EmbPattern()
    layers: list[dict[str, Any]] = []
    first = True
    if hybrid_body:
        pattern, body_layers, first = emit_body_tatami(
            pattern,
            work_r,
            work_mask_r,
            dens_mask,
            subject_path,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            first=first,
            n_colors=n_colors,
            bridge_bgr=bridge,
        )
        layers.extend(body_layers)

    pattern, dens_layers, first, maps = emit_density_layers(
        pattern,
        work_r,
        dens_mask,
        n_colors=n_colors,
        step=step,
        target_height_mm=target_height_mm,
        first=first,
        detail_boost=False,
        face_only_mask=face_m,
    )
    layers.extend(dens_layers)
    pattern.end()

    dens, keep, tone = maps["dens"], maps["keep"], maps["tone"]
    core = maps.get("core")
    dens_vis = render_density_preview(dens, dens_mask)
    plan = work_r.copy()
    plan[work_mask_r == 0] = PREVIEW_BG_DARK
    face_tone = tone.copy()
    mix = (tone.astype(np.float32) * 0.55).astype(np.uint8)
    face_tone = np.where(keep[..., None] > 0, mix, face_tone)
    plan = np.where(dens_mask[..., None] > 0, face_tone, plan)
    plan = np.where(work_mask_r[..., None] == 0, np.array(PREVIEW_BG_DARK, dtype=np.uint8), plan)
    if core is not None and np.any(core):
        cv2.imwrite(str(out_dir / "feature_core.png"), core)
    cv2.imwrite(str(out_dir / "face_mask.png"), face_m)
    cv2.imwrite(str(out_dir / "dens_mask.png"), dens_mask)
    cv2.imwrite(str(out_dir / "density_map.png"), dens_vis)
    cv2.imwrite(str(out_dir / "plan_preview.png"), plan)
    cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
    cv2.imwrite(str(out_dir / "keep_mask.png"), keep)

    dst = out_dir / "auto.dst"
    write_dst(pattern, str(dst))
    write_png(pattern, str(out_dir / "auto_pyembroidery.png"))
    # thin=白底验针；thick=深色绣布主预览（行间空隙更不刺眼）
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thin.png",
        layers=layers,
        thick=False,
        background=PREVIEW_BG_WHITE,
    )
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thick.png",
        layers=layers,
        thick=True,
        background=PREVIEW_BG_DARK,
    )
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thick_white.png",
        layers=layers,
        thick=True,
        background=PREVIEW_BG_WHITE,
    )
    try:
        shutil.copyfile(out_dir / "plan_preview.png", out_dir / "auto_clean_preview.png")
    except OSError:
        # Windows 下预览被占用时跳过，不影响 DST
        pass

    thick_img = cv2.imread(str(out_dir / "auto_stitch_thick.png"))
    if thick_img is None:
        thick_img = plan
    compare = np.hstack(
        [
            ps._fit_panel(work_r),
            ps._fit_panel(dens_vis),
            ps._fit_panel(plan),
            ps._fit_panel(thick_img),
        ]
    )
    cv2.imwrite(str(out_dir / "compare_preview.png"), compare)

    report = {
        "stitches": pattern.count_stitches(),
        "color_changes": pattern.count_color_changes(),
        "layers": layers,
        "engine": "photo_stitch_density_v4_1",
        "target_height_mm": target_height_mm,
    }
    meta = {
        "engine": "photo_stitch_density_v4_1",
        "mode": "density_hybrid" if hybrid_body else "density_face",
        "figure_regions": {
            "face_frac": round(float(face_frac), 4),
            "detail_boost": False,
            "overlays_rolled_back": True,
            "enabled": False,
        },
        "n_colors": n_colors,
        "resolution": resolution,
        "step_px": step,
        "resize_scale": resize_scale,
        "roi": {
            "mode": "face",
            "xywh": [int(x0), int(y0), int(w0), int(h0)],
            "detector": face_source,
        },
        "face_detect": {"source": face_source, "mediapipe": face_source == "mediapipe"},
        "bg": bg_meta,
        "palette": [L.get("hex") for L in layers if L.get("hex")],
        "layers": layers,
        "stitches": report["stitches"],
        "color_changes": report["color_changes"],
        "v41": {
            "serpentine": True,
            "feature_core": True,
            "neighbor_only_connect": True,
            "face_step_scale": 0.85,
        },
        "v42": {
            "highlight_preserve": False,
            "feature_structure": False,
            "rolled_back": True,
            "reason": "structure/highlight worse than v4.1.1 on portrait/meme",
        },
        "preview": {
            "plan": "plan_preview.png",
            "density": "density_map.png",
            "core": "feature_core.png",
            "stitch_thin": "auto_stitch_thin.png",
            "stitch_thick": "auto_stitch_thick.png",
            "stitch_thick_white": "auto_stitch_thick_white.png",
            "dual_track": True,
            "main_thumb": "auto_stitch_thick.png",
            "fabric_bg": "dark",
        },
        "note": "v4.1.1 + dark fabric preview; white thick kept as auto_stitch_thick_white.png",
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps({**report, "density_field": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "dst": dst, "out_dir": out_dir, "report": report, "meta": meta}


def run_density_v6_unified(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    n_colors: int = 11,
    resolution: str = "high",
    target_height_mm: float = 70.0,
    bg_mode: str = "auto",
) -> dict[str, Any]:
    """V6：整主体统一彩色密度短针（Wilcom Color PhotoStitch 隐喻）；无 body tatami / 叠针。"""
    ps = _ps()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(image_path)

    subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    subject_path = out_dir / "subject.png"
    cv2.imwrite(str(subject_path), subject_bgr)
    cv2.imwrite(str(out_dir / "subject_mask.png"), subject_mask)

    cfg = dict(ps.RESOLUTION.get(resolution, ps.RESOLUTION["high"]))
    cfg["max_side"] = max(int(cfg["max_side"]), 520)
    spacing = float(cfg["spacing"]) * 0.92
    step = max(1.05, spacing * 1.0)
    n_colors = int(np.clip(max(n_colors, 10), 8, 12))

    geo = ps.detect_face_geometry(subject_bgr, subject_mask=subject_mask)
    roi_src = geo["roi"]
    face_landmarks_src = geo.get("landmarks")
    face_source = str(geo.get("source") or "unknown")
    x0, y0, w0, h0 = roi_src
    boxed = subject_bgr.copy()
    cv2.rectangle(boxed, (x0, y0), (x0 + w0, y0 + h0), (0, 180, 255), 2)
    cv2.imwrite(str(out_dir / "roi_box.png"), boxed)

    sub_px = max(1, int(np.count_nonzero(subject_mask)))
    face_frac = float(w0 * h0) / float(sub_px)

    work_r, resize_scale = ps.resize_for_resolution(
        subject_bgr, resolution, max_side=int(cfg["max_side"])
    )
    work_mask_r = cv2.resize(
        subject_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    work_mask_r = ps._strip_fringe_white(work_r, work_mask_r)
    # 整主体即密度罩
    dens_mask = work_mask_r.copy()

    roi = ps._scale_roi(roi_src, subject_bgr.shape[:2], work_r.shape[:2])
    lm_r = ps._scale_landmarks(face_landmarks_src, subject_bgr.shape[:2], work_r.shape[:2])
    face_m = ps._face_mask(
        work_r.shape[:2], roi, subject_mask=work_mask_r, landmarks=lm_r
    )
    face_m = cv2.bitwise_and(face_m, work_mask_r)

    hand_m = np.zeros(work_mask_r.shape, dtype=np.uint8)
    try:
        from embroideryforge.figure_detect import detect_figure_geometry

        figure_geo = detect_figure_geometry(
            subject_bgr, subject_mask=subject_mask, out_shape=work_r.shape[:2]
        )
        hm = figure_geo.get("hand_mask")
        if hm is not None and np.any(hm):
            hand_m = cv2.bitwise_and(hm, work_mask_r)
    except Exception:
        figure_geo = None

    pattern = EmbPattern()
    pattern, dens_layers, first, maps = emit_density_layers(
        pattern,
        work_r,
        dens_mask,
        n_colors=n_colors,
        step=step,
        target_height_mm=target_height_mm,
        first=True,
        detail_boost=True,
        hand_mask=hand_m if np.any(hand_m) else None,
        face_only_mask=face_m,
    )
    pattern.end()
    layers = dens_layers

    dens, keep, tone = maps["dens"], maps["keep"], maps["tone"]
    core = maps.get("core")
    dens_vis = render_density_preview(dens, dens_mask)
    plan = work_r.copy()
    plan[work_mask_r == 0] = PREVIEW_BG_DARK
    mix = (tone.astype(np.float32) * 0.55).astype(np.uint8)
    plan_tone = np.where(keep[..., None] > 0, mix, tone)
    plan = np.where(dens_mask[..., None] > 0, plan_tone, plan)
    plan = np.where(
        work_mask_r[..., None] == 0, np.array(PREVIEW_BG_DARK, dtype=np.uint8), plan
    )

    if core is not None and np.any(core):
        cv2.imwrite(str(out_dir / "feature_core.png"), core)
    cv2.imwrite(str(out_dir / "face_mask.png"), face_m)
    cv2.imwrite(str(out_dir / "hand_mask.png"), hand_m)
    cv2.imwrite(str(out_dir / "dens_mask.png"), dens_mask)
    cv2.imwrite(str(out_dir / "density_map.png"), dens_vis)
    cv2.imwrite(str(out_dir / "plan_preview.png"), plan)
    cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
    cv2.imwrite(str(out_dir / "keep_mask.png"), keep)
    layout = work_r.copy()
    layout[work_mask_r == 0] = PREVIEW_BG_DARK
    layout = np.where(
        dens_mask[..., None] > 0,
        (layout.astype(np.float32) * 0.65 + dens_vis.astype(np.float32) * 0.35).astype(np.uint8),
        layout,
    )
    if np.any(face_m):
        layout = np.where(
            face_m[..., None] > 0,
            (layout.astype(np.float32) * 0.6 + np.array([40, 140, 255]) * 0.4).astype(np.uint8),
            layout,
        )
    if np.any(hand_m):
        layout = np.where(
            hand_m[..., None] > 0,
            (layout.astype(np.float32) * 0.55 + np.array([80, 220, 120]) * 0.45).astype(np.uint8),
            layout,
        )
    cv2.imwrite(str(out_dir / "figure_layout.png"), layout)

    dst = out_dir / "auto.dst"
    write_dst(pattern, str(dst))
    write_png(pattern, str(out_dir / "auto_pyembroidery.png"))
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thin.png",
        layers=layers,
        thick=False,
        background=PREVIEW_BG_WHITE,
    )
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thick.png",
        layers=layers,
        thick=True,
        background=PREVIEW_BG_DARK,
    )
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thick_white.png",
        layers=layers,
        thick=True,
        background=PREVIEW_BG_WHITE,
    )
    try:
        shutil.copyfile(out_dir / "plan_preview.png", out_dir / "auto_clean_preview.png")
    except OSError:
        pass

    thick_img = cv2.imread(str(out_dir / "auto_stitch_thick.png"))
    if thick_img is None:
        thick_img = plan
    compare = np.hstack(
        [
            ps._fit_panel(work_r),
            ps._fit_panel(dens_vis),
            ps._fit_panel(plan),
            ps._fit_panel(thick_img),
        ]
    )
    cv2.imwrite(str(out_dir / "compare_preview.png"), compare)

    report = {
        "stitches": pattern.count_stitches(),
        "color_changes": pattern.count_color_changes(),
        "layers": layers,
        "engine": "photo_stitch_density_v6",
        "target_height_mm": target_height_mm,
    }
    meta = {
        "engine": "photo_stitch_density_v6",
        "mode": "density_unified",
        "n_colors": n_colors,
        "resolution": resolution,
        "step_px": step,
        "resize_scale": resize_scale,
        "face_frac": round(float(face_frac), 4),
        "n_hands": int((figure_geo or {}).get("n_hands") or 0),
        "roi": {
            "mode": "face",
            "xywh": [int(x0), int(y0), int(w0), int(h0)],
            "detector": face_source,
        },
        "face_detect": {"source": face_source, "mediapipe": face_source == "mediapipe"},
        "bg": bg_meta,
        "palette": [L.get("hex") for L in layers if L.get("hex")],
        "layers": layers,
        "stitches": report["stitches"],
        "color_changes": report["color_changes"],
        "v6": {
            "unified_subject_density": True,
            "no_body_tatami": True,
            "no_fold_overlays": True,
            "soft_weight_face_hand": True,
        },
        "preview": {
            "plan": "plan_preview.png",
            "density": "density_map.png",
            "core": "feature_core.png",
            "stitch_thin": "auto_stitch_thin.png",
            "stitch_thick": "auto_stitch_thick.png",
            "stitch_thick_white": "auto_stitch_thick_white.png",
            "dual_track": True,
            "main_thumb": "auto_stitch_thick.png",
            "fabric_bg": "dark",
        },
        "note": "V6 unified Color PhotoStitch-style; V5 hybrid overlays rolled back",
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps({**report, "density_field": True, "density_unified": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "dst": dst, "out_dir": out_dir, "report": report, "meta": meta}


def estimate_face_frac(image_path: str | Path, *, bg_mode: str = "auto") -> float:
    """粗算脸 ROI / 主体面积比，供 CLI 自动选 V6。"""
    ps = _ps()
    image_path = Path(image_path)
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        return 1.0
    subject_bgr, subject_mask, _ = prepare_subject(original, mode=bg_mode)
    geo = ps.detect_face_geometry(subject_bgr, subject_mask=subject_mask)
    x0, y0, w0, h0 = geo["roi"]
    sub_px = max(1, int(np.count_nonzero(subject_mask)))
    return float(w0 * h0) / float(sub_px)


def run_density_photo(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    n_colors: int = 10,
    resolution: str = "high",
    target_height_mm: float = 70.0,
    bg_mode: str = "auto",
    hybrid_body: bool = True,
    unified: bool | None = None,
) -> dict[str, Any]:
    """密度场分发：unified=True 强制 V6；None 时半身自动 V6；False 强制 v4.1.1。"""
    use_v6 = unified is True
    if unified is None and hybrid_body:
        try:
            use_v6 = estimate_face_frac(image_path, bg_mode=bg_mode) < 0.45
        except Exception:
            use_v6 = False
    if use_v6:
        return run_density_v6_unified(
            image_path,
            out_dir,
            n_colors=n_colors,
            resolution=resolution,
            target_height_mm=target_height_mm,
            bg_mode=bg_mode,
        )
    return run_density_v4(
        image_path,
        out_dir,
        n_colors=n_colors,
        resolution=resolution,
        target_height_mm=target_height_mm,
        bg_mode=bg_mode,
        hybrid_body=hybrid_body,
    )
