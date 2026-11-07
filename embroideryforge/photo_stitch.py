"""照片绣 Photo-Stitch（P1）：色调量化 + 短填针，对标 Wilcom 彩色照片绣形态。

非目标：追平 Wilcom；不替换 chibi / stitch-auto 基线。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyembroidery import EmbPattern, write_dst, write_png

from embroideryforge.autofill import (
    _hex_to_rgb_name,
    _running_polylines,
    _tatami_polylines,
    _to_emb_xy,
)
from embroideryforge.bg import prepare_subject
from embroideryforge.preview import render_clean_preview
from embroideryforge.segment import segment_image

BODY_TAGS = ("clothes", "hat", "hair")

RESOLUTION = {
    # stitch_len 略长、spacing 略紧：缩略图更成团，少「碎草」感
    "low": {"max_side": 160, "spacing": 2.0, "stitch_len": 2.6},
    "mid": {"max_side": 240, "spacing": 1.35, "stitch_len": 2.1},
    "high": {"max_side": 360, "spacing": 0.72, "stitch_len": 1.9},
}

# 8x8 Bayer（有序抖动），用于亮度→疏密
_BAYER8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
)


def _bgr_to_hex(bgr: np.ndarray | list[int]) -> str:
    b, g, r = (int(x) for x in bgr[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def detect_face_roi_haar(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Haar 回退：返回 (x, y, w, h)；失败则 None。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    # 取最大脸并外扩；下缘多留下巴/下颚，避免与 body 层撕开
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    pad_x, pad_top = int(w * 0.38), int(h * 0.48)
    pad_bot = int(h * 0.95)  # Haar 框常切在嘴下，需明显下扩
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_top)
    x1 = min(bgr.shape[1], x + w + pad_x)
    y1 = min(bgr.shape[0], y + h + pad_bot)
    return x0, y0, x1 - x0, y1 - y0


def detect_face_geometry(
    bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """优先 MediaPipe 关键点；失败回退 Haar + 下巴精炼。

    返回 dict: roi (x,y,w,h), landmarks (Nx2|None), source (mediapipe|haar|bust)
    """
    from embroideryforge.face_detect import detect_face_roi_mediapipe

    mp = detect_face_roi_mediapipe(bgr)
    if mp is not None:
        roi, pts = mp
        # MediaPipe 关键点已含下巴：禁止按主体 0.9 下拉 ROI（会吞发/肩进脸罩）
        return {"roi": roi, "landmarks": pts, "source": "mediapipe"}

    haar = detect_face_roi_haar(bgr)
    if haar is not None:
        roi = haar
        if subject_mask is not None:
            roi = refine_face_roi_to_chin(roi, subject_mask, chin_frac=0.88)
        return {"roi": roi, "landmarks": None, "source": "haar"}

    roi = _default_bust_roi(bgr)
    if subject_mask is not None:
        roi = refine_face_roi_to_chin(roi, subject_mask, chin_frac=0.88)
    return {"roi": roi, "landmarks": None, "source": "bust"}


def detect_face_roi(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """兼容旧接口：返回 (x, y, w, h)；几何失败则 None（极少）。"""
    geo = detect_face_geometry(bgr, subject_mask=None)
    return geo.get("roi")


def _default_bust_roi(bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = bgr.shape[:2]
    # 上中部半身窗（无人脸检测时的保底）
    x0, y0 = int(w * 0.15), int(h * 0.02)
    x1, y1 = int(w * 0.85), int(h * 0.72)
    return x0, y0, x1 - x0, y1 - y0


def refine_face_roi_to_chin(
    roi: tuple[int, int, int, int],
    subject_mask: np.ndarray,
    *,
    chin_frac: float = 0.88,
) -> tuple[int, int, int, int]:
    """按主体轮廓把脸窗下缘拉到下巴附近，避免 Haar 切在嘴上导致下颚进 body。"""
    x, y, w, h = roi
    ys, xs = np.where(subject_mask > 0)
    if len(ys) < 80:
        return roi
    sy0, sy1 = int(ys.min()), int(ys.max())
    sx0, sx1 = int(xs.min()), int(xs.max())
    chin_y = sy0 + int((sy1 - sy0) * chin_frac)
    y1 = max(y + h, chin_y)
    y1 = min(y1, subject_mask.shape[0])
    x0 = min(x, max(0, sx0 - 4))
    x1 = max(x + w, min(subject_mask.shape[1], sx1 + 4))
    return int(x0), int(y), int(x1 - x0), int(y1 - y)


def resize_for_resolution(
    bgr: np.ndarray, resolution: str, *, max_side: int | None = None
) -> tuple[np.ndarray, float]:
    cfg = RESOLUTION.get(resolution, RESOLUTION["high"])
    side = int(max_side if max_side is not None else cfg["max_side"])
    h, w = bgr.shape[:2]
    scale = side / max(h, w, 1)
    if scale >= 0.99:
        return bgr.copy(), 1.0
    out = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def quantize_tones(
    bgr: np.ndarray,
    mask: np.ndarray,
    n_colors: int = 9,
    *,
    pre_smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Lab k-means 量化。返回 (tone_bgr, label_map, centers_bgr)。"""
    n_colors = max(2, min(int(n_colors), 16))
    fg = mask > 0
    if int(fg.sum()) < 50:
        flat = np.full_like(bgr, 255)
        return flat, np.full(bgr.shape[:2], -1, dtype=np.int32), []

    src = cv2.bilateralFilter(bgr, 7, 55, 55) if pre_smooth else bgr
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[fg]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compact, labels, centers = cv2.kmeans(
        samples, n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
    centers_bgr = cv2.cvtColor(centers_u8.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)

    label_map = np.full(bgr.shape[:2], -1, dtype=np.int32)
    label_map[fg] = labels.ravel().astype(np.int32)
    tone = np.full_like(bgr, 255)
    for i, c in enumerate(centers_bgr):
        tone[label_map == i] = c
    return tone, label_map, [centers_bgr[i] for i in range(len(centers_bgr))]


def density_preview(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = np.where(mask > 0, 255 - gray, 255).astype(np.uint8)
    return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)


def _color_order_dark_first(centers_bgr: list[np.ndarray]) -> list[int]:
    lum = [float(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]) for c in centers_bgr]
    return sorted(range(len(centers_bgr)), key=lambda i: lum[i])


def apply_density_gate(
    mask: np.ndarray,
    gray: np.ndarray,
    *,
    keep_dark_pct: float = 0.85,
    highlight_only: bool = True,
) -> np.ndarray:
    """疏密门控。默认只疏化高光，中调/暗部保满，避免「脸被挖空」。"""
    if mask.shape != gray.shape:
        gray = cv2.resize(gray, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    out = mask.copy()
    fg = mask > 0
    if not np.any(fg):
        return out
    inv = (255.0 - gray.astype(np.float32)) / 255.0
    dens = np.clip(inv**0.90, 0.0, 1.0)
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    thr = _BAYER8[yy % 8, xx % 8]
    if highlight_only:
        # 只疏「极亮」高光，保留更多中亮肤色/浅衣，避免缩略图干涩
        keep = (inv >= 0.10) | (dens >= thr * 0.22)
    else:
        keep = (dens >= thr * (1.05 - 0.35 * keep_dark_pct)) | (inv >= 0.45)
    out[fg & ~keep] = 0
    return out


def dark_feature_mask(
    bgr: np.ndarray,
    face_mask: np.ndarray,
    *,
    percentile: float = 14.0,
) -> np.ndarray:
    """脸内最暗部 + 局部对比 → 五官/镜框强制层。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg = face_mask > 0
    if int(fg.sum()) < 80:
        return np.zeros(face_mask.shape, dtype=np.uint8)
    vals = gray[fg]
    thr = float(np.percentile(vals, percentile))
    dark = fg & (gray <= thr)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.5)
    local = fg & ((blur.astype(np.int16) - gray.astype(np.int16)) >= 14)
    u8 = ((dark | local).astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros_like(u8)
    face_area = int(fg.sum())
    max_comp = max(400, int(face_area * 0.06))
    min_comp = 10
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if min_comp <= a <= max_comp:
            keep[markers == i] = 255
    # 略加粗，保证短针可见
    keep = cv2.dilate(keep, np.ones((2, 2), np.uint8), iterations=1)
    return keep

def clean_layer_mask(
    mask: np.ndarray, *, min_area: int = 28, harden: bool = False
) -> np.ndarray:
    """闭运算填缝 + 去掉碎斑；harden 用于少色实填（更实心）。"""
    if harden:
        # 轻闭运算：填针缝，不把五官并成大色斑
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        min_area = max(min_area, 22)
    else:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, ncc):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            keep[markers == i] = 255
    return keep


def filter_short_polylines(
    polys: list[list[tuple[float, float]]], *, min_span: float = 2.4
) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for poly in polys:
        if len(poly) < 2:
            continue
        span = 0.0
        for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
            span += float(np.hypot(x1 - x0, y1 - y0))
        if span >= min_span or len(poly) >= 4:
            out.append(poly)
    return out


def estimate_block_angles(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    block: int = 20,
) -> dict[tuple[int, int], float]:
    """分块主方向（沿等亮度 ≈ 梯度法向），邻块平滑，减轻整脸百叶窗。"""
    if gray.shape != mask.shape:
        gray = cv2.resize(gray, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h, w = mask.shape
    raw: dict[tuple[int, int], float] = {}
    for by in range(0, h, block):
        for bx in range(0, w, block):
            y1, x1 = min(h, by + block), min(w, bx + block)
            m = mask[by:y1, bx:x1] > 0
            if int(m.sum()) < 12:
                continue
            vx = float(gx[by:y1, bx:x1][m].mean())
            vy = float(gy[by:y1, bx:x1][m].mean())
            if vx * vx + vy * vy < 1e-3:
                ang = 0.0
            else:
                # 针向沿边缘（垂直梯度）
                ang = float(np.degrees(np.arctan2(vy, vx)) + 90.0)
            ang = ((ang + 90.0) % 180.0) - 90.0
            # 限制过大竖向，避免乱针
            ang = float(np.clip(ang, -32.0, 32.0))
            raw[(by, bx)] = ang
    # 邻块平均
    smoothed: dict[tuple[int, int], float] = {}
    for (by, bx), ang in raw.items():
        acc = [ang]
        for dy, dx in ((-block, 0), (block, 0), (0, -block), (0, block)):
            nb = raw.get((by + dy, bx + dx))
            if nb is not None:
                acc.append(nb)
        smoothed[(by, bx)] = float(sum(acc) / len(acc))
    return smoothed


def _strip_fringe_white(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只清主体外缘近白，保留白衣/高领内部，避免缩略图衣身被挖空成草图。"""
    out = mask.copy()
    near_white = (bgr[:, :, 0] > 248) & (bgr[:, :, 1] > 248) & (bgr[:, :, 2] > 248)
    core = cv2.erode(out, np.ones((5, 5), np.uint8), iterations=2)
    out[near_white & (core == 0)] = 0
    return out


def block_oriented_polylines(
    mask: np.ndarray,
    gray: np.ndarray,
    *,
    spacing: float,
    stitch_len: float,
    block: int = 28,
    fallback_angle: float = 0.0,
) -> list[list[tuple[float, float]]]:
    """按块方向短填；轻重叠（约 2/3 步）兼顾接缝与缩略图整齐。"""
    angles = estimate_block_angles(gray, mask, block=block)
    h, w = mask.shape
    polys: list[list[tuple[float, float]]] = []
    if not angles:
        raw = _tatami_polylines(
            mask, angle_deg=fallback_angle, spacing=spacing, stitch_len=stitch_len
        )
        return filter_short_polylines(raw, min_span=max(2.8, stitch_len * 1.4))

    def _angle_at(y0: int, x0: int) -> float:
        key = (y0 - (y0 % block), x0 - (x0 % block))
        if key in angles:
            return angles[key]
        best, best_d = fallback_angle, 1e18
        for (by, bx), ang in angles.items():
            d = (by - y0) ** 2 + (bx - x0) ** 2
            if d < best_d:
                best, best_d = ang, d
        return best

    stride = max(14, (block * 2) // 3)
    for by in range(0, h, stride):
        for bx in range(0, w, stride):
            y1, x1 = min(h, by + block), min(w, bx + block)
            sub = mask[by:y1, bx:x1]
            if int(np.count_nonzero(sub)) < 12:
                continue
            ang = _angle_at(by, bx)
            local = _tatami_polylines(
                sub, angle_deg=ang, spacing=spacing, stitch_len=stitch_len
            )
            for poly in local:
                polys.append([(x + bx, y + by) for x, y in poly])
    return filter_short_polylines(polys, min_span=max(2.8, stitch_len * 1.4))


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


def build_photo_pattern(
    label_map: np.ndarray,
    centers_bgr: list[np.ndarray],
    mask: np.ndarray,
    *,
    spacing: float,
    stitch_len: float,
    target_height_mm: float,
    pattern: EmbPattern | None = None,
    first: bool = True,
    tag_prefix: str = "tone",
    gray: np.ndarray | None = None,
    density_gate: bool = True,
    oriented: bool = True,
    block: int = 20,
    poster: bool = False,
) -> tuple[EmbPattern, list[dict[str, Any]], bool]:
    h, _w = label_map.shape
    scale = (target_height_mm / max(h, 1)) * 10.0
    if pattern is None:
        pattern = EmbPattern()
    layers_stats: list[dict[str, Any]] = []
    order = _color_order_dark_first(centers_bgr)
    if gray is None:
        gray = np.zeros((h, _w), dtype=np.uint8)
        for i, c in enumerate(centers_bgr):
            lum = int(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2])
            gray[label_map == i] = lum
    if poster:
        density_gate = False
        oriented = False

    for idx in order:
        layer_mask = ((label_map == idx) & (mask > 0)).astype(np.uint8) * 255
        # poster 不再 harden：硬化会并块成马赛克墙
        layer_mask = clean_layer_mask(layer_mask, min_area=20, harden=False)
        if density_gate:
            layer_mask = apply_density_gate(
                layer_mask,
                gray,
                keep_dark_pct=0.92,
                highlight_only=True,
            )
            layer_mask = clean_layer_mask(layer_mask, min_area=18, harden=False)
        if int(np.count_nonzero(layer_mask)) < 16:
            continue
        hex_c = _bgr_to_hex(centers_bgr[idx])
        lum = float(
            0.114 * centers_bgr[idx][0]
            + 0.587 * centers_bgr[idx][1]
            + 0.299 * centers_bgr[idx][2]
        )
        if poster:
            sp = spacing * 0.75
            sl = stitch_len * 1.0
        else:
            sp = spacing * (0.82 if lum < 90 else (0.92 if lum < 160 else 1.0))
            sl = stitch_len * (0.92 if lum < 90 else 1.0)
        # 缩略图成团：同前缀脸层共用近水平角，避免多色交叉成「乱草」
        if tag_prefix.startswith("face") or tag_prefix == "tone":
            fallback = 8.0 if lum >= 90 else 0.0
        else:
            fallback = float((idx * 11) % 24) - 12.0
        if oriented:
            polys = block_oriented_polylines(
                layer_mask,
                gray,
                spacing=sp,
                stitch_len=sl,
                block=block,
                fallback_angle=fallback,
            )
            stitch_name = "photo_oriented_fill"
            angle = fallback
        else:
            polys = filter_short_polylines(
                _tatami_polylines(layer_mask, angle_deg=fallback, spacing=sp, stitch_len=sl),
                min_span=max(2.0, sl * 1.1),
            )
            stitch_name = "photo_poster_fill" if poster else "photo_density_fill"
            angle = fallback
        first, n_pts = _emit_polys(
            pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
        )
        if n_pts < 2:
            continue
        layers_stats.append(
            {
                "tag": f"{tag_prefix}_{idx}",
                "hex": hex_c,
                "bgr": [int(x) for x in centers_bgr[idx]],
                "stitch": stitch_name,
                "angle_deg": angle,
                "points": n_pts,
                "segments": len(polys),
                "enabled": True,
            }
        )
    return pattern, layers_stats, first


def _median_skin_hex(bgr: np.ndarray, face_mask: np.ndarray) -> str:
    fg = face_mask > 0
    if not np.any(fg):
        return "#d2b7a6"
    pix = bgr[fg]
    med = np.median(pix, axis=0)
    return _bgr_to_hex(med)


def emit_face_underlay(
    pattern: EmbPattern,
    bgr: np.ndarray,
    face_mask: np.ndarray,
    *,
    spacing: float,
    stitch_len: float,
    target_height_mm: float,
    first: bool,
) -> tuple[EmbPattern, list[dict[str, Any]], bool]:
    """肤色薄底层：极疏同色填，减少放大透白。"""
    layers: list[dict[str, Any]] = []
    base = clean_layer_mask(face_mask, min_area=40)
    if int(np.count_nonzero(base)) < 80:
        return pattern, layers, first
    h = face_mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    hex_c = _median_skin_hex(bgr, face_mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    polys = filter_short_polylines(
        _tatami_polylines(
            base,
            angle_deg=12.0,
            spacing=max(1.45, spacing * 1.55),
            stitch_len=max(1.7, stitch_len * 1.25),
        ),
        min_span=max(2.8, stitch_len * 1.4),
    )
    first, n_pts = _emit_polys(
        pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
    )
    if n_pts >= 2:
        layers.append(
            {
                "tag": "face_underlay",
                "hex": hex_c,
                "stitch": "photo_underlay",
                "points": n_pts,
                "segments": len(polys),
                "enabled": True,
                "role": "underlay",
            }
        )
    return pattern, layers, first


def emit_dark_features(
    pattern: EmbPattern,
    bgr: np.ndarray,
    face_mask: np.ndarray,
    *,
    spacing: float,
    stitch_len: float,
    target_height_mm: float,
    first: bool,
    hex_c: str = "#1a1210",
    dense_core: bool = True,
) -> tuple[EmbPattern, list[dict[str, Any]], bool, np.ndarray]:
    """深色五官层（建议叠在色调之后）；方向跟随 + 可选核心再加密。"""
    feat = clean_layer_mask(dark_feature_mask(bgr, face_mask), min_area=12)
    layers: list[dict[str, Any]] = []
    if int(np.count_nonzero(feat)) < 20:
        return pattern, layers, first, feat
    h = face_mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    # 缩略图优先：五官同向密填 + 轮廓，避免分块定向叠成「乱草」
    polys = filter_short_polylines(
        _tatami_polylines(
            feat,
            angle_deg=0.0,
            spacing=max(0.65, spacing * 0.62),
            stitch_len=max(1.2, stitch_len * 0.85),
        ),
        min_span=2.0,
    )
    run = filter_short_polylines(
        _running_polylines(feat, stitch_len=max(1.0, stitch_len * 0.65)),
        min_span=2.5,
    )
    polys = polys + run
    if dense_core:
        core = cv2.erode(feat, np.ones((2, 2), np.uint8), iterations=1)
        core = clean_layer_mask(core, min_area=8)
        if int(np.count_nonzero(core)) >= 16:
            polys += filter_short_polylines(
                _tatami_polylines(
                    core,
                    angle_deg=0.0,
                    spacing=max(0.5, spacing * 0.48),
                    stitch_len=max(1.1, stitch_len * 0.75),
                ),
                min_span=1.8,
            )
    first, n_pts = _emit_polys(
        pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
    )
    if n_pts >= 2:
        layers.append(
            {
                "tag": "face_dark",
                "hex": hex_c,
                "stitch": "photo_dark_oriented",
                "points": n_pts,
                "segments": len(polys),
                "enabled": True,
            }
        )
    return pattern, layers, first, feat


def _scale_landmarks(
    points: np.ndarray | None,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray | None:
    if points is None or len(points) == 0:
        return None
    sy = dst_shape[0] / max(src_shape[0], 1)
    sx = dst_shape[1] / max(src_shape[1], 1)
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def _scale_roi(
    roi: tuple[int, int, int, int], src_shape: tuple[int, int], dst_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    sy = dst_shape[0] / max(src_shape[0], 1)
    sx = dst_shape[1] / max(src_shape[1], 1)
    x2 = int(round(x * sx))
    y2 = int(round(y * sy))
    w2 = max(1, int(round(w * sx)))
    h2 = max(1, int(round(h * sy)))
    return x2, y2, w2, h2


def _face_mask(
    shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    subject_mask: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """优先关键点凸包脸罩；否则椭圆 + ROI∩主体下半并集。"""
    if landmarks is not None and len(landmarks) >= 50:
        from embroideryforge.face_detect import face_mask_from_landmarks

        return face_mask_from_landmarks(
            shape, landmarks, subject_mask=subject_mask, roi=roi
        )

    h, w = shape
    x, y, rw, rh = roi
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = x + rw // 2, y + int(rh * 0.55)
    axes = (max(1, int(rw * 0.48)), max(1, int(rh * 0.62)))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    if subject_mask is not None and subject_mask.shape[:2] == (h, w):
        # ROI 矩形内、椭圆中心以下的主体像素并入脸罩（下巴/下颚）
        rect = np.zeros((h, w), dtype=np.uint8)
        y1 = min(h, y + rh)
        x1 = min(w, x + rw)
        rect[max(0, y) : y1, max(0, x) : x1] = 255
        below = np.zeros((h, w), dtype=np.uint8)
        below[cy:, :] = 255
        chin = cv2.bitwise_and(subject_mask, rect)
        chin = cv2.bitwise_and(chin, below)
        mask = cv2.bitwise_or(mask, chin)
        mask = cv2.bitwise_and(mask, subject_mask)
    return mask


def _fit_panel(img: np.ndarray, th: int = 320) -> np.ndarray:
    h, w = img.shape[:2]
    s = th / max(h, 1)
    return cv2.resize(img, (max(1, int(w * s)), th), interpolation=cv2.INTER_AREA)


def run_photo_stitch(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    n_colors: int = 9,
    resolution: str = "high",
    target_height_mm: float = 70.0,
    bg_mode: str = "auto",
    face_roi: bool = True,
    full_frame: bool = False,
    mode: str = "face",
    poster: bool = False,
) -> dict[str, Any]:
    """照片绣入口。

    mode:
      - face: 仅脸/半身 ROI（P1）
      - full: 整图 Photo-Stitch
      - hybrid: 脸 Photo-Stitch + 衣帽 tatami（P2）
    poster:
      - True: 少色（≤5）近无缝实填 + 假厚预览，缩略图成团
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = mode if mode in {"face", "full", "hybrid"} else "face"
    if full_frame:
        mode = "full"
    if poster:
        # poster=密填+假厚；色数保持偏高，避免量化马赛克
        n_colors = int(np.clip(max(n_colors, 9), 9, 12))

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(image_path)

    subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    subject_path = out_dir / "subject.png"
    cv2.imwrite(str(subject_path), subject_bgr)
    cv2.imwrite(str(out_dir / "subject_mask.png"), subject_mask)

    cfg = dict(RESOLUTION.get(resolution, RESOLUTION["high"]))
    if poster:
        # 更高采样，减轻大色块锯齿
        cfg["max_side"] = max(int(cfg["max_side"]), 420)
    spacing = float(cfg["spacing"])
    stitch_len = float(cfg["stitch_len"])
    if poster:
        spacing *= 0.70
        stitch_len *= 1.05
    geo = detect_face_geometry(subject_bgr, subject_mask=subject_mask)
    roi_src = geo["roi"]
    face_landmarks_src = geo.get("landmarks")
    face_source = geo.get("source", "unknown")
    boxed = subject_bgr.copy()
    x0, y0, w0, h0 = roi_src
    cv2.rectangle(boxed, (x0, y0), (x0 + w0, y0 + h0), (0, 180, 255), 2)
    if face_landmarks_src is not None:
        for px, py in face_landmarks_src[::8]:
            cv2.circle(boxed, (int(px), int(py)), 1, (80, 220, 80), -1)
    cv2.imwrite(str(out_dir / "roi_box.png"), boxed)

    layers: list[dict[str, Any]] = []
    palette: list[str] = []
    resize_scale = 1.0
    engine = "photo_stitch_v3d_poster" if poster else "photo_stitch_v3c"
    layout_preview = None

    if mode == "hybrid":
        engine = "photo_stitch_hybrid_v3d_poster" if poster else "photo_stitch_hybrid_v3c"
        work_r, resize_scale = resize_for_resolution(
            subject_bgr, resolution, max_side=int(cfg["max_side"])
        )
        work_mask_r = cv2.resize(
            subject_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        work_mask_r = _strip_fringe_white(work_r, work_mask_r)
        roi = _scale_roi(roi_src, subject_bgr.shape[:2], work_r.shape[:2])
        lm_r = _scale_landmarks(face_landmarks_src, subject_bgr.shape[:2], work_r.shape[:2])
        face_m = _face_mask(work_r.shape[:2], roi, subject_mask=work_mask_r, landmarks=lm_r)
        face_m = cv2.bitwise_and(face_m, work_mask_r)
        face_dil = cv2.dilate(face_m, np.ones((5, 5), np.uint8), iterations=1)

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
        pattern = EmbPattern()
        first = True
        # 衣帽先填（避开脸）
        for tag in BODY_TAGS:
            if tag not in label_tags:
                continue
            tid = label_tags.index(tag)
            body = ((lab_r == tid) & (work_mask_r > 0) & (face_dil == 0)).astype(np.uint8) * 255
            body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if int(np.count_nonzero(body)) < 80:
                continue
            hex_c = hex_by_tag.get(tag, "#808080")
            angle = {"clothes": 0.0, "hat": 32.0, "hair": 58.0}.get(tag, 0.0)
            body = clean_layer_mask(body, min_area=50, harden=False)
            body_sp = spacing * (0.95 if poster else 1.08)
            polys = _tatami_polylines(
                body, angle_deg=angle, spacing=body_sp, stitch_len=stitch_len * 1.2
            )
            first, n_pts = _emit_polys(
                pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
            )
            if n_pts >= 2:
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
                palette.append(hex_c)

        # 顺序：薄底层（poster 实填可跳过）→ 脸部色调 → 深色五官 → outline
        under_layers: list[dict[str, Any]] = []
        if not poster:
            pattern, under_layers, first = emit_face_underlay(
                pattern,
                work_r,
                face_m,
                spacing=spacing,
                stitch_len=stitch_len,
                target_height_mm=target_height_mm,
                first=first,
            )
            layers.extend(under_layers)
            if under_layers:
                palette.append(str(under_layers[0]["hex"]))

        gray_face = cv2.cvtColor(work_r, cv2.COLOR_BGR2GRAY)
        tone, label_map, centers = quantize_tones(
            work_r, face_m, n_colors=n_colors, pre_smooth=poster
        )
        dens = density_preview(work_r, face_m)
        # 脸部大面：整层同向；poster=密填+平滑色调预览
        pattern, face_layers, first = build_photo_pattern(
            label_map,
            centers,
            face_m,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            pattern=pattern,
            first=first,
            tag_prefix="face",
            gray=gray_face,
            density_gate=False,
            oriented=False,
            poster=poster,
        )
        layers.extend(face_layers)
        palette.extend(_bgr_to_hex(c) for c in centers)

        pattern, dark_layers, first, feat = emit_dark_features(
            pattern,
            work_r,
            face_m,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            first=first,
            dense_core=not poster,
        )
        layers.extend(dark_layers)
        if dark_layers:
            palette.append(str(dark_layers[0]["hex"]))
        cv2.imwrite(str(out_dir / "dark_features.png"), feat)

        if "outline" in label_tags:
            oid = label_tags.index("outline")
            om = ((lab_r == oid) & (work_mask_r > 0)).astype(np.uint8) * 255
            polys = _running_polylines(om, stitch_len=min(stitch_len, 1.8))
            hex_c = hex_by_tag.get("outline", "#2b1b17")
            first, n_pts = _emit_polys(
                pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
            )
            if n_pts >= 2:
                layers.append(
                    {
                        "tag": "outline",
                        "hex": hex_c,
                        "stitch": "running",
                        "points": n_pts,
                        "segments": len(polys),
                        "enabled": True,
                    }
                )
                palette.append(hex_c)

        pattern.end()
        layout_preview = work_r.copy()
        layout_preview[face_m > 0] = (
            layout_preview[face_m > 0] * 0.55 + np.array([0, 180, 255]) * 0.45
        ).astype(np.uint8)
        body_show = (work_mask_r > 0) & (face_dil == 0)
        layout_preview[body_show] = (
            layout_preview[body_show] * 0.7 + np.array([40, 200, 80]) * 0.3
        ).astype(np.uint8)
        cv2.imwrite(str(out_dir / "layout_preview.png"), layout_preview)
        cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
        cv2.imwrite(str(out_dir / "density_preview.png"), dens)
        work_r_for_compare = work_r
        tone_for_compare = tone
        roi_meta = {
            "mode": "hybrid",
            "xywh": [int(x0), int(y0), int(w0), int(h0)],
            "detector": face_source,
        }

    else:
        # face / full
        if mode == "full" or not face_roi:
            work = subject_bgr
            work_mask = subject_mask
            roi_meta = {"mode": "full"}
        else:
            x, y, w, h = roi_src
            work = subject_bgr[y : y + h, x : x + w].copy()
            work_mask = subject_mask[y : y + h, x : x + w].copy()
            roi_meta = {
                "mode": "face_or_bust",
                "xywh": [int(x), int(y), int(w), int(h)],
                "detector": face_source,
            }

        work_r, resize_scale = resize_for_resolution(
            work, resolution, max_side=int(cfg["max_side"])
        )
        work_mask_r = cv2.resize(
            work_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        work_mask_r = _strip_fringe_white(work_r, work_mask_r)

        gray_w = cv2.cvtColor(work_r, cv2.COLOR_BGR2GRAY)
        pattern = EmbPattern()
        first = True
        under_layers = []
        if not poster:
            pattern, under_layers, first = emit_face_underlay(
                pattern,
                work_r,
                work_mask_r,
                spacing=spacing,
                stitch_len=stitch_len,
                target_height_mm=target_height_mm,
                first=first,
            )
            layers.extend(under_layers)
        tone, label_map, centers = quantize_tones(
            work_r, work_mask_r, n_colors=n_colors, pre_smooth=poster
        )
        dens = density_preview(work_r, work_mask_r)
        cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
        cv2.imwrite(str(out_dir / "density_preview.png"), dens)
        pattern, tone_layers, first = build_photo_pattern(
            label_map,
            centers,
            work_mask_r,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            pattern=pattern,
            first=first,
            gray=gray_w,
            density_gate=False,
            oriented=False,
            poster=poster,
        )
        layers.extend(tone_layers)
        pattern, dark_layers, first, feat = emit_dark_features(
            pattern,
            work_r,
            work_mask_r,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            first=first,
            dense_core=not poster,
        )
        layers.extend(dark_layers)
        cv2.imwrite(str(out_dir / "dark_features.png"), feat)
        pattern.end()
        palette = (
            [str(L["hex"]) for L in under_layers]
            + [_bgr_to_hex(c) for c in centers]
            + [str(L["hex"]) for L in dark_layers]
        )
        work_r_for_compare = work_r
        tone_for_compare = tone

    report = {
        "stitches": pattern.count_stitches(),
        "color_changes": pattern.count_color_changes(),
        "layers": layers,
        "target_height_mm": target_height_mm,
        "engine": engine,
    }

    dst = out_dir / "auto.dst"
    raw_png = out_dir / "auto_pyembroidery.png"
    write_dst(pattern, str(dst))
    write_png(pattern, str(raw_png))
    thin = out_dir / "auto_stitch_thin.png"
    thick_path = out_dir / "auto_stitch_thick.png"
    prev_meta = render_clean_preview(pattern, thick_path, layers=layers, thick=True)
    render_clean_preview(pattern, thin, layers=layers, thick=False)

    # poster：主预览用色调实色（双边平滑），避免假厚针迹把脸糊成马赛克墙
    clean = out_dir / "auto_clean_preview.png"
    if poster:
        # 缩略主图：平滑主体照片 + 脸区轻混色调（避免纯量化脸变马赛克墙）
        base = cv2.bilateralFilter(work_r_for_compare, d=7, sigmaColor=50, sigmaSpace=50)
        tone_soft = cv2.bilateralFilter(tone_for_compare, d=7, sigmaColor=50, sigmaSpace=50)
        soft = base.copy()
        # face 区：照片与色调 6:4，保留五官同时暗示分色
        if mode == "hybrid":
            fm = face_m
            if fm.shape[:2] != soft.shape[:2]:
                fm = cv2.resize(fm, (soft.shape[1], soft.shape[0]), interpolation=cv2.INTER_NEAREST)
            mix = cv2.addWeighted(base, 0.62, tone_soft, 0.38, 0)
            soft = np.where(fm[..., None] > 0, mix, soft)
        else:
            soft = cv2.addWeighted(base, 0.55, tone_soft, 0.45, 0)
        cv2.imwrite(str(clean), soft)
        thumb_prev = soft
    else:
        import shutil

        shutil.copyfile(thick_path, clean)
        thumb_prev = cv2.imread(str(clean), cv2.IMREAD_COLOR)
        if thumb_prev is None:
            thumb_prev = np.full_like(tone_for_compare, 240)

    panels = [
        _fit_panel(work_r_for_compare),
        _fit_panel(tone_for_compare),
        _fit_panel(thumb_prev),
    ]
    if layout_preview is not None and not poster:
        panels = [
            _fit_panel(work_r_for_compare),
            _fit_panel(layout_preview),
            _fit_panel(thumb_prev),
        ]
    cv2.imwrite(str(out_dir / "compare_preview.png"), np.hstack(panels))

    meta = {
        "engine": engine,
        "mode": mode,
        "poster": bool(poster),
        "n_colors": n_colors,
        "resolution": resolution,
        "spacing_px": spacing,
        "stitch_len_px": stitch_len,
        "resize_scale": resize_scale,
        "roi": roi_meta,
        "face_detect": {"source": face_source, "mediapipe": face_source == "mediapipe"},
        "bg": bg_meta,
        "palette": palette,
        "layers": layers,
        "stitches": report["stitches"],
        "color_changes": report["color_changes"],
        "target_height_mm": target_height_mm,
        "source_image": str(image_path),
        "preview": {
            "thick": True,
            "stroke_width": prev_meta.get("stroke_width"),
            "thin_file": thin.name,
            "thick_file": thick_path.name,
            "poster_tone_thumb": bool(poster),
        },
        "note": (
            "poster：主预览=平滑色调实色（防马赛克）；针迹见 auto_stitch_thick/thin.png"
            if poster
            else "主预览假厚；细线见 auto_stitch_thin.png"
        ),
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps({**report, "photo_stitch": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "dst": dst,
        "out_dir": out_dir,
        "report": report,
        "meta": meta,
    }
