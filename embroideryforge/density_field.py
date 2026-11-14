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


def unsharp_masked(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    radius: float = 1.2,
    amount: float = 0.45,
) -> np.ndarray:
    """仅在 mask 内做轻 unsharp，抬五官对比。"""
    if not np.any(mask):
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), max(0.5, float(radius)))
    sharp = cv2.addWeighted(bgr, 1.0 + float(amount), blur, -float(amount), 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    m = mask > 0
    out = bgr.copy()
    out[m] = sharp[m]
    return out


def skin_mask_hsv(bgr: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
    """粗肤色罩（HSV），可选限制在 region 内。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin = ((H <= 25) | (H >= 165)) & (S >= 18) & (S <= 170) & (V >= 45) & (V <= 250)
    out = (skin.astype(np.uint8) * 255)
    if region is not None and region.shape[:2] == out.shape[:2]:
        out = cv2.bitwise_and(out, region)
    return out


def tighten_bust_face_mask(
    person_mask: np.ndarray,
    face_m: np.ndarray,
    face_source: str,
    *,
    upper_frac: float = 0.30,
    hat_skip_frac: float = 0.22,
    bgr: np.ndarray | None = None,
) -> np.ndarray:
    """bust 回退：去掉帽冠、收上半身窗；有图时优先肤色∩上窗。

    mediapipe/haar 也可用肤色收紧（去掉帽檐/发顶误入）。
    """
    ys, xs = np.where(person_mask > 0)
    if len(ys) < 20:
        return face_m
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ph = y1 - y0 + 1
    skip = max(0, int(ph * float(hat_skip_frac)))
    band_h = max(1, int(ph * float(upper_frac)))
    upper = np.zeros_like(person_mask)
    upper[y0 + skip : y0 + skip + band_h, x0 : x1 + 1] = 255
    upper = cv2.bitwise_and(upper, person_mask)

    if str(face_source) in {"mediapipe", "haar"}:
        tight = cv2.bitwise_and(face_m, person_mask) if np.any(face_m) else face_m
        if bgr is not None and bgr.shape[:2] == person_mask.shape[:2]:
            skin_u8 = skin_mask_hsv(bgr, person_mask)
            skin_u8 = cv2.morphologyEx(skin_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # 只在脸罩邻域用肤色收紧，避免整身肤色把罩撑大
            near = cv2.dilate(tight, np.ones((9, 9), np.uint8), iterations=1)
            skin_face = cv2.bitwise_and(skin_u8, near)
            if int(np.count_nonzero(skin_face)) >= 100:
                merged = cv2.bitwise_or(cv2.bitwise_and(tight, skin_u8), skin_face)
                if int(np.count_nonzero(merged)) >= 80:
                    tight = merged
        return tight

    tight = cv2.bitwise_and(face_m, upper) if np.any(face_m) else upper
    if bgr is not None and bgr.shape[:2] == person_mask.shape[:2]:
        skin_u8 = skin_mask_hsv(bgr, upper)
        skin_u8 = cv2.morphologyEx(skin_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if int(np.count_nonzero(skin_u8)) >= 120:
            tight = skin_u8
    if int(np.count_nonzero(tight)) < 80:
        tight = upper
    return tight


def face_structure_dens_boost(
    dens: np.ndarray,
    gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    percentile: float = 52.0,
    gain: float = 1.38,
) -> np.ndarray:
    """脸内 Laplacian 高响应区抬 dens，利五官短针。"""
    if face_mask is None or not np.any(face_mask):
        return dens
    fm = face_mask > 0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    amp = np.abs(lap)
    vals = amp[fm]
    if vals.size < 40:
        return dens
    thr = float(np.percentile(vals, percentile))
    hot = fm & (amp >= thr)
    out = dens.copy()
    out[hot] = np.minimum(1.0, out[hot] * float(gain) + 0.10)
    return out


def remap_labels_to_centers(
    bgr: np.ndarray,
    mask: np.ndarray,
    centers: list[np.ndarray],
    *,
    base_labels: np.ndarray | None = None,
) -> np.ndarray:
    """在 mask 内按 Lab 最近邻把像素重绑到已有 centers（同色线、重分块）。

    用于 dens_face：CLAHE 后局部明暗能改 label，而不另开第二套调色板。
    """
    h, w = mask.shape[:2]
    if base_labels is not None and base_labels.shape[:2] == (h, w):
        out = base_labels.copy()
    else:
        out = np.zeros((h, w), dtype=np.int32)
    if not centers or not np.any(mask):
        return out
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cents = []
    for c in centers:
        cb = np.asarray(c, dtype=np.uint8).reshape(1, 1, 3)
        cents.append(cv2.cvtColor(cb, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32))
    C = np.stack(cents, axis=0)
    ys, xs = np.where(mask > 0)
    pix = lab[ys, xs]
    chunk = 20000
    best = np.zeros(len(ys), dtype=np.int32)
    for i0 in range(0, len(ys), chunk):
        i1 = min(len(ys), i0 + chunk)
        d = pix[i0:i1, None, :] - C[None, :, :]
        best[i0:i1] = np.argmin(np.sum(d * d, axis=2), axis=1)
    out[ys, xs] = best
    return out


def clahe_face_bgr(
    bgr: np.ndarray,
    face_mask: np.ndarray,
    *,
    clip_limit: float = 2.6,
    blend: float = 0.72,
    tile: int = 4,
) -> np.ndarray:
    """脸区强 CLAHE（仅 L），blend=新图权重。"""
    if not np.any(face_mask):
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    L = lab[..., 0]
    L2 = L.copy()
    ys, xs = np.where(face_mask > 0)
    y0, y1 = max(0, int(ys.min()) - 2), min(L.shape[0], int(ys.max()) + 3)
    x0, x1 = max(0, int(xs.min()) - 2), min(L.shape[1], int(xs.max()) + 3)
    patch = clahe.apply(L[y0:y1, x0:x1])
    b = float(np.clip(blend, 0.0, 1.0))
    mixed = L[y0:y1, x0:x1].astype(np.float32) * (1.0 - b) + patch.astype(np.float32) * b
    L2[y0:y1, x0:x1] = np.clip(mixed, 0, 255).astype(np.uint8)
    lab[..., 0] = np.where(face_mask > 0, L2, L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def dampen_face_highlights(
    bgr: np.ndarray,
    face_mask: np.ndarray,
    *,
    pct: float = 88.0,
    pull: float = 0.82,
) -> np.ndarray:
    """压脸区高光 L，减轻 CLAHE 后偏白/死亮。"""
    if not np.any(face_mask):
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    fm = face_mask > 0
    vals = L[fm]
    if vals.size < 40:
        return bgr
    thr = float(np.percentile(vals, pct))
    hot = fm & (L >= thr)
    L[hot] = L[hot] * float(pull) + thr * (1.0 - float(pull))
    lab[..., 0] = np.clip(L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blend_face_labels_structure(
    base_labels: np.ndarray,
    remapped: np.ndarray,
    gray: np.ndarray,
    face_mask: np.ndarray,
    *,
    percentile: float = 58.0,
) -> np.ndarray:
    """肤色区保留 FG label（防漂白）；仅结构高响应区采用重映射。"""
    out = base_labels.copy()
    fm = face_mask > 0
    if not np.any(fm):
        return out
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    vals = lap[fm]
    if vals.size < 40:
        return out
    thr = float(np.percentile(vals, percentile))
    hot = fm & (lap >= thr)
    out[hot] = remapped[hot]
    return out


def chroma_weight_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lab 色度权：鲜艳区（红联/金瓦）加权，避免大面灰褐占满 BG 色槽。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    ch = np.sqrt(a * a + b * b)
    w = np.where(mask > 0, 1.0 + np.clip(ch / 28.0, 0.0, 3.5), 0.0).astype(np.float32)
    return w


def boost_bg_chroma_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    sat_gain: float = 1.18,
    center_pull: float = 1.12,
) -> np.ndarray:
    """量化前轻抬饱和；量化后对 centers 另做 restore。"""
    if not np.any(mask):
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask > 0
    hsv[..., 1][m] = np.clip(hsv[..., 1][m] * float(sat_gain), 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def restore_center_chroma(
    centers: list[np.ndarray],
    *,
    pull: float = 1.14,
    max_shift: float = 18.0,
) -> list[np.ndarray]:
    """把 Lab a/b 略拉离中性灰，减轻 BG 发褐发灰。"""
    out: list[np.ndarray] = []
    for c in centers:
        cb = np.asarray(c, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(cb, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
        a = lab[1] - 128.0
        b = lab[2] - 128.0
        a2 = float(np.clip(a * float(pull), -max_shift * 2, max_shift * 2))
        b2 = float(np.clip(b * float(pull), -max_shift * 2, max_shift * 2))
        # 限制单次偏移幅度，避免荧光
        a2 = a + float(np.clip(a2 - a, -max_shift, max_shift))
        b2 = b + float(np.clip(b2 - b, -max_shift, max_shift))
        lab[1] = 128.0 + a2
        lab[2] = 128.0 + b2
        lab_u8 = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR).reshape(3)
        out.append(bgr.astype(np.uint8))
    return out


def fg_color_weight_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    face_mask: np.ndarray | None = None,
    hand_mask: np.ndarray | None = None,
) -> np.ndarray:
    """FG 量化权：青绿/冷色衣物抬权；脸手权减弱，避免褐肤占满色槽。"""
    w = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
    if not np.any(mask):
        return w
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    ch = np.sqrt(a * a + b * b)
    m = mask > 0
    # 青绿/青蓝（OpenCV H: 35–100）+ Lab a 偏绿
    cool = m & (
        ((H >= 35.0) & (H <= 100.0) & (S >= 28.0) & (V >= 35.0))
        | ((a < -2.5) & (S >= 22.0) & (V >= 35.0))
    )
    cool_boost = 1.0 + 2.6 * np.clip(S / 180.0, 0.2, 1.0) + 0.8 * np.clip(ch / 40.0, 0.0, 1.0)
    w = np.where(cool, w + cool_boost, w)
    # 一般鲜艳非肤（避开 H 低暖色）略抬
    vivid = m & (ch >= 18.0) & (S >= 40.0) & ~((H <= 25.0) | (H >= 160.0))
    w = np.where(vivid & ~cool, w + 1.2, w)
    if face_mask is not None and face_mask.shape[:2] == mask.shape[:2]:
        w = np.where((face_mask > 0) & m, w + 0.65, w)
    if hand_mask is not None and hand_mask.shape[:2] == mask.shape[:2]:
        w = np.where((hand_mask > 0) & m, w + 0.65, w)
    return w


def boost_cool_garment_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    sat_gain: float = 1.22,
) -> np.ndarray:
    """仅抬冷色衣物饱和，助青绿进色槽；不动肤色暖区。"""
    if not np.any(mask):
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = mask > 0
    cool = m & (H >= 35.0) & (H <= 100.0) & (S >= 22.0) & (V >= 30.0)
    if not np.any(cool):
        return bgr
    hsv[..., 1][cool] = np.clip(hsv[..., 1][cool] * float(sat_gain), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def balance_fg_centers(
    centers: list[np.ndarray],
    *,
    cool_pull: float = 1.22,
    warm_a_pull: float = 0.72,
    max_shift: float = 14.0,
) -> list[np.ndarray]:
    """FG 中心：冷色回拉色度；暖肤 a 略压，减轻侧脸偏红。"""
    out: list[np.ndarray] = []
    for c in centers:
        cb = np.asarray(c, dtype=np.uint8).reshape(1, 1, 3)
        bgr0 = cb.reshape(3)
        lab = cv2.cvtColor(cb, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
        hsv = cv2.cvtColor(cb, cv2.COLOR_BGR2HSV).astype(np.float32).reshape(3)
        L, a, b = float(lab[0]), float(lab[1] - 128.0), float(lab[2] - 128.0)
        H, S = float(hsv[0]), float(hsv[1])
        # 青绿/冷：加强 a 负向与饱和感
        cool = (H >= 35.0 and H <= 100.0 and S >= 20.0) or (a < -2.0) or (
            int(bgr0[1]) >= int(bgr0[2]) + 6 and int(bgr0[1]) >= int(bgr0[0]) + 4
        )
        if cool:
            a2 = a * float(cool_pull)
            b2 = b * (1.0 + 0.08 * (float(cool_pull) - 1.0))
        elif a > 5.5 and 35.0 <= L <= 210.0 and (H <= 28.0 or H >= 155.0):
            # 暖肤：压 a，减侧脸发红
            a2 = a * float(warm_a_pull)
            b2 = b * 0.92
        else:
            a2, b2 = a, b
        a2 = a + float(np.clip(a2 - a, -max_shift, max_shift))
        b2 = b + float(np.clip(b2 - b, -max_shift, max_shift))
        lab[1] = 128.0 + a2
        lab[2] = 128.0 + b2
        lab_u8 = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        out.append(cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR).reshape(3).astype(np.uint8))
    return out


def cool_garment_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """青绿/青蓝衣物罩（OpenCV HSV）；与 hue-band cool 桶对齐。

    G>R + 开运算：去掉脸噪点误入 cool（曾导致强制留针斑块）。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    B, G, R = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    m = mask > 0
    cool = m & (G.astype(np.int16) > R.astype(np.int16)) & (
        ((H >= 35) & (H <= 100) & (S >= 18) & (V >= 28))
        | ((a < -2.5) & (S >= 18) & (V >= 28))
    )
    u8 = (cool.astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return u8


def reserve_cool_garment_palette(
    bgr: np.ndarray,
    mask: np.ndarray,
    label_map: np.ndarray,
    centers: list[np.ndarray],
    *,
    n_slots: int = 3,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """给冷色衣物预留独立色槽，避免被褐肤 k-means 吞成灰。

    仅改写 cool 像素的 label；原有暖色中心不动。
    """
    cool = cool_garment_mask(bgr, mask)
    n_cool = int(np.count_nonzero(cool))
    if n_cool < 400 or not centers:
        return label_map, centers
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[cool > 0]
    n_k = int(np.clip(min(n_slots, max(2, n_cool // 800)), 2, 4))
    n_k = min(n_k, max(2, len(samples)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _c, _l, cool_c = cv2.kmeans(
        samples, n_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    cool_u8 = np.clip(cool_c, 0, 255).astype(np.uint8)
    cool_bgr = cv2.cvtColor(cool_u8.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
    # 略抬冷色饱和
    cool_bgr = balance_fg_centers([cool_bgr[i] for i in range(len(cool_bgr))], cool_pull=1.28)

    out_centers = list(centers) + list(cool_bgr)
    base = len(centers)
    out_labels = label_map.copy()
    ys, xs = np.where(cool > 0)
    pix = lab[ys, xs]
    C = cool_u8.astype(np.float32)
    chunk = 20000
    best = np.zeros(len(ys), dtype=np.int32)
    for i0 in range(0, len(ys), chunk):
        i1 = min(len(ys), i0 + chunk)
        d = pix[i0:i1, None, :] - C[None, :, :]
        best[i0:i1] = np.argmin(np.sum(d * d, axis=2), axis=1)
    out_labels[ys, xs] = base + best
    return out_labels, out_centers


def render_density_preview(dens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = np.clip(dens * 255.0, 0, 255).astype(np.uint8)
    vis = np.where(mask > 0, vis, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def dither_keep_mask(
    dens: np.ndarray,
    mask: np.ndarray,
    *,
    floor: float = 0.08,
    gain: float = 1.0,
) -> np.ndarray:
    """Bayer 留针。gain>1 降低阈值（更密）；默认 step 网格已饱和，预算主要靠 gain。"""
    ps = _ps()
    h, w = dens.shape
    g = float(np.clip(gain, 0.35, 2.6))
    yy, xx = np.indices((h, w))
    thr = ps._BAYER8[yy % 8, xx % 8]
    keep = ((dens >= thr * (0.92 / g)) | (dens >= (0.78 / g))) & (dens >= floor) & (mask > 0)
    bright = (dens >= 0.05) & (dens < 0.28) & (mask > 0)
    soft = bright & (dens >= thr * (0.38 / g))
    mid_bright = (dens >= 0.18) & (dens < 0.45) & (mask > 0)
    bridge = mid_bright & (dens >= thr * (0.55 / g))
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
    """按 step 采 keep。

    注意：旧实现 `int(round(step))` 使 step∈(0.5,1.5) 全变成 1，
    `--stitches-per-mm2` 加大时人眼无感。step<1 时用多相网格加密。
    """
    h, w = keep.shape
    step_f = max(0.35, float(step))
    grid = np.zeros_like(keep)
    if step_f >= 1.0:
        step_i = max(1, int(round(step_f)))
        grid[0:h:step_i, 0:w:step_i] = 255
    else:
        # step=0.5 → 主相+半相共约 4 倍采样；step=0.7 → 约 2 倍
        grid[0:h:1, 0:w:1] = 255
        # 再按密度稀疏化：亮区抽稀，避免 step≪1 时针爆
        if step_f > 0.55:
            # 保留全部暗点；亮区隔点
            bright = (dens < 0.45) & (mask > 0)
            thin = np.zeros_like(keep)
            thin[0:h:2, 0:w:2] = 255
            grid = np.where(bright, np.minimum(grid, thin), grid)
    out = cv2.bitwise_and(keep, grid)
    dark = (dens >= 0.55) & (mask > 0)
    if step_f >= 1.0:
        step_i = max(1, int(round(step_f)))
        fine_step = max(1, step_i - 1)
        fine = np.zeros_like(keep)
        fine[0:h:fine_step, 0:w:fine_step] = 255
        out = np.where(
            dark,
            np.maximum(out, cv2.bitwise_and(fine, (mask > 0).astype(np.uint8) * 255)),
            out,
        )
    else:
        # 暗部全采（已在全像素网格上）
        out = np.where(dark & (mask > 0), np.maximum(out, 255), out)
    # 五官核：半步相位网格 + 放宽 dens 门限
    if core_mask is not None and np.any(core_mask):
        cstep = max(1, int(round(max(step_f, 0.5) * core_step_scale)))
        cgrid = np.zeros_like(keep)
        cgrid[0:h:cstep, 0:w:cstep] = 255
        off = max(1, cstep // 2) if cstep >= 2 else 1
        if cstep >= 2:
            cgrid[off:h:cstep, off:w:cstep] = 255
        else:
            cgrid[1:h:2, 1:w:2] = 255
            if step_f < 0.75:
                cgrid[0:h:2, 1:w:2] = 255
                cgrid[1:h:2, 0:w:2] = 255
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
    """眼/眉/嘴等暗核：分位暗部 + 局部对比 + 脸内 Canny，限制在脸窗中上部。"""
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
    local = fg & ((blur.astype(np.int16) - gray.astype(np.int16)) >= 10)
    # 脸内边缘：加强五官轮廓，非衣褶结构层
    edges = cv2.Canny(gray, 36, 100)
    edge = fg & (edges > 0)
    u8 = ((dark | local | edge).astype(np.uint8) * 255)
    # 只用脸窗上 78% 高度，少把下巴阴影当核
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    cut = y0 + int((y1 - y0) * 0.78)
    band = np.zeros_like(u8)
    band[: cut + 1, :] = 255
    u8 = cv2.bitwise_and(u8, band)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    keep = np.zeros_like(u8)
    face_area = int(fg.sum())
    max_comp = max(420, int(face_area * 0.08))
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 4 <= a <= max_comp:
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
    subject_weight: np.ndarray | None = None,
    bg_dens_scale: float = 0.42,
    bg_step_scale: float = 1.75,
    bg_floor: float | None = None,
    tag_prefix: str = "dens",
    keep_floor: float | None = None,
    pre_smooth: bool | None = None,
    edge_boost: bool = False,
    fixed_label_map: np.ndarray | None = None,
    fixed_centers: list[np.ndarray] | None = None,
    omit_mask: np.ndarray | None = None,
    structure_gain: float | None = None,
    soften_highlights: bool = True,
    animal_palette: bool = False,
    portrait_p0: bool = False,
    reserve_bgr: list[np.ndarray] | None = None,
    density_gamma: float | None = None,
    keep_floor_scale: float = 1.0,
    dither_gain: float = 1.0,
    photo_blend: bool = False,
) -> tuple[EmbPattern, list[dict[str, Any]], bool, dict[str, np.ndarray]]:
    ps = _ps()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scene_mode = subject_weight is not None and subject_weight.shape[:2] == mask.shape
    has_face = face_only_mask is not None and np.any(face_only_mask)
    do_smooth = (not detail_boost) if pre_smooth is None else bool(pre_smooth)
    reuse_palette = (
        fixed_label_map is not None
        and fixed_centers is not None
        and fixed_label_map.shape[:2] == mask.shape[:2]
        and len(fixed_centers) > 0
    )
    is_bg_track = str(tag_prefix).startswith("dens_bg")
    is_fg_track = str(tag_prefix).startswith("dens_fg")
    is_face_track = str(tag_prefix).startswith("dens_face")
    animal_mode = bool(animal_palette) and (is_bg_track or is_fg_track)
    p0 = bool(portrait_p0) and not animal_mode
    reserve_use = list(reserve_bgr) if (p0 and reserve_bgr and (is_fg_track or is_face_track)) else None

    # 半身/场景/分轨：BG/FG 放开到 16 保色准（青绿衣等）
    if scene_mode:
        n_q = int(np.clip(int(n_colors) + (2 if detail_boost else 1), 10, 16))
    elif is_bg_track or is_fg_track:
        n_q = int(np.clip(int(n_colors), 10, 16))
    elif detail_boost and has_face:
        n_q = int(np.clip(int(n_colors) + 2, 10, 14))
    else:
        n_q = int(np.clip(int(n_colors) + (2 if detail_boost else 1), 8, 12))

    if reuse_palette:
        label_map = fixed_label_map.copy()
        centers = [np.asarray(c, dtype=np.uint8).reshape(3) for c in fixed_centers]
        # 不在此硬贴线表：全图 snap 会造成蒙红滤镜（人评 smoke < v66）
        tone = np.full_like(bgr, 255)
        for i, c in enumerate(centers):
            tone[label_map == i] = c
    elif is_bg_track or is_fg_track:
        # V6.6：Wilcom 式压平色板；人像 FG 暖肤压 a；动物域关压肤并保毛色/绿叶
        flatten_kw: dict[str, Any] = {
            "n_colors": n_q,
            "pre_smooth": False,
            "dampen_warm_skin": bool(is_fg_track) and not animal_mode,
        }
        if animal_mode and is_fg_track:
            flatten_kw["protect_chroma"] = 14.0
            flatten_kw["ab_weight"] = 2.2
        elif animal_mode and is_bg_track:
            flatten_kw["protect_chroma"] = 12.0
            flatten_kw["boost_green"] = 0.85
        if reserve_use:
            flatten_kw["reserve_bgr"] = reserve_use
        tone, label_map, centers = ps.quantize_tones_flatten(bgr, mask, **flatten_kw)
        # P0 仅 Reserve 保护 + 灰垫底序；线表只作层 meta 标注，不改写色心
    else:
        weight_q = None
        weight_power = 1.0
        if scene_mode:
            wq = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
            wq = np.where(subject_weight > 0, wq + 1.0, wq)
            if face_only_mask is not None and face_only_mask.shape[:2] == mask.shape:
                wq = np.where(face_only_mask > 0, wq + 2.0, wq)
            if hand_mask is not None and hand_mask.shape[:2] == mask.shape:
                wq = np.where(hand_mask > 0, wq + 2.0, wq)
            weight_q = wq
            weight_power = 4.0
        elif has_face or (hand_mask is not None and np.any(hand_mask)):
            wq = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
            if face_only_mask is not None and face_only_mask.shape[:2] == mask.shape:
                wq = np.where(face_only_mask > 0, wq + 1.5, wq)
            if hand_mask is not None and hand_mask.shape[:2] == mask.shape:
                wq = np.where(hand_mask > 0, wq + 1.5, wq)
            weight_q = wq
            weight_power = 4.0

        tone, label_map, centers = ps.quantize_tones(
            bgr,
            mask,
            n_colors=n_q,
            pre_smooth=do_smooth,
            weight_mask=weight_q,
            weight_power=weight_power if weight_q is not None else 1.0,
        )
    if density_gamma is not None:
        dens_gamma = float(np.clip(float(density_gamma), 0.35, 2.5))
    else:
        dens_gamma = (
            0.65
            if (detail_boost and has_face)
            else (0.68 if (animal_mode and is_fg_track) else (0.70 if detail_boost else 0.78))
        )
    dens = density_map(gray, mask, gamma=dens_gamma)
    # 场景/有脸罩时更宽核，便于五官留针；动物 FG 略宽核补毛纹暗部
    core_pct = (
        18.0
        if (detail_boost and (scene_mode or has_face))
        else (16.0 if (animal_mode and is_fg_track) else (14.0 if detail_boost else 22.0))
    )
    # 五官核优先在真脸罩上算，避免手区脏核
    core_src = face_only_mask if face_only_mask is not None and np.any(face_only_mask) else mask
    core = feature_core_mask(gray, core_src, percentile=core_pct)
    dens_use = dens.copy()
    face = mask > 0
    do_edge = bool(edge_boost) or (animal_mode and is_fg_track)
    # 建筑/街景边缘加密，减轻 BG「糊成一团」；动物 FG 同开
    if do_edge:
        edges = cv2.Canny(gray, 40, 120)
        e = (edges > 0) & face
        if np.any(e):
            dens_use[e] = np.minimum(1.0, dens_use[e] * 1.40 + 0.10)
            e_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
            e2 = e_dil & face & ~e
            dens_use[e2] = np.minimum(1.0, dens_use[e2] * 1.18 + 0.04)
    # 人物+场景：背景降密（V6.2 缓和），主体略抬
    sub_w = None
    if scene_mode:
        sub_w = subject_weight > 0
        bg = face & ~sub_w
        dens_use[bg] = dens_use[bg] * float(bg_dens_scale)
        dens_use[sub_w] = np.minimum(1.0, dens_use[sub_w] * 1.12 + 0.04)
    # 中调微抬
    mid = face & (dens >= 0.28) & (dens <= 0.82)
    if sub_w is not None:
        mid = mid & sub_w
    boost_m = 1.18 if detail_boost else 1.1
    dens_use[mid] = np.minimum(1.0, dens_use[mid] * boost_m + (0.08 if detail_boost else 0.05))
    # 高光→中调：把极亮压到可留针区间，保留层次而非死白
    bright = face & (dens < 0.32)
    if sub_w is not None:
        bright = bright & sub_w
    if np.any(bright):
        d = dens[bright]
        # 0..0.32 → 0.12..0.40
        dens_use[bright] = np.clip(0.12 + (d / 0.32) * 0.28, 0.12, 0.42).astype(np.float32)
    # 整脸罩 dens 抬（突破仅核区加密）
    sg_face = float(1.42 if structure_gain is None else structure_gain)
    if face_only_mask is not None and np.any(face_only_mask):
        fm0 = (face_only_mask > 0) & (mask > 0)
        dens_use[fm0] = np.minimum(1.0, dens_use[fm0] * 1.48 + 0.10)
        dens_use = face_structure_dens_boost(
            dens_use,
            gray,
            (fm0.astype(np.uint8) * 255),
            percentile=38.0 if structure_gain is not None else 50.0,
            gain=sg_face,
        )
    if np.any(core):
        boost = core > 0
        dens_use[boost] = np.minimum(1.0, dens_use[boost] * (1.72 if detail_boost else 1.3) + (0.28 if detail_boost else 0.14))
    # 手区 dens 抬（无结构叠层）
    if hand_mask is not None and np.any(hand_mask):
        hm0 = (hand_mask > 0) & (mask > 0)
        dens_use[hm0] = np.minimum(1.0, dens_use[hm0] * 1.48 + 0.10)
        dens_use = face_structure_dens_boost(
            dens_use,
            gray,
            (hm0.astype(np.uint8) * 255),
            percentile=48.0 if structure_gain is not None else 55.0,
            gain=float(1.30 if structure_gain is None else max(1.30, structure_gain * 0.9)),
        )
    # 动物 FG：无 dens_face 时整主体结构 dens，暗部/毛纹轮廓留针（非丝理方向场）
    if animal_mode and is_fg_track:
        dens_use = face_structure_dens_boost(
            dens_use,
            gray,
            mask,
            percentile=50.0,
            gain=1.30,
        )
        dens_use[face] = np.minimum(1.0, dens_use[face] * 1.10 + 0.03)
    if not reuse_palette:
        if soften_highlights and not (is_bg_track or is_fg_track):
            # 分桶轨已自管色心；旧统一轨仍软化高光
            centers = soften_highlight_centers(list(centers))
            tone = np.full_like(bgr, 255)
            for i, c in enumerate(centers):
                tone[label_map == i] = np.asarray(c, dtype=np.uint8).reshape(3)
    step_scale = 0.72 if detail_boost else 0.85
    # 允许 step<1（预算加密）；旧版 max(1.0, ...) 会抹掉 --stitches-per-mm2
    face_step = max(0.45, float(step) * step_scale)
    floor = float(keep_floor) if keep_floor is not None else (0.04 if detail_boost else 0.05)
    floor = float(np.clip(floor * float(keep_floor_scale), 0.012, 0.12))
    # 场景/FG：脸核更细；背景疏密缓和
    if (scene_mode or has_face) and detail_boost:
        core_step = 0.22
    elif detail_boost:
        core_step = 0.35
    else:
        core_step = 0.45
    if sub_w is not None:
        floor_bg = float(0.07 if bg_floor is None else bg_floor)
        keep_sub = sample_keep_grid(
            dither_keep_mask(dens_use, mask, floor=floor, gain=dither_gain),
            dens_use,
            mask,
            face_step,
            core_mask=core,
            core_step_scale=core_step,
        )
        bg_step = max(0.55, float(step) * float(bg_step_scale))
        keep_bg = sample_keep_grid(
            dither_keep_mask(dens_use, mask, floor=floor_bg, gain=dither_gain),
            dens_use,
            mask,
            bg_step,
            core_mask=None,
        )
        keep = np.where(sub_w, keep_sub, keep_bg).astype(np.uint8)
        keep = np.where(mask > 0, keep, 0).astype(np.uint8)
    else:
        keep = sample_keep_grid(
            dither_keep_mask(dens_use, mask, floor=floor, gain=dither_gain),
            dens_use,
            mask,
            face_step,
            core_mask=core,
            core_step_scale=core_step,
        )
    # 半身：主体再叠半相网格，逼近 1px 密度
    if detail_boost:
        h0, w0 = keep.shape
        phase = np.zeros_like(keep)
        phase[0:h0:1, 0:w0:1] = 255
        phase[1:h0:2, 1:w0:2] = 255
        phase_zone = mask > 0
        if sub_w is not None:
            phase_zone = phase_zone & sub_w
        keep = np.maximum(
            keep,
            ((phase > 0) & phase_zone & (dither_keep_mask(dens_use, mask, floor=0.04, gain=dither_gain) > 0)).astype(
                np.uint8
            )
            * 255,
        )
    # 脸/手：绕过网格，Bayer 全分辨率留针（突破 step≥1 上限）
    if face_only_mask is not None and np.any(face_only_mask):
        fm = (face_only_mask > 0) & (mask > 0)
        fkeep = dither_keep_mask(
            dens_use, (fm.astype(np.uint8) * 255), floor=0.02, gain=dither_gain
        )
        keep = np.where(fm, np.maximum(keep, fkeep), keep)
        if detail_boost:
            h0, w0 = keep.shape
            phase_f = np.zeros_like(keep)
            phase_f[0:h0:1, 0:w0:1] = 255
            phase_f[1:h0:2, 1:w0:2] = 255
            keep = np.maximum(
                keep,
                ((phase_f > 0) & fm & (fkeep > 0)).astype(np.uint8) * 255,
            )
    if hand_mask is not None and np.any(hand_mask):
        hm = (hand_mask > 0) & (mask > 0)
        hkeep = dither_keep_mask(dens_use, (hm.astype(np.uint8) * 255), floor=0.02, gain=dither_gain)
        keep = np.where(hm, np.maximum(keep, hkeep), keep)
        if detail_boost:
            h0, w0 = keep.shape
            phase_h = np.zeros_like(keep)
            phase_h[0:h0:1, 0:w0:1] = 255
            phase_h[1:h0:2, 1:w0:2] = 255
            keep = np.maximum(
                keep,
                ((phase_h > 0) & hm & (hkeep > 0)).astype(np.uint8) * 255,
            )
    # 挖空：让 dens_face/hand 独占该区，避免 FG 同色叠针把细节糊死
    if omit_mask is not None and omit_mask.shape[:2] == keep.shape[:2] and np.any(omit_mask):
        keep = np.where(omit_mask > 0, 0, keep).astype(np.uint8)
    h = mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    step_i = max(1, int(round(face_step)))
    gap = max(2, step_i + 1)
    layers: list[dict[str, Any]] = []
    if p0 and centers:
        from embroideryforge.thread_chart import sew_order_indices

        order = sew_order_indices(list(centers), grey_underpaint=True)
    else:
        order = sorted(
            range(len(centers)),
            key=lambda i: float(
                0.114 * centers[i][0] + 0.587 * centers[i][1] + 0.299 * centers[i][2]
            ),
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
        thread_meta: dict[str, Any] = {}
        if p0:
            from embroideryforge.thread_chart import classify_center_role, snap_bgr_to_chart

            sm = snap_bgr_to_chart(centers[idx])
            thread_meta = {
                "thread_code": sm["code"],
                "thread_role": sm["role"],
                "role_hsv": classify_center_role(centers[idx]),
                "delta_e": sm["delta_e"],
            }
        first, n_pts = _emit_polys(
            pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
        )
        if n_pts >= 2:
            layer_rec: dict[str, Any] = {
                "tag": f"{tag_prefix}_{idx}",
                "hex": hex_c,
                "bgr": [int(x) for x in centers[idx]],
                "stitch": "density_serpentine",
                "points": n_pts,
                "segments": len(polys),
                "enabled": True,
                "track": (
                    "bg"
                    if tag_prefix.startswith("dens_bg")
                    else (
                        "fg"
                        if tag_prefix.startswith("dens_fg")
                        else (
                            "face"
                            if tag_prefix.startswith("dens_face")
                            else (
                                "hand"
                                if tag_prefix.startswith("dens_hand")
                                else "unified"
                            )
                        )
                    )
                ),
            }
            if thread_meta:
                layer_rec.update(thread_meta)
            layers.append(layer_rec)
    if photo_blend and len(centers) >= 2:
        second, mix = ps.second_nearest_map(
            bgr, mask, centers, label_map, ab_weight=2.2
        )
        yy, xx = np.indices(mask.shape[:2])
        bayer = ps._BAYER8[yy % 8, xx % 8]
        blend_on = (
            (keep > 0)
            & (second >= 0)
            & (second != label_map)
            & (mix >= 0.22)
            & (mix <= 0.62)
            & (bayer < 0.40)
        )
        for idx in order:
            ys, xs = np.where(blend_on & (second == idx))
            if len(xs) < 8:
                continue
            polys = points_to_serpentine_runs(
                ys, xs, max_gap=gap * 2, max_row_jump=1, max_connect_dx=gap * 2
            )
            hex_c = ps._bgr_to_hex(centers[idx])
            first, n_pts = _emit_polys(
                pattern, polys, hex_c=hex_c, scale=scale, height=h, first=first
            )
            if n_pts >= 2:
                layers.append(
                    {
                        "tag": f"{tag_prefix}_blend_{idx}",
                        "hex": hex_c,
                        "bgr": [int(x) for x in centers[idx]],
                        "stitch": "density_blend",
                        "points": n_pts,
                        "segments": len(polys),
                        "enabled": True,
                        "track": "blend",
                    }
                )
    maps = {
        "dens": dens,
        "keep": keep,
        "tone": tone,
        "core": core,
        "gray": gray,
        "label_map": label_map,
        "centers": centers,
    }
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
    domain: str = "auto",
    portrait_p0: bool | None = None,
    stitches_per_mm2: float | None = None,
    density_contrast: float | None = None,
    stitch_mask: str | Path | None = None,
    photo_blend: bool = False,
) -> dict[str, Any]:
    """V6：整主体统一彩色密度短针；bg=keep/scene 时 V6.3 人物/背景分轨再整合。

    domain: auto|figure|animal — animal 强制跳过 dens_face（防宠物误检成人脸）。
    portrait_p0: 人像 P0（Reserve+灰垫底；线表仅标注）。None/默认关——全图贴表曾致蒙红，人评不如 v66。
    stitches_per_mm2 / density_contrast：S2 密度预算旋钮（不换填法）。
    stitch_mask: 只绣这张罩。方案 B 用来限定一只动物，不改 dens 公式。
    """
    ps = _ps()
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_s = str(domain or "auto").strip().lower()
    if domain_s not in ("auto", "figure", "animal"):
        domain_s = "auto"
    # 默认关：须显式 --portrait-p0
    if portrait_p0 is None:
        p0_on = False
    else:
        p0_on = bool(portrait_p0) and domain_s != "animal"
    dens_gamma_override = (
        float(density_contrast) if density_contrast is not None else None
    )
    # 高预算且未显式给 contrast 时，略降 gamma → 中调 dens 升高 → 更多 keep
    if (
        dens_gamma_override is None
        and stitches_per_mm2 is not None
        and float(stitches_per_mm2) > 6.0
    ):
        dens_gamma_override = float(
            np.clip(0.78 * (6.0 / float(stitches_per_mm2)) ** 0.35, 0.45, 0.78)
        )

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(image_path)

    if stitch_mask is not None:
        raw_m = cv2.imread(str(stitch_mask), cv2.IMREAD_GRAYSCALE)
        if raw_m is None:
            raise FileNotFoundError(stitch_mask)
        if raw_m.shape[:2] != original.shape[:2]:
            raw_m = cv2.resize(
                raw_m,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        subject_bgr = original.copy()
        subject_mask = np.where(raw_m > 127, 255, 0).astype(np.uint8)
        bg_meta = {
            "mode": "stitch_mask",
            "applied": True,
            "scene": False,
            "method": "given_mask",
            "b0_one_animal": True,
        }
    else:
        subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    scene = bool(bg_meta.get("scene"))
    person_mask = bg_meta.get("subject_mask")
    if person_mask is None:
        person_mask = subject_mask
    subject_path = out_dir / "subject.png"
    cv2.imwrite(str(subject_path), subject_bgr)
    cv2.imwrite(str(out_dir / "subject_mask.png"), person_mask if scene else subject_mask)

    cfg = dict(ps.RESOLUTION.get(resolution, ps.RESOLUTION["high"]))
    # V6.3c/V6.6：场景 720 工作边（840 针数涨、人评无感，已回退）
    cfg["max_side"] = max(int(cfg["max_side"]), 520 if not scene else 720)
    spacing = float(cfg["spacing"]) * (0.92 if not scene else 0.88)
    step = max(1.0, spacing * 1.0)
    step_before_budget = float(step)
    budget_stitches = (
        float(stitches_per_mm2) if stitches_per_mm2 is not None else None
    )
    budget_floor_scale = 1.0
    if budget_stitches is not None and budget_stitches > 0:
        # 线性映射。参照 6：保持默认 step；12 → step≈0.5；3 → step≈2。
        # 注意：旧 sample_keep_grid 把 step<1 全打成 1，已修多相采样。
        ref = 6.0
        ratio = ref / max(0.5, budget_stitches)
        # step<1 会走进「亮部隔点抽稀」，加密预算反而更疏。加密只降 Bayer 阈值。
        if budget_stitches >= ref:
            step = float(step_before_budget)
        else:
            step = float(np.clip(step_before_budget * ratio, 1.0, 2.8))
        if budget_stitches > ref:
            budget_floor_scale = float(np.clip((ref / budget_stitches) ** 1.4, 0.28, 1.0))
        elif budget_stitches < ref:
            budget_floor_scale = float(np.clip((ref / budget_stitches) ** 0.8, 1.0, 2.2))
        dither_gain = float(np.clip((budget_stitches / ref) ** 1.35, 0.4, 2.4))
    else:
        dither_gain = 1.0
    # V6.2 场景：色数上限 16（Wilcom 复杂图量级）
    n_colors = int(np.clip(max(n_colors, 10), 8, 16 if scene else 12))
    bg_dens_scale = 0.58 if scene else None
    bg_step_scale = 1.40 if scene else None

    guide_mask = person_mask if scene else subject_mask
    geo = ps.detect_face_geometry(subject_bgr, subject_mask=guide_mask)
    roi_src = geo["roi"]
    face_landmarks_src = geo.get("landmarks")
    all_landmarks_src = list(geo.get("all_landmarks") or [])
    if not all_landmarks_src and face_landmarks_src is not None:
        all_landmarks_src = [face_landmarks_src]
    n_faces_detect = int(geo.get("n_faces") or len(all_landmarks_src) or 0)
    face_source = str(geo.get("source") or "unknown")
    x0, y0, w0, h0 = roi_src
    boxed = subject_bgr.copy()
    cv2.rectangle(boxed, (x0, y0), (x0 + w0, y0 + h0), (0, 180, 255), 2)
    # 次要脸 ROI 用青绿框标出，便于对照
    for pts_i in all_landmarks_src[1:]:
        try:
            from embroideryforge.face_detect import roi_from_landmarks

            rx, ry, rw, rh = roi_from_landmarks(pts_i, subject_bgr.shape[:2])
            cv2.rectangle(boxed, (rx, ry), (rx + rw, ry + rh), (80, 220, 80), 2)
        except Exception:
            pass
    cv2.imwrite(str(out_dir / "roi_box.png"), boxed)

    guide_px = max(1, int(np.count_nonzero(guide_mask)))
    face_frac = float(w0 * h0) / float(guide_px)

    work_r, resize_scale = ps.resize_for_resolution(
        subject_bgr, resolution, max_side=int(cfg["max_side"])
    )
    work_mask_r = cv2.resize(
        subject_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    if not scene:
        work_mask_r = ps._strip_fringe_white(work_r, work_mask_r)
    dens_mask = work_mask_r.copy()
    person_r = cv2.resize(
        guide_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    if scene:
        person_r = cv2.morphologyEx(person_r, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    roi = ps._scale_roi(roi_src, subject_bgr.shape[:2], work_r.shape[:2])
    mm_per_px = float(target_height_mm) / float(max(work_r.shape[0], 1))
    face_width_mm = float(roi[2]) * mm_per_px
    face_width_warn = bool(face_width_mm < 100.0) and domain_s != "animal"
    lm_r = ps._scale_landmarks(face_landmarks_src, subject_bgr.shape[:2], work_r.shape[:2])
    # V6.6b：多脸罩并集（次要脸 ≥ 主脸面积 12% 才并入，防噪）
    face_m = np.zeros(work_r.shape[:2], dtype=np.uint8)
    primary_px = 0
    for i, pts_src in enumerate(all_landmarks_src):
        lm_i = ps._scale_landmarks(pts_src, subject_bgr.shape[:2], work_r.shape[:2])
        if lm_i is None:
            continue
        from embroideryforge.face_detect import roi_from_landmarks

        roi_i = roi_from_landmarks(lm_i, work_r.shape[:2])
        fm_i = ps._face_mask(
            work_r.shape[:2], roi_i, subject_mask=person_r, landmarks=lm_i
        )
        fm_i = cv2.bitwise_and(fm_i, person_r)
        px_i = int(np.count_nonzero(fm_i))
        if i == 0:
            primary_px = max(px_i, 1)
            face_m = fm_i
        elif px_i >= max(80, int(primary_px * 0.12)):
            face_m = cv2.bitwise_or(face_m, fm_i)
    if not np.any(face_m):
        face_m = ps._face_mask(
            work_r.shape[:2], roi, subject_mask=person_r, landmarks=lm_r
        )
        face_m = cv2.bitwise_and(face_m, person_r)
    # Haar 补检：并入与已有脸罩重叠低的侧脸窗
    try:
        haar_rois = ps.detect_all_face_rois_haar(subject_bgr)
    except Exception:
        haar_rois = []
    primary_px = max(int(np.count_nonzero(face_m)), primary_px, 1)
    for hr in haar_rois[:5]:
        roi_h = ps._scale_roi(hr, subject_bgr.shape[:2], work_r.shape[:2])
        fm_h = ps._face_mask(work_r.shape[:2], roi_h, subject_mask=person_r, landmarks=None)
        fm_h = cv2.bitwise_and(fm_h, person_r)
        px_h = int(np.count_nonzero(fm_h))
        if px_h < max(80, int(primary_px * 0.12)):
            continue
        overlap = int(np.count_nonzero((fm_h > 0) & (face_m > 0)))
        if overlap > 0.40 * px_h:
            continue
        face_m = cv2.bitwise_or(face_m, fm_h)
        n_faces_detect = max(n_faces_detect, 1) + 1
    face_m = tighten_bust_face_mask(
        person_r, face_m, face_source, bgr=work_r
    )
    # 人脸密轨门控：animal 域强制关；auto 需 MediaPipe 真人脸；figure 尽量保留
    mp_face_ok = bool(
        face_source == "mediapipe"
        and face_landmarks_src is not None
        and int(n_faces_detect) >= 1
        and int(np.count_nonzero(face_m)) >= 120
    )
    if domain_s == "animal":
        valid_human_face = False
    elif domain_s == "figure":
        valid_human_face = bool(mp_face_ok or int(np.count_nonzero(face_m)) >= 120)
    else:
        valid_human_face = bool(mp_face_ok)
    if not valid_human_face:
        face_m = np.zeros(work_r.shape[:2], dtype=np.uint8)
        n_faces_detect = 0

    reserve_bgr: list[np.ndarray] = []
    if p0_on and np.any(face_m):
        from embroideryforge.thread_chart import sample_reserve_bgr

        reserve_bgr = sample_reserve_bgr(work_r, face_m)

    hand_m = np.zeros(work_mask_r.shape, dtype=np.uint8)
    try:
        from embroideryforge.figure_detect import detect_figure_geometry

        figure_geo = detect_figure_geometry(
            subject_bgr, subject_mask=guide_mask, out_shape=work_r.shape[:2]
        )
        hm = figure_geo.get("hand_mask")
        if hm is not None and np.any(hm):
            hand_m = cv2.bitwise_and(hm, person_r if scene else work_mask_r)
            # 略膨胀，枪柄/指缝别丢
            hand_m = cv2.dilate(hand_m, np.ones((5, 5), np.uint8), iterations=1)
            if scene:
                hand_m = cv2.bitwise_and(hand_m, person_r)
            else:
                hand_m = cv2.bitwise_and(hand_m, work_mask_r)
    except Exception:
        figure_geo = None

    pattern = EmbPattern()
    split = bool(scene)
    # V6.5：BG/FG 色相分桶量化；脸/手密轨沿用 V6.4d
    n_colors_bg = int(np.clip(max(12, n_colors + 2), 12, 16))
    n_colors_fg = int(np.clip(max(12, n_colors + 2), 12, 16))
    n_colors_face = int(np.clip(max(5, n_colors // 2), 5, 7))
    layers_face: list[dict[str, Any]] = []
    layers_hand: list[dict[str, Any]] = []

    if split:
        # V6.3：BG 洞略腐蚀、FG 略膨胀 → 边缘重叠，后绣人物盖缝
        ker = np.ones((3, 3), np.uint8)
        person_erode = cv2.erode(person_r, ker, iterations=2)
        person_dilate = cv2.dilate(person_r, ker, iterations=1)
        bg_mask = cv2.bitwise_and(dens_mask, cv2.bitwise_not(person_erode))
        fg_mask = cv2.bitwise_and(dens_mask, person_dilate)
        cv2.imwrite(str(out_dir / "bg_mask.png"), bg_mask)
        cv2.imwrite(str(out_dir / "fg_mask.png"), fg_mask)

        # 背景：禁 bilateral；轻锐化；色相分桶保红联/金瓦
        animal_run = domain_s == "animal"
        bg_bgr = unsharp_masked(work_r, bg_mask, radius=1.0, amount=0.35)
        pattern, layers_bg, first, maps_bg = emit_density_layers(
            pattern,
            bg_bgr,
            bg_mask,
            n_colors=n_colors_bg,
            step=max(0.45, float(step) * 1.18),
            target_height_mm=target_height_mm,
            first=True,
            detail_boost=False,
            tag_prefix="dens_bg",
            keep_floor=0.05,
            pre_smooth=False,
            edge_boost=True,
            soften_highlights=False,
            animal_palette=animal_run,
            portrait_p0=p0_on,
            reserve_bgr=reserve_bgr or None,
            density_gamma=dens_gamma_override,
            keep_floor_scale=budget_floor_scale,
            dither_gain=dither_gain,
            photo_blend=photo_blend,
        )
        face_fg = cv2.bitwise_and(face_m, fg_mask)
        hand_fg = cv2.bitwise_and(hand_m, fg_mask) if np.any(hand_m) else None
        # 人物整体锐化；FG 轻 CLAHE（脸细节）；色相分桶保青绿衣
        fg_bgr = unsharp_masked(work_r, fg_mask, radius=1.15, amount=0.55)
        if np.any(face_fg):
            fg_bgr = unsharp_masked(fg_bgr, face_fg, radius=0.9, amount=0.70)
            fg_bgr = clahe_face_bgr(fg_bgr, face_fg, clip_limit=1.5, blend=0.40)

        # 实心脸/手窗；FG 挖空给 dens_face/hand 独占
        face_pass = face_fg.copy() if np.any(face_fg) else cv2.bitwise_and(face_m, fg_mask)
        if np.any(face_pass):
            face_pass = cv2.dilate(face_pass, np.ones((3, 3), np.uint8), iterations=1)
            face_pass = cv2.bitwise_and(face_pass, fg_mask)
        hand_pass = None
        if hand_fg is not None and np.any(hand_fg):
            hand_pass = cv2.dilate(hand_fg, np.ones((3, 3), np.uint8), iterations=1)
            hand_pass = cv2.bitwise_and(hand_pass, fg_mask)

        omit_for_fg = np.zeros_like(fg_mask)
        if np.any(face_pass):
            carve = cv2.erode(face_pass, np.ones((3, 3), np.uint8), iterations=1)
            if int(np.count_nonzero(carve)) < 80:
                carve = face_pass
            omit_for_fg = cv2.bitwise_or(omit_for_fg, carve)
        if hand_pass is not None and np.any(hand_pass):
            carve_h = cv2.erode(hand_pass, np.ones((2, 2), np.uint8), iterations=1)
            if int(np.count_nonzero(carve_h)) < 40:
                carve_h = hand_pass
            omit_for_fg = cv2.bitwise_or(omit_for_fg, carve_h)

        pattern, layers_fg, first, maps_fg = emit_density_layers(
            pattern,
            fg_bgr,
            fg_mask,
            n_colors=n_colors_fg,
            step=max(0.45, float(step) * (0.90 if animal_run else 0.95)),
            target_height_mm=target_height_mm,
            first=first,
            detail_boost=True,
            hand_mask=hand_fg if hand_fg is not None and np.any(hand_fg) else None,
            face_only_mask=face_fg if np.any(face_fg) else face_m,
            tag_prefix="dens_fg",
            pre_smooth=False,
            omit_mask=omit_for_fg if np.any(omit_for_fg) else None,
            soften_highlights=False,
            edge_boost=bool(animal_run),
            animal_palette=animal_run,
            portrait_p0=p0_on,
            reserve_bgr=reserve_bgr or None,
            density_gamma=dens_gamma_override,
            keep_floor_scale=budget_floor_scale,
            dither_gain=dither_gain,
            photo_blend=photo_blend,
        )
        layers = layers_bg + layers_fg
        dens = np.maximum(maps_bg["dens"], maps_fg["dens"])
        keep = np.maximum(maps_bg["keep"], maps_fg["keep"]).astype(np.uint8)
        tone = maps_bg["tone"].copy()
        tone = np.where(fg_mask[..., None] > 0, maps_fg["tone"], tone)
        core = maps_fg.get("core")

        # V6.4b：同色线 + CLAHE 后重映射 label；结构 dens 加力
        # 动物/无人脸：跳过 dens_face（只跑 BG/FG）
        fg_labels = maps_fg.get("label_map")
        fg_centers = maps_fg.get("centers")
        if (
            valid_human_face
            and np.any(face_pass)
            and int(np.count_nonzero(face_pass)) >= 120
            and fg_labels is not None
            and fg_centers
        ):
            # V6.4d：贴近 v64 肤色 — 轻 CLAHE；重映射仅落在结构区，大面肤色用 FG label
            face_bgr = clahe_face_bgr(fg_bgr, face_pass, clip_limit=1.45, blend=0.30)
            face_bgr = dampen_face_highlights(face_bgr, face_pass, pct=80.0, pull=0.68)
            face_bgr = unsharp_masked(face_bgr, face_pass, radius=0.85, amount=0.48)
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_remap = remap_labels_to_centers(
                face_bgr, face_pass, fg_centers, base_labels=fg_labels
            )
            # V6.6c：结构区略扩（percentile↓）+ dens 抬；V6.6d：脸轨更细 step
            face_labels = blend_face_labels_structure(
                fg_labels, face_remap, face_gray, face_pass, percentile=52.0
            )
            pattern, layers_face, first, maps_face = emit_density_layers(
                pattern,
                face_bgr,
                face_pass,
                n_colors=len(fg_centers),
                step=max(0.45, float(step) * 0.50),
                target_height_mm=target_height_mm,
                first=first,
                detail_boost=True,
                face_only_mask=face_pass,
                tag_prefix="dens_face",
                pre_smooth=False,
                keep_floor=0.010,
                fixed_label_map=face_labels,
                fixed_centers=fg_centers,
                structure_gain=1.62,
                portrait_p0=p0_on,
                reserve_bgr=reserve_bgr or None,
                density_gamma=dens_gamma_override,
                keep_floor_scale=budget_floor_scale,
            dither_gain=dither_gain,
            photo_blend=photo_blend,
            )
            layers = layers + layers_face
            n_colors_face = len(fg_centers)
            dens = np.maximum(dens, maps_face["dens"])
            keep = np.maximum(keep, maps_face["keep"]).astype(np.uint8)
            tone = np.where(face_pass[..., None] > 0, maps_face["tone"], tone)
            if maps_face.get("core") is not None and np.any(maps_face["core"]):
                core = maps_face["core"]
            cv2.imwrite(str(out_dir / "face_pass_mask.png"), face_pass)

        if (
            hand_pass is not None
            and np.any(hand_pass)
            and int(np.count_nonzero(hand_pass)) >= 80
            and fg_labels is not None
            and fg_centers
        ):
            hand_bgr = clahe_face_bgr(fg_bgr, hand_pass, clip_limit=1.4, blend=0.28)
            hand_bgr = dampen_face_highlights(hand_bgr, hand_pass, pct=82.0, pull=0.75)
            hand_bgr = unsharp_masked(hand_bgr, hand_pass, radius=0.90, amount=0.38)
            hand_gray = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2GRAY)
            hand_remap = remap_labels_to_centers(
                hand_bgr, hand_pass, fg_centers, base_labels=fg_labels
            )
            hand_labels = blend_face_labels_structure(
                fg_labels, hand_remap, hand_gray, hand_pass, percentile=55.0
            )
            pattern, layers_hand, first, maps_hand = emit_density_layers(
                pattern,
                hand_bgr,
                hand_pass,
                n_colors=len(fg_centers),
                step=max(0.45, float(step) * 0.70),
                target_height_mm=target_height_mm,
                first=first,
                detail_boost=True,
                hand_mask=hand_pass,
                tag_prefix="dens_hand",
                pre_smooth=False,
                keep_floor=0.016,
                fixed_label_map=hand_labels,
                fixed_centers=fg_centers,
                structure_gain=1.38,
                portrait_p0=p0_on,
                reserve_bgr=reserve_bgr or None,
                density_gamma=dens_gamma_override,
                keep_floor_scale=budget_floor_scale,
            dither_gain=dither_gain,
            photo_blend=photo_blend,
            )
            layers = layers + layers_hand
            dens = np.maximum(dens, maps_hand["dens"])
            keep = np.maximum(keep, maps_hand["keep"]).astype(np.uint8)
            tone = np.where(hand_pass[..., None] > 0, maps_hand["tone"], tone)
            cv2.imwrite(str(out_dir / "hand_pass_mask.png"), hand_pass)
        maps = {"dens": dens, "keep": keep, "tone": tone, "core": core}
    else:
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
            subject_weight=None,
            bg_dens_scale=0.42,
            bg_step_scale=1.75,
            portrait_p0=p0_on,
            reserve_bgr=reserve_bgr or None,
            density_gamma=dens_gamma_override,
            keep_floor_scale=budget_floor_scale,
            dither_gain=dither_gain,
            photo_blend=photo_blend,
        )
        layers = dens_layers
        dens, keep, tone = maps["dens"], maps["keep"], maps["tone"]
        core = maps.get("core")

    pattern.end()

    dens_vis = render_density_preview(dens, dens_mask)
    plan = work_r.copy()
    if not scene:
        plan[work_mask_r == 0] = PREVIEW_BG_DARK
    mix = (tone.astype(np.float32) * 0.55).astype(np.uint8)
    plan_tone = np.where(keep[..., None] > 0, mix, tone)
    plan = np.where(dens_mask[..., None] > 0, plan_tone, plan)
    if not scene:
        plan = np.where(
            work_mask_r[..., None] == 0, np.array(PREVIEW_BG_DARK, dtype=np.uint8), plan
        )

    if core is not None and np.any(core):
        cv2.imwrite(str(out_dir / "feature_core.png"), core)
    cv2.imwrite(str(out_dir / "face_mask.png"), face_m)
    cv2.imwrite(str(out_dir / "hand_mask.png"), hand_m)
    cv2.imwrite(str(out_dir / "person_mask.png"), person_r)
    cv2.imwrite(str(out_dir / "dens_mask.png"), dens_mask)
    cv2.imwrite(str(out_dir / "density_map.png"), dens_vis)
    cv2.imwrite(str(out_dir / "plan_preview.png"), plan)
    cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
    cv2.imwrite(str(out_dir / "keep_mask.png"), keep)
    layout = work_r.copy()
    if not scene:
        layout[work_mask_r == 0] = PREVIEW_BG_DARK
    layout = np.where(
        dens_mask[..., None] > 0,
        (layout.astype(np.float32) * 0.65 + dens_vis.astype(np.float32) * 0.35).astype(np.uint8),
        layout,
    )
    if np.any(person_r):
        layout = np.where(
            person_r[..., None] > 0,
            (layout.astype(np.float32) * 0.7 + np.array([60, 180, 90]) * 0.3).astype(np.uint8),
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
    bg_meta_json = {
        k: v
        for k, v in bg_meta.items()
        if not isinstance(v, np.ndarray)
    }
    meta = {
        "engine": "photo_stitch_density_v6",
        "mode": "density_split" if split else "density_unified",
        "n_colors": n_colors,
        "n_colors_bg": n_colors_bg if split else None,
        "n_colors_fg": n_colors_fg if split else None,
        "n_colors_face": (n_colors_face if split and layers_face else None),
        "resolution": resolution,
        "step_px": step,
        "step_px_before_budget": step_before_budget,
        "stitches_per_mm2": budget_stitches,
        "dither_gain": dither_gain,
        "density_contrast": dens_gamma_override,
        "resize_scale": resize_scale,
        "face_frac": round(float(face_frac), 4),
        "face_width_mm": round(float(face_width_mm), 2),
        "face_width_warn": bool(face_width_warn),
        "warnings": (
            [f"face_width_mm={face_width_mm:.1f}<100; dens not raised"]
            if face_width_warn
            else []
        ),
        "n_hands": int((figure_geo or {}).get("n_hands") or 0),
        "scene": {
            "enabled": bool(scene),
            "person_px": int(np.count_nonzero(person_r)),
            "frame_px": int(np.count_nonzero(dens_mask)),
            "bg_dens_scale": bg_dens_scale,
            "bg_step_scale": bg_step_scale,
            "bg_floor": 0.07 if scene else None,
            "split_tracks": bool(split),
            "fg_unsharp": bool(split),
        },
        "roi": {
            "mode": "face",
            "xywh": [int(x0), int(y0), int(w0), int(h0)],
            "detector": face_source,
        },
        "domain": (
            "animal"
            if domain_s == "animal" or (domain_s == "auto" and not valid_human_face)
            else "figure"
        ),
        "domain_request": domain_s,
        "face_detect": {
            "source": face_source,
            "mediapipe": face_source == "mediapipe",
            "valid_human_face": bool(valid_human_face),
        },
        "bg": bg_meta_json,
        "palette": [L.get("hex") for L in layers if L.get("hex")],
        "layers": layers,
        "stitches": report["stitches"],
        "color_changes": report["color_changes"],
        "v6": {
            "unified_subject_density": not split,
            "no_body_tatami": True,
            "no_fold_overlays": True,
            "soft_weight_face_hand": True,
            "person_plus_scene": bool(scene),
            "v63_person_bg_split": bool(split),
            "v63c_face_dense_track": bool(split and bool(layers_face)),
            "v63d_face_share_fg_palette": bool(split and bool(layers_face)),
            "v64_person_detail": bool(split),
            "v64_hand_dense_track": bool(split and bool(layers_hand)),
            "v64b_face_carve_remap": bool(split and bool(layers_face)),
            "v64c_face_tone_soft": bool(split and bool(layers_face)),
            "v64d_structure_only_remap": bool(split and bool(layers_face)),
            "v64e_bg_chroma": False,
            "v64f_fg_cool_balance": False,
            "v65_hue_band_palette": False,
            "v65b_global_rebind": False,
            "v65c_no_label_median": False,
            "v66_flatten_palette": bool(split),
            "v66b_multi_face": bool(split and n_faces_detect > 1),
            "v66c_face_structure": bool(split and bool(layers_face)),
            "v66d_face_finer_step": bool(split and bool(layers_face)),
            "v66_animal_a0": bool(split and not valid_human_face and domain_s != "animal"),
            "v66_animal_a1": bool(split and domain_s == "animal"),
            "animal_no_skin_dampen": bool(split and domain_s == "animal"),
            "animal_fg_structure_dens": bool(split and domain_s == "animal"),
            "v66_portrait_p0": bool(p0_on),
            "density_budget": bool(budget_stitches is not None or dens_gamma_override is not None),
            "photo_blend": bool(photo_blend),
            "portrait_thread_chart": False,
            "portrait_thread_meta_only": bool(p0_on),
            "portrait_reserve": bool(p0_on and bool(reserve_bgr)),
            "portrait_grey_underpaint": bool(p0_on),
            "portrait_no_global_snap": True,
            "n_reserve_colors": int(len(reserve_bgr)),
            "n_faces": int(n_faces_detect),
            "v62b_face_hand_fullres": True,
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
        "note": (
            (
                "V6.6 animal A1: no warm-skin dampen; FG structure dens + edges; BG green protect; dens_face skipped; no fur direction"
                if (split and domain_s == "animal")
                else (
                    "V6.6 animal A0: flatten palette BG/FG only; dens_face skipped (no MediaPipe human face)"
                    if (split and not valid_human_face)
                    else (
                        "V6.6 + portrait P0: Reserve skin/lips/eyes + grey underpaint; no global thread-chart snap (avoids red cast)"
                        if p0_on
                        else "V6.6: flatten palette + multi-face dens_face union; max_side=720 (detail bumps rolled back)"
                    )
                )
            )
            if split
            else "V6 unified subject density"
        ),
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                **report,
                "density_field": True,
                "density_unified": not split,
                "density_split": bool(split),
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    subject_bgr, subject_mask, meta = prepare_subject(original, mode=bg_mode)
    guide = meta.get("subject_mask") if isinstance(meta.get("subject_mask"), np.ndarray) else subject_mask
    geo = ps.detect_face_geometry(subject_bgr, subject_mask=guide)
    x0, y0, w0, h0 = geo["roi"]
    sub_px = max(1, int(np.count_nonzero(guide)))
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
    domain: str = "auto",
    portrait_p0: bool | None = None,
    stitches_per_mm2: float | None = None,
    density_contrast: float | None = None,
    stitch_mask: str | Path | None = None,
    photo_blend: bool = False,
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
            domain=domain,
            portrait_p0=portrait_p0,
            stitches_per_mm2=stitches_per_mm2,
            density_contrast=density_contrast,
            stitch_mask=stitch_mask,
            photo_blend=photo_blend,
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
