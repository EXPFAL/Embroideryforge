"""Embroidery-Friendly Representation：照片 → 少色硬边中间图。

去掉连续渐变，保留可绣的扁平底色块，供 segment / stitch-auto 使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def apply_efr(
    bgr: np.ndarray,
    *,
    n_colors: int = 7,
    min_region_frac: float = 0.0015,
    morph_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """返回 (efr_bgr, meta)。白底/近白像素尽量保持为白。"""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    h, w = bgr.shape[:2]
    n_colors = int(max(3, min(n_colors, 16)))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    flat = lab.reshape(-1, 3).astype(np.float32)
    near_white = np.all(bgr.reshape(-1, 3) >= 248, axis=1)
    work = flat[~near_white] if int((~near_white).sum()) >= n_colors * 40 else flat

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels_sub, centers = cv2.kmeans(work, n_colors, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    centers_u8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    centers_bgr = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    label_img = np.full((h * w,), -1, dtype=np.int32)
    if int((~near_white).sum()) >= n_colors * 40:
        label_img[~near_white] = labels_sub.flatten()
    else:
        label_img[:] = labels_sub.flatten()
    label_img = label_img.reshape(h, w)

    min_area = max(32, int(h * w * min_region_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))
    cleaned = label_img.copy()
    for i in range(n_colors):
        mask = (label_img == i).astype(np.uint8) * 255
        if not np.any(mask):
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        ncc, markers, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for c in range(1, ncc):
            if int(stats[c, cv2.CC_STAT_AREA]) >= min_area:
                keep[markers == c] = 255
        cleaned[label_img == i] = -1
        cleaned[keep > 0] = i

    # 小洞：向邻域膨胀填标签（原近白保持 -1 → 输出白）
    near_white_2d = np.all(bgr >= 248, axis=2)
    dil_k = np.ones((3, 3), np.uint8)
    for _ in range(12):
        unset = (cleaned < 0) & (~near_white_2d)
        if not np.any(unset):
            break
        for i in range(n_colors):
            dil = cv2.dilate((cleaned == i).astype(np.uint8), dil_k, iterations=1)
            fill = unset & (dil > 0)
            cleaned[fill] = i

    out = np.full_like(bgr, 255)
    for i in range(n_colors):
        m = cleaned == i
        if np.any(m):
            out[m] = centers_bgr[i]
    out[near_white_2d] = 255

    unique = len({tuple(int(x) for x in centers_bgr[i]) for i in range(n_colors) if np.any(cleaned == i)})
    meta = {
        "engine": "efr_lab_kmeans",
        "n_colors_request": n_colors,
        "n_colors_used": unique,
        "min_region_frac": min_region_frac,
        "morph_radius": morph_radius,
        "width": w,
        "height": h,
    }
    return out, meta


def write_efr(
    bgr: np.ndarray,
    out_dir: str | Path,
    *,
    n_colors: int = 7,
    stem: str = "efr",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    efr_bgr, meta = apply_efr(bgr, n_colors=n_colors)
    png = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}_meta.json"
    cv2.imwrite(str(png), efr_bgr)
    meta["ok"] = png.is_file() and png.stat().st_size > 50
    meta["file"] = png.name
    disk_meta = {k: v for k, v in meta.items()}
    meta_path.write_text(json.dumps(disk_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["path"] = str(png)
    meta["meta_path"] = str(meta_path)
    meta["image"] = efr_bgr
    return meta
