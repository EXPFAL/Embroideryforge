"""主体抠图：复杂照片背景先去掉，再进 palette 分区。

优先 rembg（若已安装）；否则用插画色种子重建；白底图自动跳过。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def border_stats(bgr: np.ndarray, band: int = 10) -> dict[str, float]:
    h, w = bgr.shape[:2]
    band = max(2, min(band, h // 8, w // 8))
    parts = [
        bgr[:band].reshape(-1, 3),
        bgr[-band:].reshape(-1, 3),
        bgr[:, :band].reshape(-1, 3),
        bgr[:, -band:].reshape(-1, 3),
    ]
    pix = np.concatenate(parts, axis=0).astype(np.float32)
    return {
        "mean": float(pix.mean()),
        "std": float(pix.std()),
        "near_white_frac": float(np.mean(np.all(pix >= 245, axis=1))),
    }


def detect_bg_mode(bgr: np.ndarray) -> str:
    st = border_stats(bgr)
    if st["near_white_frac"] >= 0.55 and st["std"] < 35:
        return "skip"
    if st["mean"] >= 235 and st["std"] < 28:
        return "skip"
    return "contrast"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if ncc <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def mask_near_white(bgr: np.ndarray, thr: int = 248) -> np.ndarray:
    fg = (np.any(bgr < thr, axis=2)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_component(fg)


def mask_rembg(bgr: np.ndarray) -> np.ndarray | None:
    try:
        from rembg import remove
        from PIL import Image
    except ImportError:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(Image.fromarray(rgb).convert("RGBA"))
    alpha = np.array(rgba)[:, :, 3]
    mask = (alpha > 12).astype(np.uint8) * 255
    mask = _largest_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def mask_illustration_recon(bgr: np.ndarray) -> np.ndarray:
    """无 rembg 时的兜底：从粉/肤/绿种子长到西装灰，再贴描边。"""
    h, w = bgr.shape[:2]
    B, G, R = cv2.split(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    pink = ((H <= 12) | (H >= 165)) & (S >= 28) & (V >= 75)
    green = (H >= 35) & (H <= 95) & (S >= 25) & (V >= 70)
    skin = ((H <= 22) | (H >= 170)) & (S >= 18) & (S <= 130) & (V >= 140)
    suit = (
        (S <= 40)
        & (V >= 85)
        & (V <= 175)
        & (B.astype(int) <= R.astype(int) + 12)
        & (B.astype(int) <= G.astype(int) + 12)
    )
    dark = V <= 40
    wine = ((H <= 8) | (H >= 170)) & (S >= 55) & (V >= 50) & (V <= 170)
    whiteish = (V >= 245) & (S <= 30)
    band = max(8, min(h, w) // 35)
    nb = np.zeros((h, w), dtype=bool)
    nb[:band] = True
    nb[-band:] = True
    nb[:, :band] = True
    nb[:, -band:] = True

    seed = (pink | green | skin | wine) & ~nb
    allow = (seed | suit | whiteish) & ~nb
    recon = seed.astype(np.uint8) * 255
    allow_u = allow.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    for _ in range(150):
        nxt = cv2.bitwise_and(cv2.dilate(recon, k, iterations=1), allow_u)
        if np.array_equal(nxt, recon):
            break
        recon = nxt
    near = cv2.dilate(recon, np.ones((5, 5), np.uint8), iterations=2) > 0
    recon[((dark & near) & ~nb).astype(np.uint8) > 0] = 255
    recon = _largest_component(recon)
    recon = cv2.morphologyEx(recon, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    inv = cv2.bitwise_not(recon)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv)
    for cid in range(1, n):
        x, y, ww, hh, area = st[cid]
        if area < h * w * 0.08 and x > 0 and y > 0 and x + ww < w and y + hh < h:
            recon[lab == cid] = 255
    recon[nb] = 0
    return recon


def prepare_subject(
    bgr: np.ndarray,
    *,
    mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    返回 (白底主体图, mask 0/255, meta)。
    mode: auto | off | rembg | contrast | white
    """
    if mode == "off":
        full = np.full(bgr.shape[:2], 255, dtype=np.uint8)
        return bgr.copy(), full, {"mode": "off", "applied": False, **border_stats(bgr)}

    chosen = detect_bg_mode(bgr) if mode == "auto" else mode
    if chosen == "skip" or (mode == "auto" and chosen == "skip"):
        return bgr.copy(), mask_near_white(bgr), {
            "mode": "auto",
            "applied": False,
            "chosen": "skip",
            **border_stats(bgr),
        }

    method = "unknown"
    if chosen == "white":
        mask = mask_near_white(bgr)
        method = "white"
    else:
        mask = None
        if mode in {"auto", "rembg", "contrast", "grabcut"}:
            mask = mask_rembg(bgr)
            if mask is not None:
                method = "rembg"
        if mask is None:
            mask = mask_illustration_recon(bgr)
            method = "illustration_recon"

    white = np.full_like(bgr, 255)
    white[mask > 0] = bgr[mask > 0]
    frac = float(np.mean(mask > 0))
    if frac < 0.015 or frac > 0.95:
        return bgr.copy(), mask_near_white(bgr), {
            "mode": mode,
            "applied": False,
            "chosen": chosen,
            "method": method,
            "fallback": "mask_fraction_out_of_range",
            "fg_frac": frac,
            **border_stats(bgr),
        }
    return white, mask, {
        "mode": mode,
        "applied": True,
        "chosen": chosen,
        "method": method,
        "fg_frac": frac,
        **border_stats(bgr),
    }
