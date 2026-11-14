"""人像 Photo-Stitch P0：肤色向线色表贴色 + Reserve 角色 + 灰垫底序。

对标 Brother Chart (Photo Stitch) / Hatch 少色写真流程的可吸收部分：
真线色近似、锁肤唇眼、低 chroma 灰先绣。不实现 PS1 叠层混色。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# code, role, BGR — 精简「写真肤色向」表（近似 Brother/Madeira 常用档，非官方编号）
# role: skin | lips | eye | hair | grey | accent | other
FLESH_PHOTO_CHART: list[tuple[str, str, tuple[int, int, int]]] = [
    # greys / underpaint
    ("PS-K1", "grey", (28, 28, 28)),
    ("PS-K2", "grey", (48, 48, 50)),
    ("PS-K3", "grey", (72, 72, 74)),
    ("PS-K4", "grey", (98, 98, 100)),
    ("PS-K5", "grey", (128, 128, 130)),
    ("PS-K6", "grey", (168, 168, 170)),
    ("PS-K7", "grey", (210, 210, 212)),
    ("PS-W1", "grey", (236, 236, 238)),
    # skin ramp (peach → tan)
    ("PS-S1", "skin", (210, 220, 242)),
    ("PS-S2", "skin", (190, 205, 232)),
    ("PS-S3", "skin", (170, 188, 220)),
    ("PS-S4", "skin", (150, 170, 205)),
    ("PS-S5", "skin", (130, 155, 195)),
    ("PS-S6", "skin", (110, 140, 180)),
    ("PS-S7", "skin", (95, 125, 165)),
    ("PS-S8", "skin", (80, 105, 145)),
    ("PS-S9", "skin", (70, 95, 130)),
    ("PS-S10", "skin", (55, 80, 110)),
    # lips
    ("PS-L1", "lips", (90, 80, 170)),
    ("PS-L2", "lips", (75, 70, 150)),
    ("PS-L3", "lips", (60, 55, 130)),
    ("PS-L4", "lips", (100, 90, 190)),
    # eyes / dark iris-ish
    ("PS-E1", "eye", (40, 45, 55)),
    ("PS-E2", "eye", (55, 70, 90)),
    ("PS-E3", "eye", (70, 90, 50)),
    ("PS-E4", "eye", (90, 100, 40)),
    # hair
    ("PS-H1", "hair", (25, 22, 20)),
    ("PS-H2", "hair", (40, 35, 30)),
    ("PS-H3", "hair", (55, 48, 40)),
    ("PS-H4", "hair", (75, 70, 55)),
    ("PS-H5", "hair", (100, 110, 140)),
    ("PS-H6", "hair", (45, 55, 90)),
    # accents / common garments
    ("PS-A1", "accent", (40, 50, 180)),  # red
    ("PS-A2", "accent", (50, 140, 50)),  # green
    ("PS-A3", "accent", (160, 90, 40)),  # blue
    ("PS-A4", "accent", (40, 180, 220)),  # yellow
    ("PS-A5", "accent", (90, 90, 40)),  # teal
    ("PS-A6", "other", (70, 100, 140)),  # khaki
    ("PS-A7", "other", (100, 80, 60)),  # denim-ish
    ("PS-A8", "other", (60, 90, 120)),  # brown
]


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    c = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)


def delta_e76(lab1: np.ndarray, lab2: np.ndarray, *, ab_weight: float = 2.0) -> float:
    """加权 Lab 距离（写真贴线够用；避免再引依赖）。"""
    d = lab1.astype(np.float64) - lab2.astype(np.float64)
    d[1] *= float(ab_weight)
    d[2] *= float(ab_weight)
    return float(np.sqrt(np.dot(d, d)))


def chart_labs() -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """(code, role, bgr_u8, lab)."""
    out: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for code, role, bgr in FLESH_PHOTO_CHART:
        b = np.asarray(bgr, dtype=np.uint8)
        out.append((code, role, b, _bgr_to_lab(b)))
    return out


def snap_bgr_to_chart(
    bgr: np.ndarray,
    *,
    ab_weight: float = 2.05,
) -> dict[str, Any]:
    """最近邻贴到写真线表。"""
    lab = _bgr_to_lab(bgr)
    best = None
    best_d = 1e18
    for code, role, cb, clab in chart_labs():
        d = delta_e76(lab, clab, ab_weight=ab_weight)
        if d < best_d:
            best_d = d
            best = (code, role, cb, d)
    assert best is not None
    code, role, cb, d = best
    return {
        "code": code,
        "role": role,
        "bgr": cb.copy(),
        "delta_e": round(float(d), 3),
        "src_bgr": [int(x) for x in np.asarray(bgr).reshape(3)],
    }


def snap_centers_to_chart(
    centers: list[np.ndarray],
    *,
    ab_weight: float = 2.05,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """批量贴色；同 code 合并由调用方决定（此处每心独立贴）。"""
    out: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for c in centers:
        m = snap_bgr_to_chart(c, ab_weight=ab_weight)
        out.append(np.asarray(m["bgr"], dtype=np.uint8).reshape(3))
        meta.append(m)
    return out, meta


def classify_center_role(bgr: np.ndarray) -> str:
    """HSV 粗分角色（Reserve / 垫底序用）。"""
    b, g, r = (int(x) for x in np.asarray(bgr).reshape(3))
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = (int(x) for x in hsv)
    a = float(_bgr_to_lab([b, g, r])[1] - 128.0)
    bb = float(_bgr_to_lab([b, g, r])[2] - 128.0)
    chroma = float(np.sqrt(a * a + bb * bb))
    if chroma < 14.0 and s <= 40:
        return "grey"
    if (h <= 8 or h >= 170) and s >= 70 and 80 <= v <= 210:
        return "lips"
    if 3 <= h <= 35 and 8 <= s <= 95 and v >= 100:
        return "skin"
    if v <= 55 and s <= 70:
        return "eye" if chroma < 25 else "hair"
    if v <= 95 and s <= 85:
        return "hair"
    if s >= 70:
        return "accent"
    return "other"


def chroma_lab_bgr(bgr: np.ndarray) -> float:
    lab = _bgr_to_lab(bgr)
    a = float(lab[1] - 128.0)
    b = float(lab[2] - 128.0)
    return float(np.sqrt(a * a + b * b))


def luminance_bgr(bgr: np.ndarray) -> float:
    c = np.asarray(bgr, dtype=np.float64).reshape(3)
    return float(0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2])


def sew_order_indices(centers: list[np.ndarray], *, grey_underpaint: bool = True) -> list[int]:
    """低 chroma 灰先绣（垫底），再按亮度从暗到亮。"""

    def key(i: int) -> tuple[int, float]:
        c = centers[i]
        ch = chroma_lab_bgr(c)
        lum = luminance_bgr(c)
        if grey_underpaint and (ch < 16.0 or classify_center_role(c) == "grey"):
            return (0, lum)
        return (1, lum)

    return sorted(range(len(centers)), key=key)


def sample_reserve_bgr(
    bgr: np.ndarray,
    face_mask: np.ndarray | None,
    *,
    max_each: int = 2,
) -> list[np.ndarray]:
    """从脸罩采样肤/唇 Reserve 色（BGR uint8）。"""
    if face_mask is None or not np.any(face_mask) or bgr is None:
        return []
    fm = face_mask > 0
    if int(np.count_nonzero(fm)) < 80:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ys, xs = np.where(fm)
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(1, y1 - y0)

    skin = fm & (h >= 3) & (h <= 35) & (s >= 8) & (s <= 100) & (v >= 100)
    yy = np.arange(bgr.shape[0], dtype=np.int32)[:, None]
    lip = fm & ((h <= 8) | (h >= 170)) & (s >= 70) & (v >= 80) & (v <= 210)
    lip = lip & (yy >= (y0 + int(0.45 * span)))

    out: list[np.ndarray] = []
    for mask, _role in ((skin, "skin"), (lip, "lips")):
        if int(np.count_nonzero(mask)) < 40:
            continue
        pix = bgr[mask]
        # 取 1–2 个代表：中位 + 略深分位
        med = np.median(pix, axis=0).astype(np.uint8)
        out.append(med)
        if max_each >= 2 and len(pix) >= 80:
            dark = np.percentile(pix.astype(np.float32), 30, axis=0).astype(np.uint8)
            if float(np.linalg.norm(dark.astype(np.float32) - med.astype(np.float32))) >= 12:
                out.append(dark)
    # eye: upper face dark low-sat
    upper = fm & (yy <= (y0 + int(0.55 * span))) & (v <= 70) & (s <= 60)
    if int(np.count_nonzero(upper)) >= 30:
        out.append(np.median(bgr[upper], axis=0).astype(np.uint8))
    return out


def protect_mask_for_centers(
    centers_lab: np.ndarray,
    reserve_bgr: list[np.ndarray],
    *,
    max_de: float = 22.0,
    ab_weight: float = 2.05,
) -> np.ndarray:
    """centers_lab (N,3) → bool protected by proximity to reserve colors."""
    n = len(centers_lab)
    prot = np.zeros(n, dtype=bool)
    if not reserve_bgr:
        return prot
    rlabs = [_bgr_to_lab(r) for r in reserve_bgr]
    for i in range(n):
        for rl in rlabs:
            if delta_e76(centers_lab[i], rl, ab_weight=ab_weight) <= max_de:
                prot[i] = True
                break
    return prot
