from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class ColorInfo:
    bgr: list[int]
    hex: str
    ratio: float


@dataclass
class AnalyzeResult:
    width: int
    height: int
    colors: list[ColorInfo]
    external_contours: int
    outline_points: list[list[int]]
    overlay_bgr: np.ndarray
    swatch_bgr: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "external_contours": self.external_contours,
            "outline_point_count": len(self.outline_points),
            "colors": [asdict(c) for c in self.colors],
        }


def _dominant_colors(bgr: np.ndarray, k: int = 4) -> list[ColorInfo]:
    pixels = bgr.reshape(-1, 3).astype(np.float32)
    mask = np.any(pixels < 245, axis=1)
    data = pixels[mask] if mask.sum() >= k else pixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    total = int(counts.sum()) or 1
    out: list[ColorInfo] = []
    for i in order:
        b, g, r = map(int, centers[i])
        out.append(
            ColorInfo(
                bgr=[b, g, r],
                hex=f"#{r:02x}{g:02x}{b:02x}",
                ratio=round(float(counts[i]) / total, 4),
            )
        )
    return out


def analyze_image(path: str | Path, color_k: int = 4) -> AnalyzeResult:
    path = Path(path)
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    h, w = bgr.shape[:2]
    colors = _dominant_colors(bgr, k=color_k)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    overlay = bgr.copy()
    cv2.drawContours(overlay, contours[:8], -1, (0, 0, 255), 2)

    outline: list[list[int]] = []
    if contours:
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.005 * peri, True)
        outline = [[int(p[0][0]), int(p[0][1])] for p in approx]

    swatch = np.zeros((80, max(400, color_k * 100), 3), dtype=np.uint8)
    for i, c in enumerate(colors[:color_k]):
        x0, x1 = i * 100, (i + 1) * 100
        swatch[:, x0:x1] = c.bgr

    return AnalyzeResult(
        width=w,
        height=h,
        colors=colors,
        external_contours=len(contours),
        outline_points=outline,
        overlay_bgr=overlay,
        swatch_bgr=swatch,
    )


def suggest_stitch_plan(result: AnalyzeResult) -> list[dict[str, Any]]:
    """无 API 时的确定性针法建议，供 UI/说明使用；后续可替换为 LLM。"""
    plan: list[dict[str, Any]] = []
    for i, c in enumerate(result.colors):
        # 近白跳过填针建议
        if min(c.bgr) > 230:
            continue
        role = "fill" if c.ratio >= 0.15 else "detail"
        stitch = "tatami" if role == "fill" else "satin"
        plan.append(
            {
                "region_id": f"c{i}",
                "hex": c.hex,
                "ratio": c.ratio,
                "role": role,
                "suggested_stitch": stitch,
                "density_hint": "medium" if role == "fill" else "fine",
            }
        )
    if result.outline_points:
        plan.insert(
            0,
            {
                "region_id": "outline",
                "hex": "#000000",
                "ratio": None,
                "role": "outline",
                "suggested_stitch": "running",
                "density_hint": "contour",
            },
        )
    return plan
