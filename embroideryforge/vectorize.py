"""位图 → SVG 矢量旁路。

优先 VTracer（子进程，避免原生崩溃拖垮主进程）；失败则 OpenCV 轮廓 SVG 回退。
不替代 stitch-auto 主链。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _opencv_contour_svg(image_path: Path, svg_path: Path, *, max_colors: int = 8) -> dict[str, Any]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]
    # 简单量化 + 每色轮廓
    small = cv2.resize(bgr, (min(w, 400), min(h, 400)), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    k = min(max_colors, max(2, len(np.unique(data, axis=0)) // 20 or 2))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    centers = centers.astype(np.uint8)
    scale_x = w / small.shape[1]
    scale_y = h / small.shape[0]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    for i, center in enumerate(centers):
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hex_c = "#{:02x}{:02x}{:02x}".format(int(center[2]), int(center[1]), int(center[0]))
        for cnt in contours:
            if cv2.contourArea(cnt) < 40:
                continue
            pts = cnt.reshape(-1, 2)
            d = "M " + " L ".join(f"{p[0]*scale_x:.1f} {p[1]*scale_y:.1f}" for p in pts) + " Z"
            parts.append(f'<path d="{d}" fill="{hex_c}" fill-opacity="0.85" stroke="none"/>')
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")
    return {"engine": "opencv_contours", "colors": int(k)}


def _vtracer_subprocess(image_path: Path, svg_path: Path) -> None:
    code = (
        "import vtracer,sys;"
        f"vtracer.convert_image_to_svg_py(r'{image_path}', r'{svg_path}', "
        "colormode='color', hierarchical='stacked', mode='spline', "
        "filter_speckle=4, color_precision=6, layer_difference=16, "
        "corner_threshold=60, length_threshold=4.0, max_iterations=10, "
        "splice_threshold=45, path_precision=3)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0 or not svg_path.is_file() or svg_path.stat().st_size < 50:
        err = (proc.stderr or proc.stdout or f"exit={proc.returncode}")[:500]
        raise RuntimeError(f"vtracer failed: {err}")


def vectorize_image(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    prefer_vtracer: bool = True,
) -> dict[str, Any]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    svg_path = out_dir / "vector.svg"
    engine_meta: dict[str, Any] = {}
    notes: list[str] = []

    if prefer_vtracer:
        try:
            import vtracer  # noqa: F401

            _vtracer_subprocess(image_path, svg_path)
            engine_meta = {"engine": "vtracer"}
        except Exception as exc:  # noqa: BLE001
            notes.append(f"vtracer unavailable or crashed: {exc}")
            engine_meta = _opencv_contour_svg(image_path, svg_path)
            notes.append("fell back to opencv_contours")
    else:
        engine_meta = _opencv_contour_svg(image_path, svg_path)

    meta = {
        "ok": svg_path.is_file() and svg_path.stat().st_size > 50,
        "source": image_path.name,
        "svg": svg_path.name,
        "bytes": svg_path.stat().st_size if svg_path.is_file() else 0,
        **engine_meta,
        "notes": notes,
        "note": "矢量旁路；DST 仍走 stitch-auto 栅格分区主链",
    }
    (out_dir / "vector_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
