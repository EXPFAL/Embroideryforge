"""动物 A2 spike：单连通域密度+方向场 → 流线 → 可缝 DST。

锚点：Liu et al. 2023 CGF / embroidery-streamlines 思想（α,v → 流线 → 连通路径）。
不替换 V6.6 dens；仅显式 --streamlines-patch 时走本模块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyembroidery import EmbPattern, write_dst, write_png

from embroideryforge.autofill import _hex_to_rgb_name, _to_emb_xy
from embroideryforge.bg import prepare_subject
from embroideryforge.preview import PREVIEW_BG_DARK, PREVIEW_BG_WHITE, render_clean_preview


def _bgr_to_hex(bgr: np.ndarray | list[int] | tuple[int, ...]) -> str:
    b, g, r = (int(x) for x in list(bgr)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def largest_component_mask(mask: np.ndarray, *, min_area: int = 80) -> np.ndarray:
    """取最大连通域，保证单连通斑块。"""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if int(stats[idx, cv2.CC_STAT_AREA]) < int(min_area):
        return m * 255
    out = np.zeros_like(m)
    out[labels == idx] = 255
    return out


def structure_tensor_field(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回单位方向场 (vx, vy) 与相干度 [0,1]；mask 外为零。"""
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), max(0.6, float(sigma)))
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    # 结构张量：梯度方向 θ，流线取切向 θ+π/2（沿边缘/毛向，不穿出轮廓）
    ixx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    iyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    ixy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    tmp = np.sqrt((ixx - iyy) ** 2 + 4.0 * ixy * ixy) + 1e-8
    ang = 0.5 * np.arctan2(2.0 * ixy, ixx - iyy)
    vx = (-np.sin(ang)).astype(np.float32)
    vy = (np.cos(ang)).astype(np.float32)
    vx = cv2.GaussianBlur(vx, (0, 0), 2.2)
    vy = cv2.GaussianBlur(vy, (0, 0), 2.2)
    coh = tmp / (ixx + iyy + 1e-8)
    coh = np.clip(coh, 0.0, 1.0)
    nrm = np.sqrt(vx * vx + vy * vy) + 1e-8
    vx = vx / nrm
    vy = vy / nrm
    m = mask > 0
    vx = np.where(m, vx, 0.0).astype(np.float32)
    vy = np.where(m, vy, 0.0).astype(np.float32)
    coh = np.where(m, coh, 0.0).astype(np.float32)
    return vx, vy, coh


def density_from_gray(gray: np.ndarray, mask: np.ndarray, *, gamma: float = 0.85) -> np.ndarray:
    inv = (255.0 - gray.astype(np.float32)) / 255.0
    dens = np.clip(inv**gamma, 0.08, 1.0)
    return np.where(mask > 0, dens, 0.0).astype(np.float32)


def _sample_field(field: np.ndarray, x: float, y: float) -> float:
    h, w = field.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return 0.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx, ty = float(x - x0), float(y - y0)
    v00 = float(field[y0, x0])
    v01 = float(field[y0, x1])
    v10 = float(field[y1, x0])
    v11 = float(field[y1, x1])
    return (1 - ty) * ((1 - tx) * v00 + tx * v01) + ty * ((1 - tx) * v10 + tx * v11)


def _in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0


def _trace_one(
    x0: float,
    y0: float,
    vx: np.ndarray,
    vy: np.ndarray,
    mask: np.ndarray,
    *,
    step: float,
    max_len: int,
) -> list[tuple[float, float]]:
    """沿方向场双向积分；积分中只检查 mask（间距由种子/占位控制）。"""

    def integrate(sign: float) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        x, y = float(x0), float(y0)
        px, py = 0.0, 0.0
        for _ in range(max_len):
            if not _in_mask(mask, x, y):
                break
            pts.append((x, y))
            vxx = _sample_field(vx, x, y)
            vyy = _sample_field(vy, x, y)
            n = (vxx * vxx + vyy * vyy) ** 0.5
            if n < 1e-4:
                if px * px + py * py < 1e-8:
                    break
                vxx, vyy = px, py
            else:
                vxx, vyy = vxx / n, vyy / n
                if px * vxx + py * vyy < 0:
                    vxx, vyy = -vxx, -vyy
            px, py = vxx, vyy
            x += sign * vxx * step
            y += sign * vyy * step
        return pts

    fwd = integrate(1.0)
    bwd = integrate(-1.0)
    if bwd:
        bwd = list(reversed(bwd[:-1])) if fwd else list(reversed(bwd))
    line = bwd + fwd
    return line if len(line) >= 3 else []


def _mark_occupied(
    occupied: np.ndarray,
    dens: np.ndarray,
    pts: list[tuple[float, float]],
    *,
    sep_base: float,
) -> None:
    """沿流线画细带；暗区（密）带宽更窄。"""
    h, w = occupied.shape
    for x, y in pts[::2]:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        d = float(dens[yi, xi])
        # 带宽上限 2：避免暗区仍留大片空白
        rad = max(1, min(2, int(round(sep_base / max(d, 0.35)))))
        cv2.circle(occupied, (xi, yi), rad, 255, -1)


def generate_streamlines(
    dens: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    mask: np.ndarray,
    *,
    sep_base: float = 1.35,
    step: float = 0.55,
    max_len: int = 1200,
    max_lines: int = 18000,
) -> list[list[tuple[float, float]]]:
    """密铺流线：网格+密度加权种子；占位只约束种子间距。"""
    h, w = mask.shape
    occupied = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    # 1) 规则网格种子（保证覆盖）；sep<1 时按 1px 扫
    seeds: list[tuple[float, float]] = []
    grid = max(1, int(np.ceil(sep_base)))
    for y in range(0, h, grid):
        for x in range(0, w, grid):
            if mask[y, x] == 0:
                continue
            seeds.append((float(x) + 0.5, float(y) + 0.5))

    # 2) 再补密度加权随机种子填缝
    weights = dens[ys, xs].astype(np.float64)
    weights = weights / (weights.sum() + 1e-8)
    n_extra = min(len(xs), max(max_lines, len(seeds) * 2))
    rng = np.random.default_rng(42)
    picks = rng.choice(len(xs), size=n_extra, replace=False, p=weights)
    for i in picks:
        seeds.append((float(xs[i]), float(ys[i])))

    lines: list[list[tuple[float, float]]] = []
    for x0, y0 in seeds:
        if len(lines) >= max_lines:
            break
        xi, yi = int(round(x0)), int(round(y0))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        if occupied[yi, xi]:
            continue
        line = _trace_one(x0, y0, vx, vy, mask, step=step, max_len=max_len)
        if len(line) < 4:
            continue
        _mark_occupied(occupied, dens, line, sep_base=sep_base)
        lines.append(line)
    return lines


def _expand_parallel_offsets(
    lines: list[list[tuple[float, float]]],
    mask: np.ndarray,
    offsets: tuple[float, ...] = (-1.0, 1.0),
) -> list[list[tuple[float, float]]]:
    """沿法向复制平行流线，提高 thick 覆盖率。"""
    if not lines:
        return lines
    h, w = mask.shape
    out: list[list[tuple[float, float]]] = list(lines)
    for poly in lines:
        if len(poly) < 2:
            continue
        for off in offsets:
            shifted: list[tuple[float, float]] = []
            for i, (x, y) in enumerate(poly):
                if i == 0:
                    dx, dy = poly[1][0] - x, poly[1][1] - y
                else:
                    dx, dy = x - poly[i - 1][0], y - poly[i - 1][1]
                n = (dx * dx + dy * dy) ** 0.5
                if n < 1e-6:
                    continue
                nx, ny = -dy / n, dx / n
                xx, yy = x + off * nx, y + off * ny
                xi, yi = int(round(xx)), int(round(yy))
                if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
                    shifted.append((xx, yy))
            if len(shifted) >= 4:
                out.append(shifted)
    return out


def _resample_poly(pts: list[tuple[float, float]], stitch_len: float) -> list[tuple[float, float]]:
    """按弧长重采样。步长小于 stitch_len 时也要跨段累计，不能收成首尾一根弦。"""
    if len(pts) < 2:
        return pts
    out: list[tuple[float, float]] = [pts[0]]
    remain = float(stitch_len)
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dx, dy = x1 - x0, y1 - y0
        seg = (dx * dx + dy * dy) ** 0.5
        if seg < 1e-6:
            continue
        walked = 0.0
        while walked + remain <= seg + 1e-6:
            walked += remain
            t = min(1.0, walked / seg)
            out.append((x0 + t * dx, y0 + t * dy))
            remain = float(stitch_len)
        remain -= seg - walked
        if remain < 1e-4:
            remain = float(stitch_len)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _connect_lines_greedy(
    lines: list[list[tuple[float, float]]],
    *,
    max_jump: float = 8.0,
) -> list[list[tuple[float, float]]]:
    """贪心把端点靠近的流线串成更长路径（近似 MST 连通）。"""
    if not lines:
        return []
    remaining = [list(L) for L in lines]
    # 按长度降序作种子
    remaining.sort(key=len, reverse=True)
    chains: list[list[tuple[float, float]]] = []
    while remaining:
        cur = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            best_i = -1
            best_d = max_jump
            best_mode = ""
            for i, other in enumerate(remaining):
                for mode, a, b in (
                    ("tt", cur[-1], other[0]),
                    ("tr", cur[-1], other[-1]),
                    ("ht", cur[0], other[0]),
                    ("hr", cur[0], other[-1]),
                ):
                    d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_i = i
                        best_mode = mode
            if best_i < 0:
                break
            other = remaining.pop(best_i)
            if best_mode == "tt":
                cur = cur + other
            elif best_mode == "tr":
                cur = cur + list(reversed(other))
            elif best_mode == "ht":
                cur = list(reversed(other)) + cur
            else:
                cur = other + cur
            changed = True
        chains.append(cur)
    return chains


def _quantize_line_colors(
    bgr: np.ndarray,
    lines: list[list[tuple[float, float]]],
    n_colors: int,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """每条链取中位色，kmeans 合并到 n_colors，返回 (hex, poly) 列表（深色优先）。"""
    if not lines:
        return []
    samples = []
    for poly in lines:
        cols = []
        stride = max(1, len(poly) // 12)
        for x, y in poly[::stride]:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < bgr.shape[0] and 0 <= xi < bgr.shape[1]:
                cols.append(bgr[yi, xi].astype(np.float32))
        if not cols:
            samples.append(np.array([128, 128, 128], dtype=np.float32))
        else:
            samples.append(np.median(np.stack(cols, axis=0), axis=0))
    data = np.stack(samples, axis=0).astype(np.float32)
    k = int(np.clip(n_colors, 2, min(12, len(lines))))
    if len(lines) <= k:
        labels = np.arange(len(lines))
        centers = data
    else:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        _, labels, centers = cv2.kmeans(data, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1)
    order = np.argsort([float(np.linalg.norm(c)) for c in centers])
    colored: list[tuple[str, list[tuple[float, float]]]] = []
    for ci in order:
        polys = [lines[i] for i in range(len(lines)) if int(labels[i]) == int(ci)]
        if not polys:
            continue
        hx = _bgr_to_hex(centers[int(ci)])
        # 不把流线端点硬缝成一笔：跨空隙的实缝会在 thick 上变成稀疏长线
        for poly in polys:
            if len(poly) >= 2:
                colored.append((hx, poly))
    return colored


def _render_stream_preview(
    bgr: np.ndarray,
    mask: np.ndarray,
    lines: list[list[tuple[float, float]]],
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    vis = bgr.copy()
    overlay = vis.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.35).astype(np.uint8)
    for poly in lines:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(overlay, [pts], False, (40, 220, 255), 1, cv2.LINE_AA)
    # 方向小箭头采样
    step = max(8, min(bgr.shape[:2]) // 40)
    for y in range(step // 2, bgr.shape[0], step):
        for x in range(step // 2, bgr.shape[1], step):
            if mask[y, x] == 0:
                continue
            dx, dy = float(vx[y, x]) * 6, float(vy[y, x]) * 6
            cv2.arrowedLine(
                overlay,
                (x, y),
                (int(x + dx), int(y + dy)),
                (80, 180, 80),
                1,
                tipLength=0.35,
            )
    return overlay


def run_streamlines_patch(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    n_colors: int = 8,
    resolution: str = "high",
    target_height_mm: float = 70.0,
    bg_mode: str = "keep",
    sep_base: float = 1.35,
    direction_weight: float = 1.0,
) -> dict[str, Any]:
    """单连通域流线绣 spike。mask 缺省时用主体最大连通域。

    返回形状对齐 dens/photo_stitch CLI：ok/dst/out_dir/report/meta。
    """
    from embroideryforge import photo_stitch as ps

    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise FileNotFoundError(image_path)

    subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    cv2.imwrite(str(out_dir / "subject.png"), subject_bgr)

    if mask_path is not None:
        raw_m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_m is None:
            raise FileNotFoundError(mask_path)
        if raw_m.shape[:2] != subject_bgr.shape[:2]:
            raw_m = cv2.resize(
                raw_m, (subject_bgr.shape[1], subject_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        patch_mask = largest_component_mask(raw_m)
        mask_source = f"file:{Path(mask_path).name}"
    else:
        base = subject_mask if subject_mask is not None else bg_meta.get("subject_mask")
        if base is None:
            base = np.full(subject_bgr.shape[:2], 255, dtype=np.uint8)
        patch_mask = largest_component_mask(base)
        mask_source = "largest_cc(subject)"

    cv2.imwrite(str(out_dir / "patch_mask.png"), patch_mask)
    cv2.imwrite(str(out_dir / "subject_mask.png"), subject_mask)

    cfg = dict(ps.RESOLUTION.get(resolution, ps.RESOLUTION["high"]))
    max_side = max(int(cfg["max_side"]), 720)
    work, _ = ps.resize_for_resolution(subject_bgr, resolution, max_side=max_side)
    work_m = cv2.resize(
        patch_mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    work_m = largest_component_mask(work_m, min_area=40)
    # 轻闭运算，填毛发孔
    work_m = cv2.morphologyEx(work_m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    dens = density_from_gray(gray, work_m)
    vx, vy, coh = structure_tensor_field(gray, work_m)
    # direction_weight：低相干处略混合水平偏向，避免噪声场
    dw = float(np.clip(direction_weight, 0.0, 1.0))
    if dw < 1.0:
        mix = (1.0 - dw) * (1.0 - coh)
        vx = vx * (1.0 - mix) + 1.0 * mix
        vy = vy * (1.0 - mix)
        nrm = np.sqrt(vx * vx + vy * vy) + 1e-8
        vx, vy = vx / nrm, vy / nrm

    # 沿切向走长流线；每条单独 JUMP，thick 才是毛向填面而不是跨区长针
    stitch_len = min(float(cfg.get("stitch_len", 1.9)), 1.15)
    lines = generate_streamlines(
        dens,
        vx,
        vy,
        work_m,
        sep_base=float(sep_base),
        step=0.85,
        max_len=900,
        max_lines=12000,
    )
    lines = [_resample_poly(L, stitch_len) for L in lines if len(L) >= 6]
    colored = _quantize_line_colors(work, lines, n_colors=int(n_colors))

    stream_prev = _render_stream_preview(work, work_m, lines, vx, vy)
    cv2.imwrite(str(out_dir / "streamlines_preview.png"), stream_prev)
    dens_u8 = (np.clip(dens, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "density_map.png"), dens_u8)
    # 方向可视化
    ang = (np.arctan2(vy, vx) + np.pi) / (2 * np.pi)
    hsv = np.zeros((*work_m.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 179).astype(np.uint8)
    hsv[..., 1] = np.where(work_m > 0, 200, 0).astype(np.uint8)
    hsv[..., 2] = np.where(work_m > 0, 220, 30).astype(np.uint8)
    cv2.imwrite(str(out_dir / "direction_field.png"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    h, _w = work.shape[:2]
    scale = (float(target_height_mm) / max(h, 1)) * 10.0
    pattern = EmbPattern()
    first = True
    n_pts = 0
    last_hex: str | None = None
    for hx, poly in colored:
        if len(poly) < 2:
            continue
        if first or hx != last_hex:
            if not first:
                pattern.color_change()
            pattern.add_thread(_hex_to_rgb_name(hx))
            first = False
            last_hex = hx
        block = [_to_emb_xy(x, y, scale, h) for x, y in poly]
        pattern.move_abs(block[0][0], block[0][1])
        for x, y in block:
            pattern.stitch_abs(x, y)
            n_pts += 1

    dst_path = out_dir / "auto.dst"
    write_dst(pattern, str(dst_path))
    write_png(pattern, str(out_dir / "auto_pyembroidery.png"))
    render_clean_preview(
        pattern, out_dir / "auto_clean_preview.png", background=PREVIEW_BG_DARK
    )
    render_clean_preview(
        pattern, out_dir / "auto_stitch_thick.png", background=PREVIEW_BG_DARK, thick=True
    )
    render_clean_preview(
        pattern,
        out_dir / "auto_stitch_thick_white.png",
        background=PREVIEW_BG_WHITE,
        thick=True,
    )

    # compare: subject | stream preview | thick
    sub_s = work.copy()
    thick = cv2.imread(str(out_dir / "auto_stitch_thick.png"), cv2.IMREAD_COLOR)
    if thick is None:
        thick = np.zeros_like(sub_s)
    else:
        thick = cv2.resize(thick, (sub_s.shape[1], sub_s.shape[0]))
    stream_r = cv2.resize(stream_prev, (sub_s.shape[1], sub_s.shape[0]))
    compare = np.hstack([sub_s, stream_r, thick])
    cv2.imwrite(str(out_dir / "compare_preview.png"), compare)

    stitches = int(pattern.count_stitches())
    report = {
        "stitches": stitches,
        "color_changes": int(pattern.count_color_changes()),
        "engine": "streamlines_patch_a2",
        "target_height_mm": float(target_height_mm),
    }
    meta = {
        "engine": "streamlines_patch_a2",
        "mode": "streamlines_patch",
        "a2_streamlines": True,
        "domain": "animal",
        "source_image": str(image_path),
        "mask_source": mask_source,
        "n_colors": int(n_colors),
        "resolution": resolution,
        "target_height_mm": float(target_height_mm),
        "work_size": [int(work.shape[1]), int(work.shape[0])],
        "sep_base": float(sep_base),
        "direction_weight": float(direction_weight),
        "n_streamlines": len(lines),
        "mean_line_pts": float(np.mean([len(L) for L in lines])) if lines else 0.0,
        "n_chains": len(colored),
        "stitches": stitches,
        "stitch_points": n_pts,
        "color_changes": report["color_changes"],
        "dst": dst_path.name,
        "palette": sorted({hx for hx, _ in colored}),
        "layers": [{"tag": f"stream_{i}", "hex": hx} for i, (hx, _) in enumerate(colored[:32])],
        "preview": {
            "streamlines": "streamlines_preview.png",
            "direction": "direction_field.png",
            "stitch_thick": "auto_stitch_thick.png",
            "compare": "compare_preview.png",
            "main_thumb": "auto_stitch_thick.png",
        },
        "artifacts": {
            "patch_mask": "patch_mask.png",
            "streamlines_preview": "streamlines_preview.png",
            "direction_field": "direction_field.png",
            "compare_preview": "compare_preview.png",
            "auto_stitch_thick": "auto_stitch_thick.png",
        },
        "v6": {
            "a2_streamlines": True,
            "note": "CGF-inspired single-patch streamlines; dens path untouched",
        },
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps({"ok": True, **report}, indent=2),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "dst": dst_path,
        "out_dir": out_dir,
        "report": report,
        "meta": meta,
    }
