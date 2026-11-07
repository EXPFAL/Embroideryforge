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

from embroideryforge.autofill import _hex_to_rgb_name, _tatami_polylines, _to_emb_xy
from embroideryforge.bg import prepare_subject
from embroideryforge.preview import render_clean_preview
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
    return keep.astype(np.uint8) * 255


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
) -> tuple[EmbPattern, list[dict[str, Any]], bool, dict[str, np.ndarray]]:
    ps = _ps()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tone, label_map, centers = ps.quantize_tones(bgr, mask, n_colors=n_colors, pre_smooth=True)
    dens = density_map(gray, mask)
    core = feature_core_mask(gray, mask, percentile=22.0)
    dens_use = dens.copy()
    face = mask > 0
    # 中调微抬：恢复 v4 点阵层次；核区再抬一档
    mid = face & (dens >= 0.28) & (dens <= 0.82)
    dens_use[mid] = np.minimum(1.0, dens_use[mid] * 1.1 + 0.05)
    if np.any(core):
        boost = core > 0
        dens_use[boost] = np.minimum(1.0, dens_use[boost] * 1.3 + 0.14)
    face_step = max(1.0, float(step) * 0.85)
    keep = sample_keep_grid(
        dither_keep_mask(dens_use, mask),
        dens_use,
        mask,
        face_step,
        core_mask=core,
        core_step_scale=0.45,
    )
    h = mask.shape[0]
    scale = (target_height_mm / max(h, 1)) * 10.0
    step_i = max(1, int(round(face_step)))
    # 串针只并「邻接点」：过宽 connect 会在空白处拉线，抹掉 v4 的点阵细节
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
    maps = {"dens": dens, "keep": keep, "tone": tone, "core": core}
    return pattern, layers, first, maps


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
) -> tuple[EmbPattern, list[dict[str, Any]], bool]:
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
    for tag in ps.BODY_TAGS:
        if tag not in label_tags:
            continue
        tid = label_tags.index(tag)
        body = ((lab_r == tid) & (work_mask > 0) & (face_dil == 0)).astype(np.uint8) * 255
        body = ps.clean_layer_mask(body, min_area=50, harden=False)
        if int(np.count_nonzero(body)) < 80:
            continue
        hex_c = hex_by_tag.get(tag, "#808080")
        angle = {"clothes": 0.0, "hat": 32.0, "hair": 58.0}.get(tag, 0.0)
        polys = ps.filter_short_polylines(
            _tatami_polylines(body, angle_deg=angle, spacing=spacing * 1.1, stitch_len=stitch_len * 1.15),
            min_span=2.5,
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

    roi_src = ps.detect_face_roi(subject_bgr) or ps._default_bust_roi(subject_bgr)
    boxed = subject_bgr.copy()
    x0, y0, w0, h0 = roi_src
    cv2.rectangle(boxed, (x0, y0), (x0 + w0, y0 + h0), (0, 180, 255), 2)
    cv2.imwrite(str(out_dir / "roi_box.png"), boxed)

    work_r, resize_scale = ps.resize_for_resolution(
        subject_bgr, resolution, max_side=int(cfg["max_side"])
    )
    work_mask_r = cv2.resize(
        subject_mask, (work_r.shape[1], work_r.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    work_mask_r = ps._strip_fringe_white(work_r, work_mask_r)
    roi = ps._scale_roi(roi_src, subject_bgr.shape[:2], work_r.shape[:2])
    face_m = ps._face_mask(work_r.shape[:2], roi)
    face_m = cv2.bitwise_and(face_m, work_mask_r)
    face_dil = cv2.dilate(face_m, np.ones((9, 9), np.uint8), iterations=2)

    pattern = EmbPattern()
    layers: list[dict[str, Any]] = []
    first = True
    if hybrid_body:
        pattern, body_layers, first = emit_body_tatami(
            pattern,
            work_r,
            work_mask_r,
            face_dil,
            subject_path,
            spacing=spacing,
            stitch_len=stitch_len,
            target_height_mm=target_height_mm,
            first=first,
            n_colors=n_colors,
        )
        layers.extend(body_layers)

    pattern, dens_layers, first, maps = emit_density_layers(
        pattern,
        work_r,
        face_m,
        n_colors=n_colors,
        step=step,
        target_height_mm=target_height_mm,
        first=first,
    )
    layers.extend(dens_layers)
    pattern.end()

    dens, keep, tone = maps["dens"], maps["keep"], maps["tone"]
    core = maps.get("core")
    dens_vis = render_density_preview(dens, face_m)
    # 计划图：色调底 + 落点半透明加深；五官核轻微着色便于对照
    plan = tone.copy()
    mix = (tone.astype(np.float32) * 0.55).astype(np.uint8)
    plan = np.where(keep[..., None] > 0, mix, plan)
    if core is not None and np.any(core):
        cv2.imwrite(str(out_dir / "feature_core.png"), core)
    cv2.imwrite(str(out_dir / "density_map.png"), dens_vis)
    cv2.imwrite(str(out_dir / "plan_preview.png"), plan)
    cv2.imwrite(str(out_dir / "tone_preview.png"), tone)
    cv2.imwrite(str(out_dir / "keep_mask.png"), keep)

    dst = out_dir / "auto.dst"
    write_dst(pattern, str(dst))
    write_png(pattern, str(out_dir / "auto_pyembroidery.png"))
    render_clean_preview(pattern, out_dir / "auto_stitch_thin.png", layers=layers, thick=False)
    render_clean_preview(pattern, out_dir / "auto_stitch_thick.png", layers=layers, thick=True)
    shutil.copyfile(out_dir / "plan_preview.png", out_dir / "auto_clean_preview.png")

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
        "n_colors": n_colors,
        "resolution": resolution,
        "step_px": step,
        "resize_scale": resize_scale,
        "roi": {"mode": "face", "xywh": [int(x0), int(y0), int(w0), int(h0)]},
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
        "preview": {
            "plan": "plan_preview.png",
            "density": "density_map.png",
            "core": "feature_core.png",
            "stitch_thin": "auto_stitch_thin.png",
            "stitch_thick": "auto_stitch_thick.png",
            "dual_track": True,
            "main_thumb": "plan_preview.png",
        },
        "note": "v4.1.1：邻接串针(不跨空拉线)+脸步密+五官核；保 v4 点阵层次",
    }
    (out_dir / "photo_stitch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps({**report, "density_field": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "dst": dst, "out_dir": out_dir, "report": report, "meta": meta}
