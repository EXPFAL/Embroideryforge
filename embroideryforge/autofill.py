"""从自动分区结果生成多色 DST（Tatami / Satin 填针 + Running 描边）。

针法参数以 stitch_plan.json 为准；无计划时用规则版生成后再执行。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyembroidery import EmbPattern, write_dst, write_png

from embroideryforge.bg import prepare_subject
from embroideryforge.efr import write_efr
from embroideryforge.plan import (
    PLAN_FILENAME,
    apply_quality,
    build_rules_plan,
    load_plan,
    ordered_enabled_layers,
    save_plan,
)
from embroideryforge.portrait_style import apply_portrait_msemb_plan, inject_facial_features
from embroideryforge.preview import render_clean_preview
from embroideryforge.segment import detect_palette_domain, segment_image


def _hex_to_rgb_name(hex_color: str) -> str:
    return hex_color if hex_color.startswith("#") else f"#{hex_color}"


def _runs_on_row(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = len(row)
    i = 0
    while i < n:
        if row[i] == 0:
            i += 1
            continue
        j = i
        while j < n and row[j] != 0:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _map_pt(xr: float, yr: float, Minv: np.ndarray) -> tuple[float, float]:
    out = cv2.transform(np.array([[[xr, yr]]], dtype=np.float32), Minv)[0, 0]
    return float(out[0]), float(out[1])


def _tatami_polylines(
    mask: np.ndarray, angle_deg: float, spacing: float, stitch_len: float
) -> list[list[tuple[float, float]]]:
    """返回多段折线；段与段之间应由 JUMP 连接，避免长实缝跨空。"""
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot = cv2.warpAffine(
        mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    Minv = cv2.invertAffineTransform(M)
    polylines: list[list[tuple[float, float]]] = []
    stagger = True
    y = 0.0
    while y < h:
        yi = int(round(y))
        if 0 <= yi < h:
            for x0, x1 in _runs_on_row(rot[yi]):
                if x1 - x0 < 1:
                    continue
                start = x0 + (spacing * 0.45 if stagger else 0.0)
                x = start
                line: list[tuple[float, float]] = []
                while x <= x1:
                    line.append(_map_pt(x, float(yi), Minv))
                    x += stitch_len
                end = _map_pt(float(x1), float(yi), Minv)
                if not line or abs(line[-1][0] - end[0]) + abs(line[-1][1] - end[1]) > 0.5:
                    line.append(end)
                if len(line) >= 2:
                    polylines.append(line)
        stagger = not stagger
        y += spacing
    return polylines


def _running_polylines(mask: np.ndarray, stitch_len: float) -> list[list[tuple[float, float]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polylines: list[list[tuple[float, float]]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < 20:
            continue
        seq = cnt.reshape(-1, 2)
        if len(seq) < 2:
            continue
        step = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        dists = np.cumsum(step)
        dists = np.insert(dists, 0, 0.0)
        total = float(dists[-1])
        line: list[tuple[float, float]] = []
        if total < stitch_len:
            line = [(float(x), float(y)) for x, y in seq]
        else:
            n = max(2, int(total / stitch_len))
            for s in np.linspace(0, total, n):
                i = int(np.searchsorted(dists, s, side="right") - 1)
                i = min(max(i, 0), len(seq) - 2)
                t = 0.0 if dists[i + 1] == dists[i] else (s - dists[i]) / (dists[i + 1] - dists[i])
                x = seq[i, 0] * (1 - t) + seq[i + 1, 0] * t
                y = seq[i, 1] * (1 - t) + seq[i + 1, 1] * t
                line.append((float(x), float(y)))
        if len(line) >= 2:
            polylines.append(line)
    return polylines


def _satin_polylines(
    mask: np.ndarray, angle_deg: float, spacing: float
) -> list[list[tuple[float, float]]]:
    """柱状缎面（column satin）：沿角度扫描，每条色带左右缘一针折返。

    适合眼睛、补丁等窄区域；宽区域会形成较长缎面针，仍可检查打开。
    """
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot = cv2.warpAffine(
        mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    Minv = cv2.invertAffineTransform(M)
    polylines: list[list[tuple[float, float]]] = []
    flip = False
    y = 0.0
    while y < h:
        yi = int(round(y))
        if 0 <= yi < h:
            for x0, x1 in _runs_on_row(rot[yi]):
                if x1 - x0 < 1:
                    continue
                a = _map_pt(float(x0), float(yi), Minv)
                b = _map_pt(float(x1), float(yi), Minv)
                polylines.append([b, a] if flip else [a, b])
                flip = not flip
        y += max(spacing, 0.5)
    return polylines


def _to_emb_xy(x: float, y: float, scale: float, height: int) -> tuple[int, int]:
    return int(round(x * scale)), int(round((height - y) * scale))


def build_pattern_from_plan(
    plan: dict[str, Any],
    label_img: np.ndarray,
    label_tags: list[str],
    *,
    layer_colors: dict[str, str] | None = None,
) -> tuple[EmbPattern, dict[str, Any]]:
    """按 stitch_plan 生成 EmbPattern。"""
    h, _w = label_img.shape
    target_height_mm = float(plan.get("target_height_mm", 70.0))
    scale = (target_height_mm / max(h, 1)) * 10.0
    pattern = EmbPattern()
    stats: list[dict[str, Any]] = []

    tag_to_ids: dict[str, list[int]] = {}
    for i, tag in enumerate(label_tags):
        tag_to_ids.setdefault(tag, []).append(i)

    colors = layer_colors or {}
    erode_k = np.ones((3, 3), np.uint8)
    first = True

    for entry in ordered_enabled_layers(plan):
        tag = str(entry["tag"])
        if tag not in tag_to_ids:
            continue
        hex_color = entry.get("hex") or colors.get(tag, "#808080")
        stitch = entry.get("stitch", "tatami")
        angle = float(entry.get("angle_deg", 0.0))
        fill_erode = int(
            entry.get(
                "fill_erode",
                1 if stitch in {"tatami", "satin"} else 0,
            )
        )

        mask = np.zeros(label_img.shape, dtype=np.uint8)
        for i in tag_to_ids[tag]:
            mask[label_img == i] = 255
        if not np.any(mask):
            continue

        if stitch in {"tatami", "satin"} and fill_erode > 0:
            mask = cv2.erode(mask, erode_k, iterations=fill_erode)
            if not np.any(mask):
                continue

        if stitch == "running":
            stitch_len = float(entry.get("stitch_len_px") or 2.5)
            polys = _running_polylines(mask, stitch_len=stitch_len)
        elif stitch == "satin":
            spacing = float(entry.get("row_spacing_px") or 1.6)
            polys = _satin_polylines(mask, angle_deg=angle, spacing=spacing)
        else:
            spacing = float(entry.get("row_spacing_px") or 2.0)
            stitch_len = float(entry.get("stitch_len_px") or 3.2)
            polys = _tatami_polylines(mask, angle_deg=angle, spacing=spacing, stitch_len=stitch_len)

        if not polys:
            continue

        if not first:
            pattern.color_change()
        first = False
        pattern.add_thread(_hex_to_rgb_name(str(hex_color)))

        n_pts = 0
        for poly in polys:
            block = [_to_emb_xy(x, y, scale, h) for x, y in poly]
            if len(block) < 2:
                continue
            pattern.move_abs(block[0][0], block[0][1])
            for x, y in block:
                pattern.stitch_abs(x, y)
                n_pts += 1

        if n_pts < 2:
            continue
        stats.append(
            {
                "tag": tag,
                "stitch": stitch,
                "role": entry.get("role_hint", stitch),
                "angle_deg": angle,
                "row_spacing_px": entry.get("row_spacing_px"),
                "stitch_len_px": entry.get("stitch_len_px"),
                "order": entry.get("order"),
                "points": n_pts,
                "segments": len(polys),
                "hex": hex_color,
            }
        )

    pattern.end()
    return pattern, {
        "stitches": pattern.count_stitches(),
        "color_changes": pattern.count_color_changes(),
        "layers": stats,
        "scale_emb_per_px": scale,
        "target_height_mm": target_height_mm,
        "plan_source": plan.get("source", "unknown"),
    }


def build_pattern(
    layers: list[dict[str, Any]],
    label_img: np.ndarray,
    label_tags: list[str],
    *,
    target_height_mm: float = 70.0,
    plan: dict[str, Any] | None = None,
) -> tuple[EmbPattern, dict[str, Any]]:
    """兼容入口：无 plan 时用规则版生成。"""
    if plan is None:
        plan = build_rules_plan(layers, target_height_mm=target_height_mm)
    colors = {str(L["tag"]): L.get("hex", "#808080") for L in layers}
    return build_pattern_from_plan(plan, label_img, label_tags, layer_colors=colors)


def run_auto_dst(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    mode: str = "palette",
    k: int = 12,
    upscale: int = 2,
    target_height_mm: float = 70.0,
    plan_path: str | Path | None = None,
    use_llm: bool = False,
    bg_mode: str = "auto",
    quality: str = "dense",
    domain: str = "auto",
    use_efr: bool | None = None,
    efr_colors: int = 7,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.setRNGSeed(42)
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(image_path)
    subject_bgr, subject_mask, bg_meta = prepare_subject(original, mode=bg_mode)
    subject_path = out_dir / "subject.png"
    mask_path = out_dir / "subject_mask.png"
    cv2.imwrite(str(subject_path), subject_bgr)
    cv2.imwrite(str(mask_path), subject_mask)
    (out_dir / "bg_meta.json").write_text(
        json.dumps(bg_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resolved_hint = domain if domain in {"chibi", "portrait"} else detect_palette_domain(subject_bgr)
    # 照片域默认开 EFR；插画默认关
    efr_on = bool(use_efr) if use_efr is not None else (resolved_hint == "portrait")
    efr_meta: dict[str, Any] | None = None
    seg_input = subject_path
    efr_bgr = None
    if efr_on:
        efr_meta = write_efr(subject_bgr, out_dir, n_colors=efr_colors, stem="efr")
        efr_bgr = efr_meta.pop("image", None)
        seg_input = out_dir / "efr.png"

    use_upscale = upscale
    if resolved_hint == "portrait" and upscale > 1:
        use_upscale = 1

    seg = segment_image(
        seg_input, mode=mode, k=k, upscale=use_upscale, domain=domain
    )
    label_img = seg.pop("label_img")
    label_tags = seg.pop("label_tags")
    composite = seg.pop("composite_bgr")

    # 五官必须从去背原图抽暗部；EFR 少色会抹掉眉眼口对比，若用 efr_bgr 会只剩色块
    if seg.get("domain") == "portrait":
        label_img, label_tags, seg["layers"] = inject_facial_features(
            label_img, label_tags, seg["layers"], subject_bgr
        )
        if "features" in label_tags:
            fid = label_tags.index("features")
            feat_meta = next(L for L in seg["layers"] if L["tag"] == "features")
            bgr_f = feat_meta.get("bgr") or [40, 40, 40]
            composite = composite.copy()
            if composite.shape[:2] == label_img.shape[:2]:
                composite[label_img == fid] = bgr_f

    (out_dir / "layers.svg").write_text(seg["svg"], encoding="utf-8")
    cv2.imwrite(str(out_dir / "layers_composite.png"), composite)

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[:2] == original.shape[:2]:
            return img
        return cv2.resize(img, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    panels = [original, _fit(subject_bgr)]
    if efr_bgr is not None:
        panels.append(_fit(efr_bgr))
    panels.append(_fit(composite))
    cv2.imwrite(str(out_dir / "layers_compare.png"), np.hstack(panels))
    layers_payload = {key: val for key, val in seg.items() if key != "svg"}
    layers_payload["svg_file"] = "layers.svg"
    layers_payload["composite"] = "layers_composite.png"
    layers_payload["subject"] = subject_path.name
    layers_payload["subject_mask"] = mask_path.name
    layers_payload["bg"] = bg_meta
    layers_payload["domain"] = seg.get("domain", domain)
    layers_payload["efr"] = bool(efr_on)
    if efr_meta:
        layers_payload["efr_meta"] = {k: v for k, v in efr_meta.items() if k != "image"}
    (out_dir / "layers.json").write_text(
        json.dumps(layers_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if plan_path is not None:
        plan = load_plan(plan_path)
        if "target_height_mm" not in plan:
            plan["target_height_mm"] = target_height_mm
        if "quality" not in plan:
            plan = apply_quality(plan, quality)
    elif use_llm:
        from embroideryforge.llm import build_llm_plan

        plan = build_llm_plan(
            seg["layers"],
            target_height_mm=target_height_mm,
            fallback=True,
            quality=quality,
        )
    else:
        plan = build_rules_plan(
            seg["layers"],
            target_height_mm=target_height_mm,
            source="rules",
            quality=quality,
        )
    if seg.get("domain") == "portrait":
        plan = apply_portrait_msemb_plan(plan)
    plan_to_save = {k: v for k, v in plan.items()}
    plan_file = save_plan(plan_to_save, out_dir / PLAN_FILENAME)

    pattern, meta = build_pattern(
        seg["layers"], label_img, label_tags, target_height_mm=target_height_mm, plan=plan
    )
    dst = out_dir / "auto.dst"
    png = out_dir / "auto.png"
    clean_png = out_dir / "auto_clean_preview.png"
    raw_png = out_dir / "auto_pyembroidery.png"
    write_dst(pattern, str(dst))
    # 原始库预览：色对，但 JUMP 连线乱，仅作对照
    write_png(pattern, str(raw_png), settings={"background": "#ffffff"})

    # 主预览：断 JUMP + 与 layers 同序 hex（DST 本身不存 RGB）
    preview_info = render_clean_preview(
        pattern,
        clean_png,
        layers=meta["layers"],
    )
    # auto.png 与 clean 同图，避免「两个预览两套色」
    shutil.copyfile(clean_png, png)
    threads_sidecar = [
        {"index": i, "tag": L.get("tag"), "hex": L.get("hex")}
        for i, L in enumerate(meta["layers"])
    ]
    (out_dir / "threads.json").write_text(
        json.dumps(threads_sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "source": Path(image_path).name,
        "dst": dst.name,
        "preview": png.name,
        "clean_preview": clean_png.name,
        "pyembroidery_preview": raw_png.name,
        "threads": "threads.json",
        "plan": plan_file.name,
        "quality": plan.get("quality", quality),
        "bg": bg_meta,
        "domain": layers_payload.get("domain"),
        "efr": bool(efr_on),
        "efr_meta": layers_payload.get("efr_meta"),
        **meta,
    }
    (out_dir / "auto_stitch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "ok": meta["stitches"] > 0,
        "dst": dst,
        "png": png,
        "clean_preview": clean_png,
        "plan": plan_file,
        "report": report,
        "preview_info": preview_info,
        "efr": efr_on,
    }
