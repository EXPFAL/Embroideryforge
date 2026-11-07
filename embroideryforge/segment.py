"""自动分色分区。

默认 mode=palette：k-means 取图中真实色心 → 映射到命名调色板 → 同名合并。
比「像素直接贴固定色」更能适应 JPEG 色偏；比盲 k-means 标签更稳。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 名称, BGR 参考色（仅用于命名，不要求像素精确等于该值）
CHIBI_PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("outline", (38, 39, 59)),
    ("shoes", (10, 10, 10)),
    ("rose", (125, 125, 215)),       # #D77D7D 玫瑰
    ("hair", (198, 198, 246)),       # #F6C6C6 浅粉发
    ("skin", (197, 216, 249)),       # #F9D8C5 桃色肤
    ("clothes", (128, 128, 128)),    # 灰西装
    ("patches", (199, 193, 185)),
    ("eyes", (173, 231, 218)),       # 眼/叶绿色系
    ("leaf", (165, 217, 185)),
]

# 真人像：避免把发/影误挂成 chibi 的 rose
PORTRAIT_PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("outline", (40, 40, 40)),
    ("hair", (45, 40, 35)),
    ("skin", (180, 195, 230)),
    ("clothes", (40, 200, 230)),      # 亮黄衣等
    ("hat", (120, 150, 180)),         # 卡其帽
    ("lips", (90, 90, 180)),
    ("glasses", (150, 150, 150)),
    ("highlight", (235, 235, 235)),
]


@dataclass
class LayerInfo:
    tag: str
    hex: str
    bgr: list[int]
    area_px: int
    contour_count: int
    role_hint: str


def _bgr_to_hex(bgr: np.ndarray | list[int]) -> str:
    b, g, r = (int(x) for x in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _role_hint(tag: str) -> str:
    if tag in {"outline", "shoes"}:
        return "running_stroke"
    if tag in {"eyes", "patches", "leaf", "lips", "glasses"}:
        return "satin_or_small_tatami"
    return "tatami_fill"


def detect_palette_domain(bgr: np.ndarray) -> str:
    """粗判插画 vs 照片。

    照片：色种多、饱和度方差大；扁平插画：硬边拉普拉斯方差高、色种相对少。
    """
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (min(160, w), min(160, h)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    q = (small // 24).astype(np.int32)
    codes = q[:, :, 0] + q[:, :, 1] * 12 + q[:, :, 2] * 144
    n_unique = len(np.unique(codes))
    sat_std = float(hsv[:, :, 1].std())
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if n_unique >= 140 and sat_std >= 45 and lap < 2500:
        return "portrait"
    return "chibi"


def _name_center_portrait(
    bgr: np.ndarray, palette: list[tuple[str, tuple[int, int, int]]]
) -> str:
    b, g, r = (int(x) for x in bgr)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hh, ss, vv = (int(x) for x in hsv)

    if vv <= 38:
        return "outline"
    # 深褐/黑发（先于唇，避免大块阴影被标 lips）
    if vv <= 100 and ss <= 80:
        return "hair"
    # 亮黄外套
    if 18 <= hh <= 45 and ss >= 70 and vv >= 90:
        return "clothes"
    # 卡其/褐黄帽：中饱和褐，面积通常大于唇
    if 5 <= hh <= 30 and 25 <= ss <= 110 and 70 <= vv <= 195:
        return "hat"
    # 唇：高饱和真红/玫，范围更窄
    if (hh <= 7 or hh >= 172) and ss >= 85 and 95 <= vv <= 200:
        return "lips"
    # 肤色
    if 3 <= hh <= 32 and 10 <= ss <= 90 and vv >= 115:
        return "skin"
    # 眼镜金属/灰
    if ss <= 50 and 85 <= vv <= 205:
        return "glasses"
    if vv >= 215 and ss <= 45:
        return "highlight"

    best, best_d = "skin", 1e9
    for name, ref in palette:
        d = float(
            np.linalg.norm(
                np.array([b, g, r], dtype=np.float32) - np.array(ref, dtype=np.float32)
            )
        )
        if d < best_d:
            best_d = d
            best = name
    if best_d > 110:
        return f"misc_{_bgr_to_hex([b, g, r])[1:]}"
    return best


def _fix_oversized_lips(
    tags: list[str], counts: np.ndarray, total_fg: int
) -> list[str]:
    """唇层若占比过大，多半是帽/发误标 → 改挂 hat。"""
    if total_fg <= 0:
        return tags
    out = list(tags)
    for i, tag in enumerate(out):
        if tag == "lips" and counts[i] / total_fg > 0.04:
            out[i] = "hat"
    return out


def _preprocess(bgr: np.ndarray, upscale: int = 2) -> tuple[np.ndarray, float]:
    smooth = cv2.medianBlur(bgr, 3)
    smooth = cv2.bilateralFilter(smooth, d=5, sigmaColor=20, sigmaSpace=5)
    scale = 1.0
    if upscale > 1:
        smooth = cv2.resize(smooth, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
        scale = float(upscale)
    return smooth, scale


def _name_center(
    bgr: np.ndarray,
    palette: list[tuple[str, tuple[int, int, int]]],
    *,
    domain: str = "chibi",
) -> str:
    """色心命名：深色 / 绿 / 玫瑰 / 桃皮 / 粉发 / 灰衣，其余最近邻。"""
    if domain == "portrait":
        return _name_center_portrait(bgr, palette)
    b, g, r = (int(x) for x in bgr)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hh, ss, vv = (int(x) for x in hsv)

    # 1) 深色：描边或鞋/领结
    if vv <= 55:
        return "outline" if ss >= 12 or vv > 22 else "shoes"

    # 2) 绿色：眼睛或叶子
    if 35 <= hh <= 95 and ss >= 30 and vv >= 100:
        # 更黄绿且较亮 → eyes；稍暗 → leaf
        return "eyes" if vv >= 150 else "leaf"

    # 3) 玫瑰：红/玫红，饱和度较高、亮度中等（比头发深）
    # OpenCV H: 红约 0-10 与 170-179
    is_red = hh <= 12 or hh >= 170
    if is_red and ss >= 50 and 70 <= vv <= 200:
        return "rose"

    # 4) 桃色皮肤：偏橙黄（H 8-25），饱和度中低，很亮
    if 5 <= hh <= 28 and 15 <= ss <= 80 and vv >= 175:
        return "skin"

    # 5) 浅粉头发：品红/粉（H 高或低），饱和中等，很亮，且比玫瑰浅
    is_magenta_pink = hh >= 140 or hh <= 10
    if is_magenta_pink and 20 <= ss <= 70 and vv >= 170:
        return "hair"

    # 6) 灰色衣服
    if ss <= 40 and 60 <= vv <= 180:
        return "clothes"

    # 7) 浅灰补丁
    if ss <= 35 and vv >= 170:
        return "patches"

    # 最近邻兜底
    best, best_d = "detail", 1e9
    for name, ref in palette:
        d = float(np.linalg.norm(np.array([b, g, r], dtype=np.float32) - np.array(ref, dtype=np.float32)))
        if d < best_d:
            best_d = d
            best = name
    if best_d > 95:
        return f"misc_{_bgr_to_hex([b, g, r])[1:]}"
    return best


def _cluster_and_name(
    bgr: np.ndarray,
    k: int,
    ignore_near_white: int,
    palette: list[tuple[str, tuple[int, int, int]]],
    *,
    domain: str = "chibi",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """返回合并命名后的 label_img、tags、centers。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    bgr_flat = bgr.reshape(-1, 3)
    keep = np.any(bgr_flat < ignore_near_white, axis=1)
    data = pixels[keep] if int(keep.sum()) >= k * 30 else pixels
    keep_idx = np.where(keep)[0] if int(keep.sum()) >= k * 30 else np.arange(len(pixels))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.15)
    _, labels_sub, centers_lab = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers_lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    raw = np.full((h * w,), -1, dtype=np.int32)
    raw[keep_idx] = labels_sub.flatten()
    raw = raw.reshape(h, w)

    # 每个簇命名
    names = [_name_center(centers_bgr[i], palette, domain=domain) for i in range(k)]
    if domain == "portrait":
        # 先按簇面积修一次 lips（用 raw 计数）
        raw_counts = np.array([int((raw == i).sum()) for i in range(k)], dtype=np.int64)
        names = _fix_oversized_lips(names, raw_counts, int(np.sum(raw >= 0)))

    # 同名合并
    unique: list[str] = []
    for n in names:
        if n not in unique:
            unique.append(n)
    name_to_id = {n: i for i, n in enumerate(unique)}

    merged = np.full_like(raw, -1)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    for i, n in enumerate(names):
        mid = name_to_id[n]
        mask = raw == i
        merged[mask] = mid
        c = int(mask.sum())
        if c:
            sums[mid] += bgr[mask].sum(axis=0)
            counts[mid] += c

    if domain == "portrait":
        unique = _fix_oversized_lips(unique, counts, int(counts.sum()))
        # 若改名产生重复，再压一遍 id
        rebuilt: list[str] = []
        remap = {}
        for old_i, n in enumerate(unique):
            if n not in remap:
                remap[n] = len(rebuilt)
                rebuilt.append(n)
        if len(rebuilt) != len(unique):
            new_merged = np.full_like(merged, -1)
            new_sums = np.zeros((len(rebuilt), 3), dtype=np.float64)
            new_counts = np.zeros(len(rebuilt), dtype=np.int64)
            for old_i, n in enumerate(unique):
                ni = remap[n]
                m = merged == old_i
                new_merged[m] = ni
                new_sums[ni] += sums[old_i]
                new_counts[ni] += counts[old_i]
            merged, unique, sums, counts = new_merged, rebuilt, new_sums, new_counts

    centers = np.zeros((len(unique), 3), dtype=np.uint8)
    for i in range(len(unique)):
        if counts[i]:
            centers[i] = (sums[i] / counts[i]).astype(np.uint8)

    return merged, unique, centers


def _mask_to_svg_path(mask: np.ndarray, simplify_ratio: float = 0.0009) -> tuple[str, int]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or not contours:
        return "", 0
    hier = hierarchy[0]
    parts: list[str] = []
    count = 0
    for i, _cnt in enumerate(contours):
        if hier[i][3] != -1:
            continue
        chain = [i]
        child = hier[i][2]
        while child != -1:
            chain.append(child)
            child = hier[child][0]
        for ci in chain:
            c = contours[ci]
            peri = cv2.arcLength(c, True)
            if peri < 10:
                continue
            approx = cv2.approxPolyDP(c, max(0.35, simplify_ratio * peri), True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            cmds = [f"M {pts[0, 0]:.2f} {pts[0, 1]:.2f}"]
            for x, y in pts[1:]:
                cmds.append(f"L {x:.2f} {y:.2f}")
            cmds.append("Z")
            parts.append(" ".join(cmds))
            count += 1
    return " ".join(parts), count


def _scale_path_coords(d: str, inv_scale: float) -> str:
    if inv_scale == 1.0:
        return d
    out: list[str] = []
    tokens = d.replace(",", " ").split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in {"M", "L", "Z", "z"}:
            out.append(t)
            i += 1
            continue
        if i + 1 < len(tokens) and tokens[i + 1] not in {"M", "L", "Z", "z"}:
            out.append(f"{float(t) * inv_scale:.2f}")
            out.append(f"{float(tokens[i + 1]) * inv_scale:.2f}")
            i += 2
        else:
            out.append(t)
            i += 1
    return " ".join(out)


def _clean_mask(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    kept = 0
    for comp in range(1, ncc):
        a = int(stats[comp, cv2.CC_STAT_AREA])
        if a >= min_area:
            clean[markers == comp] = 255
            kept += a
    return clean, kept


def segment_image(
    image_path: str | Path,
    *,
    mode: str = "palette",
    k: int = 10,
    min_area_ratio: float = 0.0012,
    ignore_near_white: int = 248,
    simplify_ratio: float = 0.0009,
    upscale: int = 2,
    domain: str = "auto",
) -> dict[str, Any]:
    path = Path(image_path)
    bgr0 = cv2.imread(str(path))
    if bgr0 is None:
        raise FileNotFoundError(f"无法读取: {path}")
    h0, w0 = bgr0.shape[:2]
    bgr, scale = _preprocess(bgr0, upscale=upscale)

    if mode not in {"palette", "kmeans"}:
        raise ValueError("mode 必须是 palette 或 kmeans")

    resolved = domain if domain in {"chibi", "portrait"} else detect_palette_domain(bgr0)
    palette = PORTRAIT_PALETTE if resolved == "portrait" else CHIBI_PALETTE

    # palette 与 kmeans 都走聚类；palette 多一步命名合并
    label_img, tags, centers = _cluster_and_name(
        bgr,
        k=k,
        ignore_near_white=ignore_near_white,
        palette=palette,
        domain=resolved,
    )
    if mode == "kmeans":
        # 保留分离簇名，不合并
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        bgr_flat = bgr.reshape(-1, 3)
        keep = np.any(bgr_flat < ignore_near_white, axis=1)
        data = lab[keep]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.15)
        _, labels_sub, centers_lab = cv2.kmeans(data, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
        centers_bgr = cv2.cvtColor(
            np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_LAB2BGR
        ).reshape(-1, 3)
        raw = np.full((bgr.shape[0] * bgr.shape[1],), -1, dtype=np.int32)
        raw[np.where(keep)[0]] = labels_sub.flatten()
        label_img = raw.reshape(bgr.shape[:2])
        tags = [f"color_{i}_{_bgr_to_hex(centers_bgr[i])[1:]}" for i in range(k)]
        centers = centers_bgr

    n = len(tags)
    h, w = bgr.shape[:2]
    min_area = max(48, int(h * w * min_area_ratio))
    inv = 1.0 / scale
    z_rank = {
        "clothes": 10,
        "hat": 15,
        "hair": 20,
        "skin": 30,
        "rose": 35,
        "lips": 38,
        "patches": 40,
        "eyes": 50,
        "glasses": 52,
        "leaf": 55,
        "highlight": 58,
        "detail": 60,
        "features": 85,
        "shoes": 80,
        "outline": 90,
    }

    staged: list[tuple[int, LayerInfo, str]] = []
    for i, tag in enumerate(tags):
        mask = (label_img == i).astype(np.uint8) * 255
        if not np.any(mask):
            continue
        clean, kept = _clean_mask(mask, min_area)
        if kept < min_area:
            continue
        d, ncont = _mask_to_svg_path(clean, simplify_ratio=simplify_ratio)
        if not d:
            continue
        d = _scale_path_coords(d, inv)
        hex_color = _bgr_to_hex(centers[i])
        if tag in {"outline", "shoes"} or tag.startswith("misc_"):
            style = f'fill="none" stroke="{hex_color}" stroke-width="{1.8 * inv:.2f}"'
        else:
            style = f'fill="{hex_color}" stroke="none" fill-rule="evenodd"'
        g = (
            f'  <g id="{tag}" inkscape:label="{tag}" data-role="{_role_hint(tag)}">\n'
            f'    <path d="{d}" {style} />\n'
            f"  </g>"
        )
        info = LayerInfo(
            tag=tag,
            hex=hex_color,
            bgr=[int(x) for x in centers[i]],
            area_px=int(kept / (scale * scale)),
            contour_count=ncont,
            role_hint=_role_hint(tag),
        )
        staged.append((z_rank.get(tag, 15 + i), info, g))

    staged.sort(key=lambda t: t[0])
    layers = [s[1] for s in staged]
    svg_groups = [s[2] for s in staged]

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{w0}" height="{h0}" viewBox="0 0 {w0} {h0}">',
            f"  <title>EmbroideryForge segment ({mode}): {path.name}</title>",
            *svg_groups,
            "</svg>",
            "",
        ]
    )

    label_small = cv2.resize(label_img.astype(np.float32), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    composite = np.full_like(bgr0, 255)
    for i in range(n):
        m = label_small == i
        if np.any(m):
            composite[m] = centers[i]

    # 原分辨率标签图，供自动填针复用（与 layers 同源）
    label_orig = cv2.resize(
        label_img.astype(np.float32), (w0, h0), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    return {
        "width": w0,
        "height": h0,
        "mode": mode,
        "domain": resolved,
        "k": k,
        "source": path.name,
        "layers": [asdict(x) for x in layers],
        "svg": svg,
        "composite_bgr": composite,
        "label_img": label_orig,
        "label_tags": tags,
    }


def run_segment(
    image_path: str | Path,
    out_dir: str | Path,
    *,
    mode: str = "palette",
    k: int = 10,
    min_area_ratio: float = 0.0012,
    upscale: int = 2,
    merge_dist: float = 30.0,
    domain: str = "auto",
) -> dict[str, Any]:
    del merge_dist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = segment_image(
        image_path,
        mode=mode,
        k=k,
        min_area_ratio=min_area_ratio,
        upscale=upscale,
        domain=domain,
    )
    svg_path = out_dir / "layers.svg"
    json_path = out_dir / "layers.json"
    svg_path.write_text(result["svg"], encoding="utf-8")

    composite = result.pop("composite_bgr")
    result.pop("label_img", None)
    result.pop("label_tags", None)
    comp_path = out_dir / "layers_composite.png"
    cv2.imwrite(str(comp_path), composite)
    original = cv2.imread(str(image_path))
    if original is not None:
        if original.shape[:2] != composite.shape[:2]:
            composite = cv2.resize(
                composite, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        cv2.imwrite(str(out_dir / "layers_compare.png"), np.hstack([original, composite]))

    payload = {key: val for key, val in result.items() if key != "svg"}
    payload["svg_file"] = svg_path.name
    payload["composite"] = comp_path.name
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": len(result["layers"]) > 0, "out_dir": out_dir, "report": payload, "svg": svg_path}
