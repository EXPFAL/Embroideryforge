"""人像观感启发式（对照 MSEmb：窄条 satin / 大面 tatami / 小块轻填或 running）。

目标：减轻「整脸同向密填」的惊悚感，而不是追工业级人像绣。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# MSEmb 风格：大面错开角度；脸用更疏、更斜的填针
PORTRAIT_ANGLES = {
    "clothes": 0.0,
    "hat": 32.0,
    "hair": 58.0,
    "skin": 18.0,  # 避免 90° 竖条扫脸
    "highlight": 12.0,
    "lips": 8.0,
    "glasses": 0.0,
    "eyes": 25.0,
    "features": 0.0,
    "outline": 0.0,
}


def heal_skin_face_holes(
    label_img: np.ndarray,
    label_tags: list[str],
) -> np.ndarray:
    """把脸部肤色邻域内的小碎色块并回 skin，消除 EFR 造成的白洞/杂斑。"""
    if "skin" not in label_tags:
        return label_img
    skin_id = label_tags.index("skin")
    out = label_img.copy()
    skin = out == skin_id
    if int(skin.sum()) < 200:
        return out
    # 脸邻域：膨胀 skin，吸收其中的小异色/空洞（label<0 视作空洞则跳过）
    dil = cv2.dilate(skin.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=2) > 0
    foreign = dil & (out != skin_id) & (out >= 0)
    # outline 常为最大 id 且环绕全身，勿并回
    if "outline" in label_tags:
        foreign &= out != label_tags.index("outline")
    if "features" in label_tags:
        foreign &= out != label_tags.index("features")
    if not np.any(foreign):
        # 仅填 skin 内部空洞（若有负标签）
        return out
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(
        foreign.astype(np.uint8) * 255, connectivity=8
    )
    skin_area = int(skin.sum())
    max_comp = max(400, int(skin_area * 0.08))
    for i in range(1, ncc):
        if int(stats[i, cv2.CC_STAT_AREA]) <= max_comp:
            out[markers == i] = skin_id
    return out


def inject_facial_features(
    label_img: np.ndarray,
    label_tags: list[str],
    layers: list[dict[str, Any]],
    subject_bgr: np.ndarray,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """从肤色区内抽暗部为 features（五官/阴影线），用 satin 勾形，减轻糊脸。"""
    if "skin" not in label_tags:
        return label_img, label_tags, layers
    if subject_bgr.shape[:2] != label_img.shape[:2]:
        subject_bgr = cv2.resize(
            subject_bgr, (label_img.shape[1], label_img.shape[0]), interpolation=cv2.INTER_AREA
        )

    label_img = heal_skin_face_holes(label_img, label_tags)
    skin_id = label_tags.index("skin")
    skin = label_img == skin_id
    gray = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2GRAY)
    # 肤色区内相对暗的像素
    skin_vals = gray[skin]
    if skin_vals.size < 200:
        return label_img, label_tags, layers
    # 略放宽分位：照片/EFR 肤色区边界更碎时仍能抓住眉眼口
    thr = float(np.percentile(skin_vals, 20))
    dark = skin & (gray <= thr)
    # 局部对比：比邻域明显更暗的像素（抗均匀阴影）
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    local_dark = skin & ((blur.astype(np.int16) - gray.astype(np.int16)) >= 10)
    dark_u8 = ((dark | local_dark).astype(np.uint8) * 255)
    dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    # 只保留小连通域（五官线），丢掉半脸阴影大块——大块仍留在 skin 轻填
    ncc, markers, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
    keep = np.zeros_like(dark_u8)
    skin_area = int(skin.sum())
    max_comp = max(220, int(skin_area * 0.025))
    min_comp = 14
    for i in range(1, ncc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if min_comp <= a <= max_comp:
            keep[markers == i] = 255
    # 加粗一圈，避免 running/satin 在预览里几乎看不见
    dark_u8 = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    area = int(np.count_nonzero(dark_u8))
    if area < max(30, int(skin_area * 0.003)):
        return label_img, label_tags, layers

    # 从 skin 抠出 features；线色强制深褐（勿用区域均色，否则接近肤色看不见）
    new_id = len(label_tags)
    label_out = label_img.copy()
    label_out[dark_u8 > 0] = new_id
    tags_out = list(label_tags) + ["features"]
    feat_bgr = [28, 32, 48]  # BGR 深褐
    hex_c = "#30201c"
    layers_out = list(layers) + [
        {
            "tag": "features",
            "hex": hex_c,
            "bgr": feat_bgr,
            "area_px": area,
            "contour_count": 1,
            "role_hint": "satin_column",
        }
    ]
    return label_out, tags_out, layers_out


def apply_portrait_msemb_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """改写已有 plan：分角度、脸轻填、五官/features 用 satin 或密 running。"""
    out = dict(plan)
    layers: list[dict[str, Any]] = []
    for L in plan.get("layers", []):
        e = dict(L)
        tag = str(e.get("tag", ""))
        if tag in PORTRAIT_ANGLES:
            e["angle_deg"] = float(PORTRAIT_ANGLES[tag])

        if tag == "skin":
            e["stitch"] = "tatami"
            # 脸更疏，减少「铁面具」
            if e.get("row_spacing_px") is not None:
                e["row_spacing_px"] = round(float(e["row_spacing_px"]) * 1.45, 3)
            if e.get("stitch_len_px") is not None:
                e["stitch_len_px"] = round(float(e["stitch_len_px"]) * 1.15, 3)
            e["fill_erode"] = 1  # 勿过度腐蚀，否则脸出现大块留白更惊悚
            e["role_hint"] = "tatami_fill_soft"

        elif tag == "highlight":
            e["stitch"] = "tatami"
            if e.get("row_spacing_px") is not None:
                e["row_spacing_px"] = round(float(e["row_spacing_px"]) * 1.6, 3)
            e["fill_erode"] = max(int(e.get("fill_erode") or 1), 2)

        elif tag in {"lips", "glasses", "eyes"}:
            e["stitch"] = "satin"
            e["row_spacing_px"] = float(e.get("row_spacing_px") or 1.4)
            e["fill_erode"] = 0
            e["role_hint"] = "satin_column"

        elif tag == "features":
            # 五官：窄 satin 柱，比单线 running 在预览/成绣上更可辨
            e["stitch"] = "satin"
            e["row_spacing_px"] = 1.15
            e["stitch_len_px"] = 1.6
            e["fill_erode"] = 0
            e["hex"] = e.get("hex") or "#30201c"
            e["role_hint"] = "satin_column"
            e["order"] = max(int(e.get("order") or 50), 85)

        elif tag == "outline":
            e["stitch"] = "running"
            e["stitch_len_px"] = min(float(e.get("stitch_len_px") or 2.0), 1.8)
            e["order"] = max(int(e.get("order") or 90), 100)

        elif tag == "hair":
            e["stitch"] = "tatami"
            if e.get("row_spacing_px") is not None:
                e["row_spacing_px"] = round(float(e["row_spacing_px"]) * 0.95, 3)

        elif tag == "hat":
            e["stitch"] = "tatami"
            if e.get("row_spacing_px") is not None:
                e["row_spacing_px"] = round(float(e["row_spacing_px"]) * 1.1, 3)

        elif tag == "clothes":
            e["stitch"] = "tatami"
            # 衣服可相对密，撑主体

        layers.append(e)

    # features 排在 outline 前、填充后
    def _ord(x: dict[str, Any]) -> tuple[float, str]:
        tag = str(x.get("tag", ""))
        if tag == "features":
            return (85.0, tag)
        if tag == "outline":
            return (100.0, tag)
        return (float(x.get("order", 50)), tag)

    layers = sorted(layers, key=_ord)
    for i, e in enumerate(layers):
        e["order"] = (i + 1) * 10

    out["layers"] = layers
    out["portrait_style"] = "msemb_soft_face_v1"
    return out
