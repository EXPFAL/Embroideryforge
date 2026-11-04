"""结构化针法计划 stitch_plan.json。

规则 / LLM / 手工共用同一 schema；autofill 只认这份计划，不认散落硬编码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PLAN_FILENAME = "stitch_plan.json"

VALID_STITCH = frozenset({"tatami", "running"})

# 绣序与默认角度（规则版基线；LLM 可覆盖同名字段）
DEFAULT_ORDER = [
    "clothes",
    "hair",
    "skin",
    "rose",
    "patches",
    "eyes",
    "leaf",
    "shoes",
    "outline",
]
ANGLE_BY_TAG = {
    "clothes": 0.0,
    "hair": 40.0,
    "skin": 90.0,
    "rose": 55.0,
    "patches": 15.0,
    "eyes": 30.0,
    "leaf": 20.0,
    "shoes": 0.0,
    "outline": 0.0,
}


def _default_stitch_for_tag(tag: str, role_hint: str | None = None) -> str:
    if role_hint == "running_stroke" or tag in {"outline", "shoes"}:
        return "running"
    return "tatami"


# 密度档：dense 目标逼近手工 Ink/Stitch 针数量级
QUALITY_PRESETS: dict[str, dict[str, float]] = {
    "draft": {"spacing_scale": 1.25, "len_scale": 1.15, "run_scale": 1.1},
    "standard": {"spacing_scale": 1.0, "len_scale": 1.0, "run_scale": 1.0},
    "dense": {"spacing_scale": 0.55, "len_scale": 0.68, "run_scale": 0.85},
}


def _default_spacing(tag: str, role_hint: str | None, base: float = 2.0) -> float:
    spacing = base
    if role_hint == "satin_or_small_tatami" or tag in {"eyes", "patches", "leaf"}:
        spacing *= 1.15
    if tag == "rose":
        spacing *= 0.9
    return round(spacing, 3)


def _default_stitch_len(tag: str, role_hint: str | None, *, tatami: bool) -> float:
    if not tatami:
        return 2.5
    length = 3.2
    if role_hint == "satin_or_small_tatami" or tag in {"eyes", "patches", "leaf"}:
        length *= 0.85
    if tag == "rose":
        length *= 0.9
    return round(length, 3)


def apply_quality(plan: dict[str, Any], quality: str = "standard") -> dict[str, Any]:
    """按档位缩放行距/针长（不改角度与绣序）。"""
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    out = dict(plan)
    layers: list[dict[str, Any]] = []
    for L in plan.get("layers", []):
        e = dict(L)
        if e.get("stitch") == "tatami":
            if e.get("row_spacing_px") is not None:
                e["row_spacing_px"] = round(float(e["row_spacing_px"]) * preset["spacing_scale"], 3)
            if e.get("stitch_len_px") is not None:
                e["stitch_len_px"] = round(float(e["stitch_len_px"]) * preset["len_scale"], 3)
        else:
            if e.get("stitch_len_px") is not None:
                e["stitch_len_px"] = round(float(e["stitch_len_px"]) * preset["run_scale"], 3)
        layers.append(e)
    out["layers"] = layers
    out["quality"] = quality
    return out


def build_rules_plan(
    layers: list[dict[str, Any]],
    *,
    target_height_mm: float = 70.0,
    source: str = "rules",
    quality: str = "standard",
) -> dict[str, Any]:
    """从分区 layers 生成可执行 stitch_plan（规则版 / 假 LLM 出口）。"""
    by_tag = {str(L["tag"]): L for L in layers}
    ordered = [t for t in DEFAULT_ORDER if t in by_tag]
    ordered += [t for t in by_tag if t not in ordered]

    plan_layers: list[dict[str, Any]] = []
    for i, tag in enumerate(ordered):
        meta = by_tag[tag]
        role = meta.get("role_hint")
        stitch = _default_stitch_for_tag(tag, role)
        entry: dict[str, Any] = {
            "tag": tag,
            "stitch": stitch,
            "angle_deg": float(ANGLE_BY_TAG.get(tag, 0.0)),
            "row_spacing_px": _default_spacing(tag, role) if stitch == "tatami" else None,
            "stitch_len_px": _default_stitch_len(tag, role, tatami=(stitch == "tatami")),
            "order": (i + 1) * 10,
            "fill_erode": 0 if stitch == "running" else 1,
            "enabled": True,
            "hex": meta.get("hex", "#808080"),
            "role_hint": role or ("running_stroke" if stitch == "running" else "tatami_fill"),
        }
        plan_layers.append(entry)

    plan = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "target_height_mm": float(target_height_mm),
        "layers": plan_layers,
    }
    return apply_quality(plan, quality)


def validate_plan(plan: dict[str, Any]) -> list[str]:
    """返回错误列表；空列表表示可用。"""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]
    ver = plan.get("schema_version")
    if ver != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}, got {ver!r}")
    if "layers" not in plan or not isinstance(plan["layers"], list):
        errors.append("layers must be a list")
        return errors
    if not plan["layers"]:
        errors.append("layers is empty")
    seen: set[str] = set()
    for i, layer in enumerate(plan["layers"]):
        if not isinstance(layer, dict):
            errors.append(f"layers[{i}] must be an object")
            continue
        tag = layer.get("tag")
        if not tag or not isinstance(tag, str):
            errors.append(f"layers[{i}].tag required")
        elif tag in seen:
            errors.append(f"duplicate tag: {tag}")
        else:
            seen.add(tag)
        stitch = layer.get("stitch")
        if stitch not in VALID_STITCH:
            errors.append(f"layers[{i}].stitch must be one of {sorted(VALID_STITCH)}")
        if "order" in layer and not isinstance(layer["order"], (int, float)):
            errors.append(f"layers[{i}].order must be a number")
    return errors


def load_plan(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_plan(plan)
    if errors:
        raise ValueError("invalid stitch_plan: " + "; ".join(errors))
    return plan


def save_plan(plan: dict[str, Any], path: str | Path) -> Path:
    errors = validate_plan(plan)
    if errors:
        raise ValueError("invalid stitch_plan: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ordered_enabled_layers(plan: dict[str, Any]) -> list[dict[str, Any]]:
    layers = [L for L in plan["layers"] if L.get("enabled", True)]
    return sorted(layers, key=lambda L: (float(L.get("order", 999)), str(L["tag"])))
