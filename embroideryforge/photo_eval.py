"""照片域对照评估：naive vs EFR，不替代 chibi 五维。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from embroideryforge.plan import PLAN_FILENAME, load_plan


def photo_extras(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    plan_path = out_dir / PLAN_FILENAME
    report_path = out_dir / "auto_stitch_report.json"
    extras: dict[str, Any] = {
        "has_efr": (out_dir / "efr.png").is_file(),
        "has_features": False,
        "layer_count": 0,
        "angles": [],
        "angle_diversity": 0.0,
        "stitches": 0,
        "domain": None,
        "portrait_style": None,
    }
    if plan_path.is_file():
        plan = load_plan(plan_path)
        layers = [L for L in plan.get("layers", []) if L.get("enabled", True)]
        extras["layer_count"] = len(layers)
        extras["has_features"] = any(L.get("tag") == "features" for L in layers)
        angles = [float(L.get("angle_deg") or 0.0) for L in layers if L.get("stitch") != "running"]
        extras["angles"] = angles
        extras["angle_diversity"] = float(np.std(angles)) if len(angles) >= 2 else 0.0
        extras["portrait_style"] = plan.get("portrait_style")
    if report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        extras["stitches"] = int(rep.get("stitches") or 0)
        extras["domain"] = rep.get("domain")
        extras["efr"] = rep.get("efr")
    return extras


def compare_photo_exports(
    naive_dir: str | Path,
    efr_dir: str | Path,
    *,
    out_json: str | Path | None = None,
) -> dict[str, Any]:
    naive_dir = Path(naive_dir)
    efr_dir = Path(efr_dir)
    a = photo_extras(naive_dir)
    b = photo_extras(efr_dir)
    report = {
        "ok": True,
        "naive": {"dir": str(naive_dir), **a},
        "efr": {"dir": str(efr_dir), **b},
        "delta_stitches": int(b["stitches"] - a["stitches"]),
        "delta_angle_diversity": round(float(b["angle_diversity"] - a["angle_diversity"]), 4),
        "previews": {
            "naive": str(naive_dir / "auto_clean_preview.png"),
            "efr": str(efr_dir / "auto_clean_preview.png"),
            "efr_image": str(efr_dir / "efr.png"),
            "naive_compare": str(naive_dir / "layers_compare.png"),
            "efr_compare": str(efr_dir / "layers_compare.png"),
        },
        "note": "人评见 docs/创新点-照片EFR.md 模板；本报告只做可复现客观字段对照",
    }
    path = Path(out_json) if out_json else efr_dir.parent / "compare_portrait.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report
