from pathlib import Path

import numpy as np

from embroideryforge.autofill import _satin_polylines, build_pattern_from_plan
from embroideryforge.plan import build_rules_plan, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def test_satin_polylines_on_strip():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 15:25] = 255
    polys = _satin_polylines(mask, angle_deg=0.0, spacing=2.0)
    assert len(polys) >= 3
    assert all(len(p) == 2 for p in polys)


def test_satin_in_pattern():
    label = np.zeros((50, 50), dtype=np.int32)
    label[10:40, 10:40] = 0
    label[18:32, 18:32] = 1
    layers = [
        {"tag": "clothes", "hex": "#888888", "role_hint": "tatami_fill"},
        {"tag": "eyes", "hex": "#222222", "role_hint": "satin_or_small_tatami"},
    ]
    plan = build_rules_plan(layers, quality="standard")
    assert validate_plan(plan) == []
    assert next(L for L in plan["layers"] if L["tag"] == "eyes")["stitch"] == "satin"
    pattern, meta = build_pattern_from_plan(
        plan, label, ["clothes", "eyes"], layer_colors={"clothes": "#888888", "eyes": "#222222"}
    )
    assert pattern.count_stitches() > 20
    stitches = {L["tag"]: L["stitch"] for L in meta["layers"]}
    assert stitches.get("eyes") == "satin"
