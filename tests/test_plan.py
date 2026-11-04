import json
from pathlib import Path

from embroideryforge.autofill import build_pattern, run_auto_dst
from embroideryforge.plan import (
    build_rules_plan,
    load_plan,
    ordered_enabled_layers,
    save_plan,
    validate_plan,
)
from embroideryforge.segment import segment_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"


def test_rules_plan_schema():
    layers = [
        {"tag": "clothes", "hex": "#808080", "role_hint": "tatami_fill"},
        {"tag": "outline", "hex": "#202020", "role_hint": "running_stroke"},
    ]
    plan = build_rules_plan(layers, target_height_mm=70.0)
    assert validate_plan(plan) == []
    assert plan["schema_version"] == 1
    assert plan["source"] == "rules"
    tags = [L["tag"] for L in ordered_enabled_layers(plan)]
    assert tags == ["clothes", "outline"]
    by_tag = {L["tag"]: L for L in plan["layers"]}
    assert by_tag["outline"]["stitch"] == "running"
    assert by_tag["clothes"]["stitch"] == "tatami"


def test_plan_overrides_angle(tmp_path: Path):
    assert SAMPLE.exists()
    seg = segment_image(SAMPLE, mode="palette", k=10, upscale=2)
    plan = build_rules_plan(seg["layers"], target_height_mm=70.0)
    for L in plan["layers"]:
        if L["tag"] == "clothes":
            L["angle_deg"] = 77.0
    path = save_plan(plan, tmp_path / "stitch_plan.json")
    loaded = load_plan(path)
    pattern, meta = build_pattern(
        seg["layers"],
        seg["label_img"],
        seg["label_tags"],
        plan=loaded,
    )
    assert pattern.count_stitches() > 100
    clothes = next(x for x in meta["layers"] if x["tag"] == "clothes")
    assert clothes["angle_deg"] == 77.0


def test_stitch_auto_writes_plan(tmp_path: Path):
    result = run_auto_dst(SAMPLE, tmp_path / "auto", mode="palette", k=10)
    assert result["ok"]
    assert result["plan"].exists()
    plan = json.loads(result["plan"].read_text(encoding="utf-8"))
    assert plan["source"] == "rules"
    assert validate_plan(plan) == []
