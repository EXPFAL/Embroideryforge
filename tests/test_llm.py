import json
from pathlib import Path
from unittest.mock import patch

from embroideryforge.llm import (
    _extract_json_object,
    build_llm_plan,
    merge_llm_into_rules,
)
from embroideryforge.plan import build_rules_plan, validate_plan

LAYERS = [
    {"tag": "clothes", "hex": "#808080", "area_px": 1000, "role_hint": "tatami_fill", "contour_count": 1},
    {"tag": "outline", "hex": "#111111", "area_px": 200, "role_hint": "running_stroke", "contour_count": 2},
]


def test_extract_json_from_fence():
    text = 'Here:\n```json\n{"a": 1}\n```\n'
    assert _extract_json_object(text) == {"a": 1}


def test_merge_llm_overrides_angle():
    rules = build_rules_plan(LAYERS)
    llm_obj = {
        "layers": [
            {"tag": "clothes", "stitch": "tatami", "angle_deg": 33, "order": 5},
            {"tag": "outline", "stitch": "running", "order": 99},
        ]
    }
    plan = merge_llm_into_rules(rules, llm_obj)
    assert validate_plan(plan) == []
    assert plan["source"] == "llm"
    by_tag = {L["tag"]: L for L in plan["layers"]}
    assert by_tag["clothes"]["angle_deg"] == 33
    assert by_tag["clothes"]["hex"] == "#808080"
    assert by_tag["outline"]["stitch"] == "running"


def test_build_llm_plan_fallback_on_error():
    with patch("embroideryforge.llm.chat_completion", side_effect=RuntimeError("no network")):
        plan = build_llm_plan(LAYERS, fallback=True)
    assert plan["source"] == "rules_fallback"
    assert plan["llm_meta"]["fallback"] is True


def test_build_llm_plan_success(monkeypatch):
    fake = {
        "schema_version": 1,
        "layers": [
            {
                "tag": "clothes",
                "stitch": "tatami",
                "angle_deg": 12,
                "row_spacing_px": 1.8,
                "stitch_len_px": 3.0,
                "order": 10,
                "fill_erode": 1,
                "enabled": True,
            },
            {
                "tag": "outline",
                "stitch": "running",
                "angle_deg": 0,
                "stitch_len_px": 2.5,
                "order": 90,
                "fill_erode": 0,
                "enabled": True,
            },
        ],
        "notes": "mock",
    }

    def fake_chat(*_a, **_k):
        return json.dumps(fake)

    with patch("embroideryforge.llm.chat_completion", side_effect=fake_chat):
        plan = build_llm_plan(LAYERS, fallback=False)
    assert plan["source"] == "llm"
    assert next(L for L in plan["layers"] if L["tag"] == "clothes")["angle_deg"] == 12
