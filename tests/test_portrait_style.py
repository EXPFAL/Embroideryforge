from embroideryforge.portrait_style import apply_portrait_msemb_plan


def test_msemb_softens_skin_and_orders_features():
    plan = {
        "schema_version": 1,
        "source": "rules",
        "layers": [
            {
                "tag": "skin",
                "stitch": "tatami",
                "angle_deg": 90,
                "row_spacing_px": 2.0,
                "stitch_len_px": 3.0,
                "order": 10,
                "fill_erode": 1,
                "enabled": True,
            },
            {
                "tag": "features",
                "stitch": "tatami",
                "angle_deg": 0,
                "row_spacing_px": 2.0,
                "stitch_len_px": 3.0,
                "order": 20,
                "fill_erode": 1,
                "enabled": True,
            },
            {
                "tag": "outline",
                "stitch": "running",
                "angle_deg": 0,
                "row_spacing_px": None,
                "stitch_len_px": 2.5,
                "order": 30,
                "fill_erode": 0,
                "enabled": True,
            },
        ],
    }
    out = apply_portrait_msemb_plan(plan)
    by = {L["tag"]: L for L in out["layers"]}
    assert by["skin"]["angle_deg"] == 18.0
    assert by["skin"]["row_spacing_px"] > 2.0
    assert by["features"]["stitch"] == "satin"
    assert out["portrait_style"] == "msemb_soft_face_v1"
    tags = [L["tag"] for L in out["layers"]]
    assert tags.index("features") < tags.index("outline")
