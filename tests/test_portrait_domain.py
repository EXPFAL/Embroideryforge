from pathlib import Path

import cv2

from embroideryforge.segment import detect_palette_domain, segment_image

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"
CHIBI = ROOT / "samples" / "golden-chibi.jpg"


def test_detect_portrait_vs_chibi():
    assert PORTRAIT.exists() and CHIBI.exists()
    p = cv2.imread(str(PORTRAIT))
    c = cv2.imread(str(CHIBI))
    assert detect_palette_domain(p) == "portrait"
    assert detect_palette_domain(c) == "chibi"


def test_portrait_segment_tags():
    seg = segment_image(PORTRAIT, mode="palette", k=12, domain="portrait", upscale=1)
    tags = {L["tag"] for L in seg["layers"]}
    assert seg["domain"] == "portrait"
    assert "rose" not in tags
    assert tags & {"skin", "clothes", "hair", "hat"}
