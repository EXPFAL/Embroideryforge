from pathlib import Path

import cv2
import numpy as np

from embroideryforge.bg import detect_bg_mode, prepare_subject
from embroideryforge.autofill import run_auto_dst

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / "samples" / "golden-chibi.jpg"
COMPLEX = ROOT / "samples" / "test_chibi_complex_bg.jpg"


def test_flat_skips_bg():
    bgr = cv2.imread(str(FLAT))
    assert bgr is not None
    assert detect_bg_mode(bgr) == "skip"
    out, mask, meta = prepare_subject(bgr, mode="auto")
    assert meta["applied"] is False
    assert out.shape == bgr.shape
    assert mask.any()


def test_complex_applies_bg_removal():
    if not COMPLEX.exists():
        return
    bgr = cv2.imread(str(COMPLEX))
    assert detect_bg_mode(bgr) == "contrast"
    out, mask, meta = prepare_subject(bgr, mode="auto")
    assert meta["applied"] is True
    assert meta.get("method") in {"rembg", "illustration_recon"}
    border = np.concatenate(
        [out[0, :], out[-1, :], out[:, 0], out[:, -1]], axis=0
    )
    assert float(np.mean(border >= 250)) > 0.85
    fg = out[mask > 0]
    assert float(np.mean(np.any(fg < 240, axis=1))) > 0.35
    assert 0.05 < float(np.mean(mask > 0)) < 0.75


def test_dense_auto_dst(tmp_path: Path):
    result = run_auto_dst(
        FLAT, tmp_path / "d", mode="palette", k=10, quality="dense", bg_mode="auto"
    )
    assert result["ok"]
    assert (tmp_path / "d" / "subject.png").exists()
    assert result["report"]["quality"] == "dense"
    assert result["report"]["stitches"] > 20000
