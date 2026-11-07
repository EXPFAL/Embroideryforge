from pathlib import Path

import cv2
import numpy as np

from embroideryforge.efr import apply_efr, write_efr

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"


def test_apply_efr_reduces_colors():
    bgr = cv2.imread(str(PORTRAIT))
    assert bgr is not None
    efr, meta = apply_efr(bgr, n_colors=6)
    assert efr.shape == bgr.shape
    # 量化后唯一色显著少于原图采样
    q = (efr.reshape(-1, 3) // 8).astype(np.int32)
    codes = q[:, 0] + q[:, 1] * 40 + q[:, 2] * 1600
    assert len(np.unique(codes)) <= 40
    assert meta["n_colors_used"] <= 6


def test_write_efr(tmp_path: Path):
    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    bgr[:20] = (40, 180, 220)
    bgr[20:] = (180, 200, 40)
    meta = write_efr(bgr, tmp_path, n_colors=4)
    assert meta["ok"]
    assert (tmp_path / "efr.png").is_file()
    assert (tmp_path / "efr_meta.json").is_file()
