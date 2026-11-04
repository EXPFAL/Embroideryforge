from pathlib import Path

import cv2
import numpy as np

from embroideryforge.autofill import run_auto_dst
from embroideryforge.preview import _hex_to_bgr, render_clean_preview

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"


def test_clean_preview_uses_layer_hex(tmp_path: Path):
    result = run_auto_dst(SAMPLE, tmp_path / "auto", mode="palette", k=10)
    assert result["clean_preview"].exists()
    assert (tmp_path / "auto" / "threads.json").exists()

    # 用 report 色重渲，色板长度应等于层数
    info = render_clean_preview(
        result["dst"],
        tmp_path / "again.png",
        report_path=tmp_path / "auto" / "auto_stitch_report.json",
    )
    assert info["n_colors"] == len(result["report"]["layers"])
    assert info["stitches_drawn"] > 100

    # 图像中应能找到第一层 hex 的近似像素（允许抗锯齿偏差）
    img = cv2.imread(str(result["clean_preview"]))
    assert img is not None
    target = np.array(_hex_to_bgr(result["report"]["layers"][0]["hex"]), dtype=np.int16)
    diff = np.abs(img.astype(np.int16) - target).sum(axis=2)
    assert (diff < 40).any()
