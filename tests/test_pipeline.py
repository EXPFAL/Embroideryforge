from pathlib import Path

from embroideryforge.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-logo.jpg"


def test_golden_pipeline(tmp_path: Path):
    assert SAMPLE.exists(), "缺少黄金样例 samples/golden-logo.jpg"
    out = tmp_path / "run"
    result = run_pipeline(SAMPLE, out)
    assert result["ok"] is True
    assert (out / "embroidery.dst").exists()
    assert (out / "report.json").exists()
    assert result["export"]["stitches"] > 0
