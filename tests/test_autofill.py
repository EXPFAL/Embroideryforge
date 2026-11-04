from pathlib import Path

from embroideryforge.autofill import run_auto_dst

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"


def test_auto_dst(tmp_path: Path):
    assert SAMPLE.exists()
    result = run_auto_dst(SAMPLE, tmp_path / "auto", mode="palette", k=10)
    assert result["ok"] is True
    assert result["dst"].exists()
    assert result["report"]["stitches"] > 100
    assert result["report"]["color_changes"] >= 1
