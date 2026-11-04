from pathlib import Path

from embroideryforge.segment import run_segment

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"


def test_segment_chibi(tmp_path: Path):
    assert SAMPLE.exists()
    result = run_segment(SAMPLE, tmp_path / "seg", mode="palette")
    assert result["ok"] is True
    assert (tmp_path / "seg" / "layers.svg").exists()
    assert (tmp_path / "seg" / "layers.json").exists()
    assert (tmp_path / "seg" / "layers_compare.png").exists()
    tags = {x["tag"] for x in result["report"]["layers"]}
    assert "clothes" in tags
    assert ("hair" in tags) or ("skin" in tags)
    assert ("outline" in tags) or ("shoes" in tags)
