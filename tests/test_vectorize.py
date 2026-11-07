from pathlib import Path

from embroideryforge.vectorize import vectorize_image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"


def test_vectorize_fallback_or_vtracer(tmp_path: Path):
    assert SAMPLE.exists()
    meta = vectorize_image(SAMPLE, tmp_path / "vec", prefer_vtracer=True)
    assert meta["ok"] is True
    assert (tmp_path / "vec" / "vector.svg").is_file()
    assert meta.get("engine") in {"vtracer", "opencv_contours"}
