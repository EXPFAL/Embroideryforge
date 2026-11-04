from pathlib import Path

from embroideryforge.autofill import run_auto_dst
from embroideryforge.evaluate import evaluate_export_dir

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "golden-chibi.jpg"
REF = ROOT / "exports" / "inkscape_smoke" / "chibi_multicolor.dst"


def test_evaluate_after_stitch_auto(tmp_path: Path):
    out = tmp_path / "auto"
    run_auto_dst(SAMPLE, out, mode="palette", k=10)
    ref = REF if REF.exists() else None
    result = evaluate_export_dir(out, source_image=SAMPLE, reference_dst=ref)
    assert result["overall"] >= 0.55
    assert result["scores"]["dst_openable"] == 1.0
    assert result["scores"]["plan_valid"] == 1.0
    assert Path(result["report"]).exists()
