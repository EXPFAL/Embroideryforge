from pathlib import Path

from embroideryforge.autofill import run_auto_dst
from embroideryforge.photo_eval import compare_photo_exports, photo_extras

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"


def test_portrait_efr_writes_efr_png(tmp_path: Path):
    assert PORTRAIT.exists()
    out = tmp_path / "efr"
    result = run_auto_dst(
        PORTRAIT,
        out,
        domain="portrait",
        use_efr=True,
        use_llm=False,
        quality="draft",
        k=8,
        bg_mode="auto",
    )
    assert result["ok"]
    assert (out / "efr.png").is_file()
    assert result["report"].get("efr") is True
    extras = photo_extras(out)
    assert extras["has_efr"] is True
    assert extras["has_features"] is True, "EFR 填充后仍应从原图叠五官层"


def test_compare_photo_exports(tmp_path: Path):
    naive = tmp_path / "naive"
    efr = tmp_path / "efr"
    run_auto_dst(PORTRAIT, naive, domain="portrait", use_efr=False, use_llm=False, quality="draft", k=8)
    run_auto_dst(PORTRAIT, efr, domain="portrait", use_efr=True, use_llm=False, quality="draft", k=8)
    rep = compare_photo_exports(naive, efr, out_json=tmp_path / "cmp.json")
    assert rep["ok"]
    assert Path(rep["path"]).is_file()
