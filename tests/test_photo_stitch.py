from pathlib import Path

from embroideryforge.photo_stitch import _face_mask, quantize_tones, run_photo_stitch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"
NEW_PORTRAIT = ROOT / "samples" / "test_portrait_new.jpg"


def test_face_mask_covers_lower_roi():
    # 椭圆应盖到 ROI 下部（下颚区），避免脸-颈断层
    roi = (10, 10, 100, 120)
    m = _face_mask((200, 200), roi)
    x, y, rw, rh = roi
    lower = m[y + int(rh * 0.75) : y + rh, x : x + rw]
    assert int(np.count_nonzero(lower)) > 50


def test_quantize_tones_basic():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32] = (180, 200, 220)
    img[32:] = (40, 40, 40)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    tone, labels, centers = quantize_tones(img, mask, n_colors=2)
    assert len(centers) == 2
    assert tone.shape == img.shape
    assert labels.max() >= 0


def test_stitch_photo_writes_dst(tmp_path: Path):
    assert PORTRAIT.exists()
    out = tmp_path / "ps"
    result = run_photo_stitch(
        PORTRAIT,
        out,
        n_colors=6,
        resolution="low",
        bg_mode="auto",
        mode="face",
    )
    assert result["ok"]
    assert (out / "auto.dst").is_file()
    assert (out / "tone_preview.png").is_file()
    assert (out / "density_preview.png").is_file()
    assert (out / "photo_stitch_meta.json").is_file()
    assert (out / "compare_preview.png").is_file()
    assert result["report"]["stitches"] > 100
    assert result["report"]["color_changes"] >= 1


def test_stitch_photo_hybrid(tmp_path: Path):
    out = tmp_path / "hy"
    result = run_photo_stitch(
        PORTRAIT,
        out,
        n_colors=7,
        resolution="low",
        bg_mode="auto",
        mode="hybrid",
    )
    assert result["ok"]
    assert result["meta"]["mode"] == "hybrid"
    assert (out / "layout_preview.png").is_file()
    assert (out / "auto_stitch_thin.png").is_file()
    assert (out / "dark_features.png").is_file()
    assert result["meta"]["preview"]["stroke_width"] >= 2
    tags = [L.get("tag") for L in result["meta"]["layers"]]
    assert "face_underlay" in tags
    assert "face_dark" in tags
    assert result["report"]["stitches"] > 200


def test_stitch_photo_poster(tmp_path: Path):
    out = tmp_path / "poster"
    result = run_photo_stitch(
        PORTRAIT,
        out,
        n_colors=9,
        resolution="low",
        bg_mode="auto",
        mode="hybrid",
        poster=True,
    )
    assert result["ok"]
    assert result["meta"]["poster"] is True
    assert 9 <= result["meta"]["n_colors"] <= 12
    assert "poster" in result["meta"]["engine"]
    assert any(L.get("stitch") == "photo_poster_fill" for L in result["meta"]["layers"])


def test_density_gate_keeps_darker():
    from embroideryforge.photo_stitch import apply_density_gate

    mask = np.full((32, 32), 255, dtype=np.uint8)
    gray = np.zeros((32, 32), dtype=np.uint8)
    gray[:16] = 40
    gray[16:] = 240
    gated = apply_density_gate(mask, gray, highlight_only=True)
    assert int(np.count_nonzero(gated[:16])) == 16 * 32
    assert int(np.count_nonzero(gated[16:])) < 16 * 32


def test_clean_and_block_angles():
    from embroideryforge.photo_stitch import clean_layer_mask, estimate_block_angles

    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    mask[0, 0] = 255  # 碎点
    cleaned = clean_layer_mask(mask, min_area=20)
    assert cleaned[0, 0] == 0
    assert int(np.count_nonzero(cleaned)) > 100
    gray = np.tile(np.linspace(30, 200, 40, dtype=np.uint8), (40, 1))
    angs = estimate_block_angles(gray, cleaned, block=10)
    assert len(angs) >= 1
