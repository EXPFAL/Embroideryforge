from pathlib import Path

import numpy as np

from embroideryforge.density_field import (
    build_figure_region_masks,
    chroma_accent_mask,
    density_map,
    dither_keep_mask,
    feature_core_mask,
    highlight_preserve_mask,
    points_to_row_runs,
    points_to_serpentine_runs,
    run_density_photo,
    run_density_v4,
    run_density_v6_unified,
    skin_exterior_mask,
    soften_highlight_centers,
    structure_band_mask,
)

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"
FIGURE = ROOT / "samples" / "test_figure_uniform.jpg"
COWBOY = ROOT / "samples" / "test_figure_cowboy.jpg"


def test_density_map_darker_is_denser():
    gray = np.zeros((32, 32), dtype=np.uint8)
    gray[:16] = 40
    gray[16:] = 220
    mask = np.full((32, 32), 255, dtype=np.uint8)
    dens = density_map(gray, mask)
    assert float(dens[:16].mean()) > float(dens[16:].mean())


def test_dither_keeps_more_in_dark():
    dens = np.zeros((64, 64), dtype=np.float32)
    dens[:32] = 0.85
    dens[32:] = 0.15
    mask = np.full((64, 64), 255, dtype=np.uint8)
    keep = dither_keep_mask(dens, mask)
    assert int(np.count_nonzero(keep[:32])) > int(np.count_nonzero(keep[32:]))


def test_points_to_row_runs():
    ys = np.array([1, 1, 1, 20, 20])
    xs = np.array([1, 2, 3, 5, 6])
    polys = points_to_row_runs(ys, xs, max_gap=2)
    assert len(polys) >= 2


def test_serpentine_merges_adjacent_rows():
    ys = np.array([10, 10, 10, 11, 11, 11])
    xs = np.array([2, 3, 4, 4, 3, 2])
    polys = points_to_serpentine_runs(ys, xs, max_gap=2, max_row_jump=1, max_connect_dx=2)
    assert len(polys) == 1
    assert len(polys[0]) >= 6
    ys2 = np.array([5, 5, 5, 5])
    xs2 = np.array([1, 2, 20, 21])
    polys2 = points_to_serpentine_runs(ys2, xs2, max_gap=2, max_row_jump=1, max_connect_dx=2)
    assert len(polys2) >= 2


def test_feature_core_picks_dark_spots():
    gray = np.full((80, 80), 180, dtype=np.uint8)
    gray[20:28, 25:40] = 30
    gray[45:52, 30:50] = 40
    mask = np.full((80, 80), 255, dtype=np.uint8)
    core = feature_core_mask(gray, mask)
    assert int(np.count_nonzero(core)) > 20
    assert int(np.count_nonzero(core[20:28, 25:40])) > 0


def test_highlight_preserve_picks_bright_spots():
    gray = np.full((80, 80), 100, dtype=np.uint8)
    gray[18:26, 28:38] = 230
    gray[48:54, 32:48] = 220
    mask = np.full((80, 80), 255, dtype=np.uint8)
    hi = highlight_preserve_mask(gray, mask)
    assert int(np.count_nonzero(hi)) > 10
    assert int(np.count_nonzero(hi[18:26, 28:38])) > 0


def test_structure_band_from_core():
    core = np.zeros((40, 40), dtype=np.uint8)
    core[10:18, 12:28] = 255
    band = structure_band_mask(core)
    assert int(np.count_nonzero(band)) > 0


def test_soften_highlight_centers_pulls_bright():
    dark = np.array([30, 40, 50], dtype=np.uint8)
    mid = np.array([100, 120, 140], dtype=np.uint8)
    bright = np.array([240, 245, 250], dtype=np.uint8)
    out = soften_highlight_centers([dark, mid, bright])
    # 最亮应被拉暗一点
    assert float(out[2].mean()) < float(bright.mean())


def test_skin_exterior_near_face_tone():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    # 脸：偏暖肤
    img[10:40, 25:55] = (90, 140, 190)
    face = np.zeros((80, 80), dtype=np.uint8)
    face[10:40, 25:55] = 255
    # 手：相近肤色，在脸外
    img[50:70, 10:35] = (85, 135, 185)
    mask = np.full((80, 80), 255, dtype=np.uint8)
    skin_x = skin_exterior_mask(img, face, mask)
    assert int(np.count_nonzero(skin_x[50:70, 10:35])) > 20
    assert int(np.count_nonzero(skin_x[10:40, 25:55])) == 0


def test_chroma_accent_picks_red():
    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    img[8:18, 8:22] = (40, 40, 200)  # BGR 红
    mask = np.full((40, 40), 255, dtype=np.uint8)
    excl = np.zeros((40, 40), dtype=np.uint8)
    acc = chroma_accent_mask(img, mask, excl)
    assert int(np.count_nonzero(acc[8:18, 8:22])) > 10


def test_build_figure_regions_keys():
    img = np.full((60, 60, 3), 100, dtype=np.uint8)
    img[5:25, 20:40] = (90, 140, 190)
    face = np.zeros((60, 60), dtype=np.uint8)
    face[5:25, 20:40] = 255
    mask = np.full((60, 60), 255, dtype=np.uint8)
    regs = build_figure_region_masks(img, face, mask)
    assert set(regs) >= {"skin_exterior", "accent", "prop"}


def test_run_density_v4(tmp_path: Path):
    assert PORTRAIT.exists()
    out = tmp_path / "v41"
    result = run_density_v4(
        PORTRAIT,
        out,
        n_colors=9,
        resolution="low",
        bg_mode="auto",
        hybrid_body=True,
    )
    assert result["ok"]
    # 默认保持 v4.1.1；v4.2 结构/高光已回退；V5 叠针已关
    assert result["meta"]["engine"] == "photo_stitch_density_v4_1"
    assert result["meta"]["v42"]["rolled_back"] is True
    assert result["meta"]["figure_regions"]["overlays_rolled_back"] is True
    assert (out / "density_map.png").is_file()
    assert (out / "feature_core.png").is_file()
    assert (out / "plan_preview.png").is_file()
    assert (out / "auto.dst").is_file()
    assert result["report"]["stitches"] > 200
    dens_layers = [L for L in result["meta"]["layers"] if str(L.get("tag", "")).startswith("dens_")]
    assert dens_layers
    assert dens_layers[0].get("stitch") == "density_serpentine"
    tags = {str(L.get("tag")) for L in result["meta"]["layers"]}
    assert "clothes_fold" not in tags
    assert "hand_bones" not in tags
    assert not any(L.get("tag") == "face_structure" for L in result["meta"]["layers"])


def test_run_density_v6_figure_unified(tmp_path: Path):
    if not FIGURE.exists():
        return
    out = tmp_path / "fig_v6"
    result = run_density_v6_unified(
        FIGURE,
        out,
        n_colors=10,
        resolution="low",
        bg_mode="auto",
    )
    assert result["ok"]
    assert result["meta"]["engine"] == "photo_stitch_density_v6"
    assert result["meta"]["mode"] == "density_unified"
    tags = [str(L.get("tag")) for L in result["meta"]["layers"]]
    assert all(t.startswith("dens_") for t in tags)
    assert "clothes_fold" not in tags
    assert "hand_bones" not in tags
    assert "prop_outline" not in tags
    assert (out / "figure_layout.png").is_file()
    assert (out / "auto.dst").is_file()
    assert result["report"]["stitches"] > 500


def test_run_density_v6_scene_split(tmp_path: Path):
    if not COWBOY.exists():
        return
    out = tmp_path / "cowboy_v63"
    result = run_density_v6_unified(
        COWBOY,
        out,
        n_colors=10,
        resolution="low",
        bg_mode="keep",
    )
    assert result["ok"]
    assert result["meta"]["mode"] == "density_split"
    assert result["meta"]["v6"].get("v63_person_bg_split") is True
    tags = [str(L.get("tag")) for L in result["meta"]["layers"]]
    assert any(t.startswith("dens_bg_") for t in tags)
    assert any(t.startswith("dens_fg_") for t in tags)
    assert (out / "bg_mask.png").is_file()
    assert (out / "fg_mask.png").is_file()
    assert (out / "auto.dst").is_file()
    assert result["report"]["stitches"] > 500
    # V6.3d：若有脸密轨，色值须与 FG 共用（无第二套 k-means）
    if any(t.startswith("dens_face_") for t in tags):
        assert result["meta"]["v6"].get("v63d_face_share_fg_palette") is True
        layers = result["meta"]["layers"]
        fg_hex = {L["hex"] for L in layers if str(L.get("tag", "")).startswith("dens_fg_")}
        face_hex = {L["hex"] for L in layers if str(L.get("tag", "")).startswith("dens_face_")}
        assert face_hex and face_hex <= fg_hex


def test_run_density_photo_auto_routes_figure(tmp_path: Path):
    if not FIGURE.exists():
        return
    out = tmp_path / "auto"
    result = run_density_photo(
        FIGURE,
        out,
        n_colors=10,
        resolution="low",
        bg_mode="auto",
        hybrid_body=True,
        unified=None,
    )
    assert result["ok"]
    assert result["meta"]["engine"] == "photo_stitch_density_v6"
