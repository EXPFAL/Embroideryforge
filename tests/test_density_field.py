from pathlib import Path

import numpy as np

from embroideryforge.density_field import (
    density_map,
    dither_keep_mask,
    feature_core_mask,
    points_to_row_runs,
    points_to_serpentine_runs,
    run_density_v4,
)

ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"


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
    # 远隔行不串；同排可连
    ys = np.array([1, 1, 1, 20, 20])
    xs = np.array([1, 2, 3, 5, 6])
    polys = points_to_row_runs(ys, xs, max_gap=2)
    assert len(polys) >= 2


def test_serpentine_merges_adjacent_rows():
    # 行尾邻接才串；大跨距不并（避免空白拉线）
    ys = np.array([10, 10, 10, 11, 11, 11])
    xs = np.array([2, 3, 4, 4, 3, 2])
    polys = points_to_serpentine_runs(ys, xs, max_gap=2, max_row_jump=1, max_connect_dx=2)
    assert len(polys) == 1
    assert len(polys[0]) >= 6
    # 同行两团相隔太远 → 仍拆段
    ys2 = np.array([5, 5, 5, 5])
    xs2 = np.array([1, 2, 20, 21])
    polys2 = points_to_serpentine_runs(ys2, xs2, max_gap=2, max_row_jump=1, max_connect_dx=2)
    assert len(polys2) >= 2


def test_feature_core_picks_dark_spots():
    gray = np.full((80, 80), 180, dtype=np.uint8)
    gray[20:28, 25:40] = 30  # eye-like
    gray[45:52, 30:50] = 40  # mouth-like
    mask = np.full((80, 80), 255, dtype=np.uint8)
    core = feature_core_mask(gray, mask)
    assert int(np.count_nonzero(core)) > 20
    assert int(np.count_nonzero(core[20:28, 25:40])) > 0


def test_run_density_v4(tmp_path: Path):
    assert PORTRAIT.exists()
    out = tmp_path / "v4"
    result = run_density_v4(
        PORTRAIT,
        out,
        n_colors=9,
        resolution="low",
        bg_mode="auto",
        hybrid_body=True,
    )
    assert result["ok"]
    assert result["meta"]["engine"] == "photo_stitch_density_v4_1"
    assert result["meta"]["v41"]["serpentine"] is True
    assert (out / "density_map.png").is_file()
    assert (out / "feature_core.png").is_file()
    assert (out / "plan_preview.png").is_file()
    assert (out / "auto.dst").is_file()
    assert (out / "auto_stitch_thin.png").is_file()
    assert result["report"]["stitches"] > 200
    dens_layers = [L for L in result["meta"]["layers"] if str(L.get("tag", "")).startswith("dens_")]
    assert dens_layers
    assert dens_layers[0].get("stitch") == "density_serpentine"
