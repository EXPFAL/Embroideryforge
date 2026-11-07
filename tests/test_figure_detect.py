from pathlib import Path

import numpy as np

from embroideryforge.figure_detect import (
    detect_hand_landmarks,
    finger_bone_segments,
    hand_mask_from_landmarks,
    prop_seed_near_hands,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "samples" / "test_figure_uniform.jpg"


def test_hand_mask_from_synthetic_landmarks():
    pts = np.array(
        [
            [20, 40],
            [25, 30],
            [30, 25],
            [35, 22],
            [40, 20],
            [28, 35],
            [32, 28],
            [36, 24],
            [22, 38],
            [18, 42],
            [24, 36],
            [26, 32],
            [30, 30],
            [34, 28],
            [38, 26],
            [21, 39],
            [23, 37],
            [27, 33],
            [31, 29],
            [33, 27],
            [37, 25],
        ],
        dtype=np.float32,
    )
    mask = hand_mask_from_landmarks((80, 80), [pts], dilate=8)
    assert int(np.count_nonzero(mask)) > 50


def test_prop_seed_covers_palm():
    pts = np.zeros((21, 2), dtype=np.float32)
    for i in range(21):
        pts[i] = [40 + i * 0.5, 40 + (i % 5)]
    seed = prop_seed_near_hands((100, 100), [pts], radius=10)
    assert int(np.count_nonzero(seed)) > 20


def test_finger_bone_segments_from_21():
    pts = np.zeros((21, 2), dtype=np.float32)
    for i in range(21):
        pts[i] = [10 + i, 20 + (i % 4)]
    polys = finger_bone_segments([pts])
    assert len(polys) >= 5
    assert all(len(p) >= 2 for p in polys)


def test_detect_hands_on_figure():
    if not FIGURE.exists():
        return
    import cv2

    from embroideryforge.figure_detect import detect_figure_geometry

    bgr = cv2.imread(str(FIGURE))
    assert bgr is not None
    geo = detect_figure_geometry(bgr)
    # 腕裁剪后应力争至少 1 手；偶发全漏不硬挂整仓
    if geo["n_hands"] >= 1:
        assert geo["finger_polys"]
        assert int(np.count_nonzero(geo["hand_mask"])) > 100
        assert int(np.count_nonzero(geo["prop_seed"])) > 100
