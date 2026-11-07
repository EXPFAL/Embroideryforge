from pathlib import Path

import numpy as np

from embroideryforge.face_detect import (
    FACE_OVAL_IDX,
    face_mask_from_landmarks,
    roi_from_landmarks,
)
from embroideryforge.photo_stitch import detect_face_geometry, _face_mask

ROOT = Path(__file__).resolve().parents[1]
NEW_PORTRAIT = ROOT / "samples" / "test_portrait_new.jpg"
PORTRAIT = ROOT / "samples" / "golden-portrait.jpg"


def test_roi_from_landmarks_covers_chin_tip():
    # 合成脸廓：下巴尖在底部
    pts = np.zeros((478, 2), dtype=np.float32)
    for i, (x, y) in enumerate(
        [
            (40, 20),
            (60, 18),
            (80, 25),
            (90, 50),
            (85, 80),
            (50, 95),  # chin-ish
            (20, 80),
            (15, 50),
            (25, 25),
        ]
    ):
        if i < len(FACE_OVAL_IDX):
            pts[FACE_OVAL_IDX[i]] = (x, y)
    # 确保 152 为最下
    pts[152] = (50, 100)
    roi = roi_from_landmarks(pts, (120, 120), pad_bot_frac=0.05)
    _x, y, _w, h = roi
    assert y + h >= 100


def test_face_mask_from_landmarks_filled():
    pts = np.zeros((478, 2), dtype=np.float32)
    oval = np.array(
        [[30, 20], [70, 20], [90, 50], [70, 90], [30, 90], [10, 50]], dtype=np.float32
    )
    for i, idx in enumerate(FACE_OVAL_IDX[: len(oval)]):
        pts[idx] = oval[i]
    mask = face_mask_from_landmarks((100, 100), pts)
    assert int(np.count_nonzero(mask)) > 500


def test_detect_geometry_on_sample():
    import cv2

    path = NEW_PORTRAIT if NEW_PORTRAIT.exists() else PORTRAIT
    assert path.exists()
    bgr = cv2.imread(str(path))
    assert bgr is not None
    geo = detect_face_geometry(bgr, subject_mask=None)
    assert "roi" in geo
    x, y, w, h = geo["roi"]
    assert w > 40 and h > 40
    # MediaPipe 可用时应标 mediapipe；否则 haar/bust 也不应崩溃
    assert geo["source"] in {"mediapipe", "haar", "bust"}
    if geo["source"] == "mediapipe":
        assert geo["landmarks"] is not None
        mask = _face_mask(bgr.shape[:2], geo["roi"], landmarks=geo["landmarks"])
        # 下半 ROI 应有脸罩像素（含下巴）
        yy = y + int(h * 0.75)
        assert int(np.count_nonzero(mask[yy : y + h, x : x + w])) > 20
