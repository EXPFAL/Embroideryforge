"""MediaPipe Face Landmarker → 脸窗 ROI / 脸罩（含下巴关键点）。"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Face Mesh 脸廓（含下颌/下巴尖 152），足够定 ROI 与凸包罩
FACE_OVAL_IDX = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

# 耳廓外侧点：并入凸包，避免耳部落到 hat/hair 斜填
EAR_IDX = (
    234,
    227,
    116,
    143,
    123,
    147,
    213,
    192,
    214,
    138,
    135,
    151,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL = _ROOT / "models" / "face_landmarker.task"
_landmarker: Any | None = None
_landmarker_failed = False


def model_path() -> Path:
    env = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
    return env


def ensure_model(path: Path | None = None, *, download: bool = True) -> Path | None:
    """返回可用模型路径；缺失时可下载。失败返回 None。"""
    p = Path(path) if path is not None else model_path()
    if p.is_file() and p.stat().st_size > 1000:
        return p
    if not download:
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".task.download")
        urllib.request.urlretrieve(MODEL_URL, str(tmp))
        tmp.replace(p)
        if p.is_file() and p.stat().st_size > 1000:
            return p
    except Exception:
        return None
    return None


def _get_landmarker() -> Any | None:
    global _landmarker, _landmarker_failed
    if _landmarker is not None:
        return _landmarker
    if _landmarker_failed:
        return None
    mp_path = ensure_model(download=True)
    if mp_path is None:
        _landmarker_failed = True
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(mp_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            # V6.6b：多人侧脸也要进 dens_face；主脸仍取面积最大
            num_faces=5,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
        return _landmarker
    except Exception:
        _landmarker_failed = True
        _landmarker = None
        return None


def detect_all_face_landmarks(bgr: np.ndarray) -> list[np.ndarray]:
    """返回多张脸关键点列表（面积大→小）；失败空列表。"""
    if bgr is None or bgr.size == 0:
        return []
    landmarker = _get_landmarker()
    if landmarker is None:
        return []
    try:
        import mediapipe as mp

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return []
        out: list[np.ndarray] = []
        for face in result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in face], dtype=np.float32)
            if pts.shape[0] < 50:
                continue
            out.append(pts)
        if not out:
            return []

        def _area(pts: np.ndarray) -> float:
            x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
            y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
            return max(1.0, x1 - x0) * max(1.0, y1 - y0)

        out.sort(key=_area, reverse=True)
        return out
    except Exception:
        return []


def detect_face_landmarks(bgr: np.ndarray) -> np.ndarray | None:
    """返回主脸像素坐标 Nx2 (x,y)；失败 None。"""
    all_pts = detect_all_face_landmarks(bgr)
    return all_pts[0] if all_pts else None


def roi_from_landmarks(
    points: np.ndarray,
    shape: tuple[int, int],
    *,
    pad_x_frac: float = 0.10,
    pad_top_frac: float = 0.18,
    pad_bot_frac: float = 0.10,
) -> tuple[int, int, int, int]:
    """从关键点（优先脸廓）得外接框，略外扩含发际/下颚。"""
    h, w = shape
    idx = [i for i in FACE_OVAL_IDX if i < len(points)]
    use = points[idx] if idx else points
    x0 = float(use[:, 0].min())
    x1 = float(use[:, 0].max())
    y0 = float(use[:, 1].min())
    y1 = float(use[:, 1].max())
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    px = bw * pad_x_frac
    py_t = bh * pad_top_frac
    py_b = bh * pad_bot_frac
    xa = max(0, int(np.floor(x0 - px)))
    ya = max(0, int(np.floor(y0 - py_t)))
    xb = min(w, int(np.ceil(x1 + px)))
    yb = min(h, int(np.ceil(y1 + py_b)))
    return xa, ya, max(1, xb - xa), max(1, yb - ya)


def face_mask_from_landmarks(
    shape: tuple[int, int],
    points: np.ndarray,
    subject_mask: np.ndarray | None = None,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """脸廓凸包脸罩：只盖脸+短下巴垫，不吞发顶/衣领/肩。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    idx = [i for i in FACE_OVAL_IDX if i < len(points)]
    ear_i = [i for i in EAR_IDX if i < len(points)]
    use = points[idx] if len(idx) >= 8 else points
    # 耳点略外推，保证耳廓进脸罩
    extras: list[np.ndarray] = []
    if len(use) >= 3:
        cx = float(np.mean(use[:, 0]))
        for ei in ear_i:
            p = points[ei].astype(np.float32)
            dx = float(p[0] - cx)
            extras.append(np.array([p[0] + dx * 0.22, p[1]], dtype=np.float32))
            extras.append(p)
    if extras:
        use = np.vstack([use, np.stack(extras, axis=0)])
    pts = np.round(use).astype(np.int32)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    # 横向椭圆膨胀：扩耳、少吃发顶
    ear_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 5))
    mask = cv2.dilate(mask, ear_k, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    # 下巴尖下方短垫（相对脸高），禁止整 ROI 下半并入
    chin_idx = 152 if len(points) > 152 else int(np.argmax(use[:, 1]))
    chin_x, chin_y = float(points[chin_idx][0]), float(points[chin_idx][1])
    face_h = max(8.0, float(use[:, 1].max() - use[:, 1].min()))
    face_w = max(8.0, float(use[:, 0].max() - use[:, 0].min()))
    pad_y = int(face_h * 0.14)
    pad_x = int(face_w * 0.35)
    y0 = max(0, int(chin_y) - 2)
    y1 = min(h, int(chin_y) + pad_y)
    x0 = max(0, int(chin_x - pad_x))
    x1 = min(w, int(chin_x + pad_x))
    chin_pad = np.zeros((h, w), dtype=np.uint8)
    chin_pad[y0:y1, x0:x1] = 255
    if subject_mask is not None and subject_mask.shape[:2] == (h, w):
        chin_pad = cv2.bitwise_and(chin_pad, subject_mask)
    mask = cv2.bitwise_or(mask, chin_pad)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    if subject_mask is not None and subject_mask.shape[:2] == (h, w):
        mask = cv2.bitwise_and(mask, subject_mask)
    _ = roi  # 保留参数兼容调用方
    return mask


def detect_face_roi_mediapipe(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """成功返回 (roi, landmarks)，失败 None。"""
    pts = detect_face_landmarks(bgr)
    if pts is None:
        return None
    roi = roi_from_landmarks(pts, bgr.shape[:2])
    return roi, pts
