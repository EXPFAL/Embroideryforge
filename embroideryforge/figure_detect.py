"""MediaPipe Hand / Pose Landmarker → 手罩与臂引导（人物+动作）。

对齐 face_detect：成熟 Tasks 模型优先，失败回退空结果由调用方色域兜底。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

_ROOT = Path(__file__).resolve().parents[1]
_hand_landmarker: Any | None = None
_hand_failed = False
_pose_landmarker: Any | None = None
_pose_failed = False

# Pose：左右腕 / 肘 / 肩（MediaPipe Pose 33）
POSE_LWIRST, POSE_RWIRST = 15, 16
POSE_LELBOW, POSE_RELBOW = 13, 14
POSE_LSHOULDER, POSE_RSHOULDER = 11, 12

# Hand：指骨连线（MediaPipe 21 点）
HAND_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
    (5, 9, 13, 17),
)


def _models_dir() -> Path:
    return _ROOT / "models"


def ensure_model(url: str, filename: str, *, download: bool = True) -> Path | None:
    p = _models_dir() / filename
    if p.is_file() and p.stat().st_size > 1000:
        return p
    if not download:
        return None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".task.download")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(p)
        if p.is_file() and p.stat().st_size > 1000:
            return p
    except Exception:
        return None
    return None


def _get_hand_landmarker() -> Any | None:
    global _hand_landmarker, _hand_failed
    if _hand_landmarker is not None:
        return _hand_landmarker
    if _hand_failed:
        return None
    mp_path = ensure_model(HAND_MODEL_URL, "hand_landmarker.task", download=True)
    if mp_path is None:
        _hand_failed = True
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(mp_path))
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            # 持物/前伸手易漏检，略放宽
            min_hand_detection_confidence=0.22,
            min_hand_presence_confidence=0.22,
            min_tracking_confidence=0.22,
        )
        _hand_landmarker = vision.HandLandmarker.create_from_options(options)
        return _hand_landmarker
    except Exception:
        _hand_failed = True
        _hand_landmarker = None
        return None


def _get_pose_landmarker() -> Any | None:
    global _pose_landmarker, _pose_failed
    if _pose_landmarker is not None:
        return _pose_landmarker
    if _pose_failed:
        return None
    mp_path = ensure_model(POSE_MODEL_URL, "pose_landmarker_lite.task", download=True)
    if mp_path is None:
        _pose_failed = True
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(mp_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        return _pose_landmarker
    except Exception:
        _pose_failed = True
        _pose_landmarker = None
        return None


def _to_mp_image(bgr: np.ndarray):
    import mediapipe as mp

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_hand_landmarks(bgr: np.ndarray) -> list[np.ndarray]:
    """每只手 Nx2 像素坐标；失败空列表。"""
    if bgr is None or bgr.size == 0:
        return []
    landmarker = _get_hand_landmarker()
    if landmarker is None:
        return []
    try:
        h, w = bgr.shape[:2]
        result = landmarker.detect(_to_mp_image(bgr))
        hands: list[np.ndarray] = []
        for hand in result.hand_landmarks or []:
            pts = np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32)
            if pts.shape[0] >= 15:
                hands.append(pts)
        return hands
    except Exception:
        return []


def detect_pose_landmarks(bgr: np.ndarray) -> np.ndarray | None:
    """33x2 像素坐标；失败 None。"""
    if bgr is None or bgr.size == 0:
        return None
    landmarker = _get_pose_landmarker()
    if landmarker is None:
        return None
    try:
        h, w = bgr.shape[:2]
        result = landmarker.detect(_to_mp_image(bgr))
        if not result.pose_landmarks:
            return None
        pose = result.pose_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in pose], dtype=np.float32)
        if pts.shape[0] < 17:
            return None
        return pts
    except Exception:
        return None


def hand_mask_from_landmarks(
    shape: tuple[int, int],
    hands: list[np.ndarray],
    subject_mask: np.ndarray | None = None,
    *,
    dilate: int = 14,
) -> np.ndarray:
    """双手关键点凸包 + 膨胀 → 手掌/手指罩。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in hands:
        if pts is None or len(pts) < 3:
            continue
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    if dilate > 0 and np.any(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate | 1, dilate | 1))
        mask = cv2.dilate(mask, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if subject_mask is not None and subject_mask.shape[:2] == (h, w):
        mask = cv2.bitwise_and(mask, subject_mask)
    return mask


def forearm_guide_mask(
    shape: tuple[int, int],
    pose: np.ndarray | None,
    subject_mask: np.ndarray | None = None,
    *,
    thickness: int = 28,
) -> np.ndarray:
    """肩-肘-腕粗线带：引导袖臂区（服饰 tatami 排除手之后的臂段）。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if pose is None or len(pose) <= max(POSE_LWIRST, POSE_RWIRST):
        return mask
    chains = (
        (POSE_LSHOULDER, POSE_LELBOW, POSE_LWIRST),
        (POSE_RSHOULDER, POSE_RELBOW, POSE_RWIRST),
    )
    for a, b, c in chains:
        pts = np.round(pose[[a, b, c]]).astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness=thickness)
    if subject_mask is not None and subject_mask.shape[:2] == (h, w):
        mask = cv2.bitwise_and(mask, subject_mask)
    return mask


def prop_seed_near_hands(
    shape: tuple[int, int],
    hands: list[np.ndarray],
    *,
    radius: int = 36,
    pose: np.ndarray | None = None,
) -> np.ndarray:
    """手心/指尖外扩圆：道具（枪）常贴手，作暗色道具搜索先验。"""
    h, w = shape
    seed = np.zeros((h, w), dtype=np.uint8)
    for pts in hands:
        if pts is None or len(pts) < 5:
            continue
        # 腕=0，指尖=4,8,12,16,20
        tips = [0, 4, 8, 12, 16, 20]
        use = [i for i in tips if i < len(pts)]
        for i in use:
            x, y = int(round(float(pts[i, 0]))), int(round(float(pts[i, 1])))
            cv2.circle(seed, (x, y), radius, 255, thickness=-1)
        # 掌心：关键点均值
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        cv2.circle(seed, (int(cx), int(cy)), int(radius * 1.15), 255, thickness=-1)
        # 指尖前方：枪管常伸出手指
        if len(pts) > 8:
            palm = np.mean(pts[[0, 5, 9, 13, 17]], axis=0)
            tip = pts[8]
            fwd = tip - palm
            nrm = float(np.linalg.norm(fwd)) + 1e-3
            ahead = tip + fwd / nrm * (radius * 1.4)
            cv2.circle(seed, (int(ahead[0]), int(ahead[1])), int(radius * 1.3), 255, -1)
    # 无手关键点时：用姿态腕+肘→腕方向外扩
    if pose is not None and len(pose) > max(POSE_LWIRST, POSE_RWIRST):
        for elbow_i, wrist_i in (
            (POSE_LELBOW, POSE_LWIRST),
            (POSE_RELBOW, POSE_RWIRST),
        ):
            wrist = pose[wrist_i]
            elbow = pose[elbow_i]
            fwd = wrist - elbow
            nrm = float(np.linalg.norm(fwd)) + 1e-3
            ahead = wrist + fwd / nrm * (radius * 1.6)
            cv2.circle(seed, (int(wrist[0]), int(wrist[1])), int(radius * 1.2), 255, -1)
            cv2.circle(seed, (int(ahead[0]), int(ahead[1])), int(radius * 1.4), 255, -1)
    return seed


def finger_bone_segments(hands: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    """指骨折线，供 running 勾出手形。"""
    polys: list[list[tuple[float, float]]] = []
    for pts in hands:
        if pts is None or len(pts) < 21:
            continue
        for chain in HAND_FINGER_CHAINS:
            poly = [(float(pts[i, 0]), float(pts[i, 1])) for i in chain if i < len(pts)]
            if len(poly) >= 2:
                polys.append(poly)
    return polys


def _dedupe_hands(hands: list[np.ndarray], *, min_dist: float = 28.0) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for h in hands:
        c = np.mean(h, axis=0)
        if any(float(np.linalg.norm(c - np.mean(k, axis=0))) < min_dist for k in kept):
            continue
        kept.append(h)
    return kept


def detect_hands_wrist_crops(
    bgr: np.ndarray,
    pose: np.ndarray | None,
    *,
    crop_scale: float = 0.28,
    upscale: float = 2.0,
) -> list[np.ndarray]:
    """在姿态腕附近裁剪放大再检手，补全图漏检的持物手。"""
    if pose is None or bgr is None:
        return []
    h, w = bgr.shape[:2]
    side = int(max(96, min(h, w) * crop_scale))
    found: list[np.ndarray] = []
    for elbow_i, wrist_i in (
        (POSE_LELBOW, POSE_LWIRST),
        (POSE_RELBOW, POSE_RWIRST),
    ):
        if wrist_i >= len(pose):
            continue
        wrist = pose[wrist_i]
        elbow = pose[elbow_i] if elbow_i < len(pose) else wrist
        fwd = wrist - elbow
        nrm = float(np.linalg.norm(fwd)) + 1e-3
        center = wrist + fwd / nrm * (side * 0.12)
        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        if x1 - x0 < 48 or y1 - y0 < 48:
            continue
        crop = bgr[y0:y1, x0:x1]
        if upscale != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=upscale,
                fy=upscale,
                interpolation=cv2.INTER_LINEAR,
            )
        local = detect_hand_landmarks(crop)
        for pts in local:
            mapped = pts.astype(np.float32).copy()
            mapped[:, 0] = mapped[:, 0] / upscale + x0
            mapped[:, 1] = mapped[:, 1] / upscale + y0
            found.append(mapped)
    return found


def scale_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    if mask.shape[:2] == shape:
        return mask
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def scale_points(pts: np.ndarray | None, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray | None:
    if pts is None or len(pts) == 0:
        return pts
    sy = dst_hw[0] / max(src_hw[0], 1)
    sx = dst_hw[1] / max(src_hw[1], 1)
    out = pts.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def detect_figure_geometry(
    bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
    *,
    out_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """一次性手+姿态；可在全分辨率检测后缩放到 work 尺寸。"""
    pose = detect_pose_landmarks(bgr)
    hands = detect_hand_landmarks(bgr)
    # 腕部裁剪二次检测，尽量凑齐双手
    extra = detect_hands_wrist_crops(bgr, pose)
    if extra:
        hands = _dedupe_hands(list(hands) + extra, min_dist=max(24.0, bgr.shape[1] * 0.03))
    shape = bgr.shape[:2]
    hand_m = hand_mask_from_landmarks(
        shape, hands, subject_mask=subject_mask, dilate=11
    )
    forearm = forearm_guide_mask(shape, pose, subject_mask=subject_mask)
    prop_seed = prop_seed_near_hands(shape, hands, radius=42, pose=pose)
    if subject_mask is not None and subject_mask.shape[:2] == shape:
        prop_seed = cv2.bitwise_and(prop_seed, subject_mask)
    finger_polys = finger_bone_segments(hands)
    dst = out_shape or shape
    if dst != shape:
        hand_m = scale_mask(hand_m, dst)
        forearm = scale_mask(forearm, dst)
        prop_seed = scale_mask(prop_seed, dst)
        hands = [
            p
            for p in (scale_points(h, shape, dst) for h in hands)
            if p is not None
        ]
        pose = scale_points(pose, shape, dst)
        sy = dst[0] / max(shape[0], 1)
        sx = dst[1] / max(shape[1], 1)
        finger_polys = [
            [(x * sx, y * sy) for x, y in poly] for poly in finger_polys
        ]
    return {
        "source": {
            "hands": "mediapipe" if hands else "none",
            "pose": "mediapipe" if pose is not None else "none",
            "wrist_crop": bool(extra),
        },
        "hands": hands,
        "pose": pose,
        "hand_mask": hand_m,
        "forearm_mask": forearm,
        "prop_seed": prop_seed,
        "finger_polys": finger_polys,
        "n_hands": len(hands),
    }
