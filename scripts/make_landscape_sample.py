"""生成可复现的扁平化风景样例（非写真），便于照片域实验。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "golden-landscape.jpg"


def main() -> None:
    h, w = 720, 960
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 天空渐变 → 稍后 EFR 会压扁；样例本身略带渐变以触发照片域
    for y in range(h // 2):
        t = y / max(h // 2 - 1, 1)
        img[y] = (int(220 - 40 * t), int(180 - 30 * t), int(90 + 40 * t))  # BGR 偏蓝天
    img[h // 2 :] = (60, 140, 70)  # 草地
    # 远山
    pts = np.array([[0, h // 2], [200, 280], [450, 320], [700, 260], [960, h // 2]], np.int32)
    cv2.fillPoly(img, [pts], (90, 110, 100))
    # 太阳
    cv2.circle(img, (780, 120), 55, (40, 210, 240), -1)
    # 树干+树冠
    cv2.rectangle(img, (160, 380), (190, 520), (30, 70, 90), -1)
    cv2.circle(img, (175, 350), 70, (40, 120, 50), -1)
    cv2.imwrite(str(OUT), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
