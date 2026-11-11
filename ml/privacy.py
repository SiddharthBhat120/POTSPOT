"""Blur people and vehicles in any image PotSpot saves.

Faces and number plates both sit inside a person or vehicle box, so blurring those
whole boxes hides them without needing separate face and plate models. The cost is
that some road next to a parked car is blurred too, which is acceptable: the pothole
itself is on the road surface, rarely under a person or vehicle box.

Limits: small or far-away people and vehicles can be missed by the detector. Every
saved image is still a crop around a pothole, which keeps most of the scene out.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detect import Detector, privacy_detector


def blur_regions(img: np.ndarray, boxes: np.ndarray, protect: list[tuple[int, int, int, int]] | None = None,
                 pad: float = 0.08) -> tuple[np.ndarray, int]:
    """Strongly blur each box. Areas in `protect` (the pothole) are left sharp."""
    out = img.copy()
    h, w = img.shape[:2]
    n = 0
    for x1, y1, x2, y2, *_ in boxes:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
        c, d = min(w, int(x2 + px)), min(h, int(y2 + py))
        if c - a < 4 or d - b < 4:
            continue
        roi = out[b:d, a:c]
        k = max(15, (min(roi.shape[:2]) // 3) | 1)
        small = cv2.resize(roi, (max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)), interpolation=cv2.INTER_AREA)
        out[b:d, a:c] = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR), (k, k), 0)
        n += 1
    for x1, y1, x2, y2 in protect or []:
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out, n


def anonymise(img: np.ndarray, detector: Detector | None = None,
              protect: list[tuple[int, int, int, int]] | None = None) -> tuple[np.ndarray, int]:
    det = detector if detector is not None else privacy_detector()
    if det is None:
        return img, 0
    return blur_regions(img, det(img), protect)
