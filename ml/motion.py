"""How fast the road is moving in the video, over time (used to line up video and GPS clocks)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_motion(video: str | Path, fps: float = 2.0, width: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Median optical-flow magnitude in the lower half of the frame (the road), `fps` times a second.

    Faster riding means the road texture streams past faster, so this curve rises and falls
    with speed. Its units are pixels, not km/h: only its shape is compared with GPS speed.
    """
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(vfps / fps))
    times, mags = [], []
    prev = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (width, int(h * width / w)))
            g = cv2.cvtColor(small[small.shape[0] // 2:], cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mags.append(float(np.median(np.linalg.norm(flow, axis=2))))
                times.append(idx / vfps)
            prev = g
        idx += 1
    cap.release()
    m = np.array(mags)
    if len(m) >= 5:  # smooth bumps
        m = np.convolve(m, np.ones(3) / 3, mode="same")
    return np.array(times), m
