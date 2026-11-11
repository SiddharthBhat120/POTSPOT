"""Tiny synthetic road clips and GPX tracks for tests (no model needed)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 480, 270, 20


FOCAL_PX = 500.0     # synthetic camera focal length in pixels
CAM_H = 1.2          # camera height above the road (m)
HORIZON = 0.45       # horizon at 45% of the frame height
SPEED = 12.0         # m/s


def road_clip(path: Path, potholes: list[tuple[float, float, float]], seconds: float = 4.0,
              flicker_at: float | None = None) -> Path:
    """A grey road seen by a forward camera moving at SPEED, drawn with real perspective.

    Each pothole is (appear_s, lateral_m, width_m): it is 30 m ahead at `appear_s` and
    lies `lateral_m` to the right of the camera. Far away it is small and creeps slowly;
    close up it is big and moves fast, like real dashcam footage.
    `flicker_at` adds a one-frame dark blob (a false detection tracking should drop).
    """
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    hy = HORIZON * H
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = np.full((H, W, 3), 128, np.uint8)
        f[: int(hy)] = (200, 170, 120)  # sky
        f += rng.integers(0, 12, f.shape, dtype=np.uint8)
        for appear, lateral, width in potholes:
            z = 30.0 - SPEED * (t - appear)
            if t < appear or z < 1.0:
                continue
            y = hy + FOCAL_PX * CAM_H / z
            half_w = FOCAL_PX * width / z / 2
            half_h = max(1.5, half_w * CAM_H / z * 1.6)   # foreshortened depth on the road
            x = W / 2 + FOCAL_PX * lateral / z
            if y - half_h > H:
                continue
            # box bottom should sit on the ground row y
            cv2.ellipse(f, (int(x), int(y - half_h)), (int(half_w), int(half_h)), 0, 0, 360, (25, 25, 25), -1)
        if flicker_at is not None and abs(t - flicker_at) < 0.5 / FPS:
            cv2.circle(f, (60, 200), 12, (20, 20, 20), -1)
        vw.write(f)
    vw.release()
    return path


class BlobDetector:
    """Stand-in for YOLO: finds dark blobs in the road half of the frame."""

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g[: int(frame.shape[0] * 0.45)] = 255
        mask = (g < 60).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = [[x, y, x + w, y + h, 0.9, 0] for x, y, w, h, a in stats[1:] if a > 20]
        return np.array(out, np.float32).reshape(-1, 6)


class NoPrivacy:
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return np.zeros((0, 6), np.float32)


def gpx_line(path: Path, start=(12.8700, 74.8430), bearing_deg=90.0, speed_mps=8.0, seconds=60,
             t0=datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc), tz_suffix="Z") -> Path:
    import math
    from ml.geo import move
    lat, lon = start
    pts = []
    for s in range(seconds + 1):
        la, lo = move(lat, lon, bearing_deg, speed_mps * s)
        t = (t0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%S") + tz_suffix
        pts.append(f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><ele>20</ele><time>{t}</time></trkpt>')
    path.write_text('<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
                    f'<trk><trkseg>{"".join(pts)}</trkseg></trk></gpx>')
    return path
