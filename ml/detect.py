"""Detectors used by the pipeline, behind one small interface so tests can use fakes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "pothole.pt"
PRIVACY_WEIGHTS = ROOT / "models" / "privacy.pt"  # COCO-trained yolo11n, used only to find people and vehicles
# COCO class ids: person, bicycle, car, motorcycle, bus, truck
PRIVACY_CLASSES = [0, 1, 2, 3, 5, 7]


class Detector(Protocol):
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return an (N, 6) array of [x1, y1, x2, y2, confidence, class] in pixels."""
        ...


class YoloDetector:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, conf: float = 0.2, imgsz: int = 512,
                 classes: list[int] | None = None):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"Model weights not found: {weights}. Train with `python -m ml.train` "
                                    "or download models/pothole.pt from the repository.")
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.classes = conf, imgsz, classes

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=self.classes, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), np.float32)
        return r.boxes.data.cpu().numpy().astype(np.float32)


@lru_cache(maxsize=2)
def privacy_detector() -> YoloDetector | None:
    """People and vehicles detector for blurring. Returns None if the weights are missing."""
    if not PRIVACY_WEIGHTS.exists():
        return None
    return YoloDetector(PRIVACY_WEIGHTS, conf=0.25, imgsz=640, classes=PRIVACY_CLASSES)
