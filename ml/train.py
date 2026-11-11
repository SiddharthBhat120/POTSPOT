"""Fine-tune a small YOLO model on the pothole dataset.

    python -m ml.train                         # CPU-friendly defaults used for the shipped weights
    python -m ml.train --model yolo11s.pt --imgsz 640 --epochs 100 --device 0   # on a GPU (Colab)
    python -m ml.train --resume                # continue an interrupted run

The best weights are copied to models/pothole.pt, which the app uses.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo" / "dataset.yaml"
RUNS = ROOT / "runs"
NAME = "pothole"

# Augmentation suited to road video. Ultralytics applies these on the fly.
ROAD_AUG = dict(
    hsv_h=0.015,   # small colour shifts (different cameras)
    hsv_s=0.6,     # washed-out vs saturated
    hsv_v=0.5,     # bright sun to shade
    degrees=3.0,   # phone not perfectly level
    translate=0.1,
    scale=0.5,     # potholes near and far
    shear=1.0,
    perspective=0.0005,
    fliplr=0.5,    # roads look the same mirrored
    flipud=0.0,    # but never upside down
    mosaic=1.0,
    mixup=0.05,
    close_mosaic=5,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    last = RUNS / NAME / "weights" / "last.pt"
    if args.resume:
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=args.workers, project=str(RUNS), name=NAME,
                    exist_ok=True, patience=12, seed=120, deterministic=False, cos_lr=True, **ROAD_AUG)
    best = RUNS / NAME / "weights" / "best.pt"
    out = ROOT / "models" / "pothole.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    print(f"Best weights copied to {out}")


if __name__ == "__main__":
    main()
