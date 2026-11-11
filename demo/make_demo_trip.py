"""Build the demo trip: a short "dashcam" clip, a matching GPX track, and the ground truth.

    python demo/make_demo_trip.py

Why a simulated clip: the real test is a video you record yourself (see README). Until
then, this clip lets the whole pipeline run end to end with a known answer.

How it is made: 12 photos from the held-out TEST split (the model never trained on them)
are each shown for 3.5 seconds while a virtual camera moves forward: the view zooms in
towards the horizon, so potholes grow and slide down and out of the frame, as they do
when you ride over them. Because each photo comes with hand-drawn boxes, we know exactly
how many distinct potholes the clip contains: that is the ground truth for tracking.

The GPX track is simulated too (a 30 km/h ride with a stop). Its coordinates are
illustrative and do not follow a real road. Your own GPX replaces it.

Credit: photos from the pothole dataset by Chitale et al. (IVCNZ 2020),
https://github.com/jaygala24/pothole-detection (MIT licence).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.geo import move  # noqa: E402

TEST_IMG = ROOT / "data" / "yolo" / "images" / "test"
TEST_LBL = ROOT / "data" / "yolo" / "labels" / "test"
OUT_W, OUT_H, FPS = 1280, 720, 30
SECONDS_PER_PHOTO = 3.5
ZOOM_END = 2.2           # how far the camera "moves in" per photo
HORIZON = 0.42           # zoom centre, as a fraction of the photo height
N_PHOTOS = 12
MIN_VISIBLE_S = 0.4      # a labelled pothole counts only if visible this long at a usable size
MIN_BOX_PX = 18
START = (12.8641, 74.8417)   # illustrative start point near Mangaluru
T0 = datetime(2026, 11, 8, 7, 30, tzinfo=timezone.utc)


def pick_photos() -> list[Path]:
    """Landscape road-level photos whose potholes are in the lower part of the frame."""
    good = []
    for img_path in sorted(TEST_IMG.glob("*.jpg"), key=lambda p: int(p.stem.split("-")[1])):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes = read_boxes(img_path)
        if w < h * 1.25 or not boxes or len(boxes) > 8:
            continue
        if min(y for _, y, _, _ in boxes) < 0.38:
            continue
        good.append(img_path)
    step = max(1, len(good) // N_PHOTOS)
    return good[::step][:N_PHOTOS]


def read_boxes(img_path: Path) -> list[tuple[float, float, float, float]]:
    lbl = TEST_LBL / f"{img_path.stem}.txt"
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append(tuple(float(v) for v in p[1:]))
    return out


def view_window(w: int, h: int, zoom: float) -> tuple[float, float, float, float]:
    """The part of the photo the camera sees at a given zoom (16:9, centred on the horizon point)."""
    base_w = min(w, h * OUT_W / OUT_H)
    base_h = base_w * OUT_H / OUT_W
    vx, vy = w / 2, h * HORIZON
    # at zoom 1 the window fills the photo; it shrinks towards (vx, vy)
    x0 = (w - base_w) / 2
    y0 = max(0.0, min(h - base_h, h - base_h))  # bottom-aligned at the start: road in view
    ww, hh = base_w / zoom, base_h / zoom
    x = vx + (x0 - vx) / zoom
    y = vy + (y0 - vy) / zoom
    return x, y, ww, hh


def zoom_at(k: float) -> float:
    return 1 + (ZOOM_END - 1) * (k ** 1.6)  # accelerating approach


def to_photo(box: tuple[float, float, float, float], w: int, h: int, k: float) -> tuple[float, float, float, float]:
    """Map a box in the output video (pixels) back to the source photo (pixels) at progress k."""
    x, y, ww, hh = view_window(w, h, zoom_at(k))
    x1, y1, x2, y2 = box
    return (x + x1 * ww / OUT_W, y + y1 * hh / OUT_H, x + x2 * ww / OUT_W, y + y2 * hh / OUT_H)


def render_photo(img: np.ndarray, boxes, writer, frame_index: int, truth: list) -> int:
    h, w = img.shape[:2]
    n = int(SECONDS_PER_PHOTO * FPS)
    visible = [0] * len(boxes)
    for i in range(n):
        k = i / (n - 1)
        zoom = zoom_at(k)
        x, y, ww, hh = view_window(w, h, zoom)
        m = np.float32([[OUT_W / ww, 0, -x * OUT_W / ww], [0, OUT_H / hh, -y * OUT_H / hh]])
        frame = cv2.warpAffine(img, m, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)
        for j, (cx, cy, bw, bh) in enumerate(boxes):
            x1 = ((cx - bw / 2) * w - x) * OUT_W / ww
            x2 = ((cx + bw / 2) * w - x) * OUT_W / ww
            y1 = ((cy - bh / 2) * h - y) * OUT_H / hh
            y2 = ((cy + bh / 2) * h - y) * OUT_H / hh
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(OUT_W, x2), min(OUT_H, y2)
            inside = (cx2 - cx1) * (cy2 - cy1) / max(1.0, (x2 - x1) * (y2 - y1))
            if inside > 0.6 and min(cx2 - cx1, cy2 - cy1) >= MIN_BOX_PX:
                visible[j] += 1
    t0 = frame_index / FPS
    for j, v in enumerate(visible):
        if v / FPS >= MIN_VISIBLE_S:
            truth.append({"photo_start_s": round(t0, 2), "photo_end_s": round(t0 + n / FPS, 2),
                          "box": boxes[j], "visible_s": round(v / FPS, 2)})
    return frame_index + n


def make_gpx(duration_s: float, path: Path) -> None:
    """30 km/h with a slow-down in the middle, gently curving, one point per second."""
    lat, lon = START
    bearing = 62.0
    pts = []
    for s in range(int(duration_s) + 15):
        speed = 8.3 if not 14 <= s <= 20 else 3.0
        bearing += 0.6 if s < 25 else -0.9
        t = (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append(f'   <trkpt lat="{lat:.7f}" lon="{lon:.7f}"><ele>18.0</ele><time>{t}</time></trkpt>')
        lat, lon = move(lat, lon, bearing, speed)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<gpx version="1.1" creator="PotSpot demo (simulated track)" xmlns="http://www.topografix.com/GPX/1/1">\n'
                    ' <trk><name>PotSpot demo ride (simulated)</name><trkseg>\n' + "\n".join(pts) +
                    '\n </trkseg></trk>\n</gpx>\n')


def main() -> None:
    if not TEST_IMG.exists():
        sys.exit("Run `python -m ml.data.download` and `python -m ml.data.prepare` first.")
    out_dir = ROOT / "demo"
    raw = out_dir / "_demo_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    truth: list[dict] = []
    photos = pick_photos()
    idx = 0
    geometry = []
    for p in photos:
        img = cv2.imread(str(p))
        geometry.append({"photo": p.name, "w": img.shape[1], "h": img.shape[0], "start_frame": idx,
                         "boxes": read_boxes(p)})
        idx = render_photo(img, read_boxes(p), writer, idx, truth)
    writer.release()
    video = out_dir / "demo_trip.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-metadata", f"creation_time={T0.isoformat()}", str(video)], check=True)
        raw.unlink()
    else:
        raw.rename(video)
    duration = idx / FPS
    make_gpx(duration, out_dir / "demo_trip.gpx")
    (out_dir / "demo_ground_truth.json").write_text(json.dumps({
        "note": "Simulated forward motion over 12 held-out test photos; GPX is simulated. Offset between video and GPX: 0 s.",
        "photos": geometry, "fps": FPS, "frames_per_photo": int(SECONDS_PER_PHOTO * FPS),
        "duration_s": round(duration, 2), "offset_s": 0.0,
        "unique_potholes": len(truth), "potholes": truth,
        "credit": "Photos: Chitale et al., IVCNZ 2020, github.com/jaygala24/pothole-detection (MIT)",
    }, indent=1))
    print(f"demo_trip.mp4: {duration:.1f}s, {len(photos)} photos, {len(truth)} labelled potholes visible; "
          f"{video.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
