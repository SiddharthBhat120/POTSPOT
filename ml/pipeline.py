"""Phase 3: video in, one record per real pothole out.

    python -m ml.pipeline trip.mp4 --gpx trip.gpx --out data/trips/demo

Steps for each sampled frame (default 15 per second):
  1. The pothole model finds boxes.
  2. ByteTrack links boxes across frames, so the same pothole keeps one track id
     while it slides down the screen as you ride towards it.
  3. Each track keeps its sightings: time, box, confidence, sharpness.
Then for each track seen in at least `min_frames` frames:
  - best photo: the largest, sharpest, most confident sighting that is not cut off,
  - severity and estimated width from all sightings (ml/severity.py),
  - location from the GPX at the moment it was last seen (closest to the vehicle),
    pushed `lookahead_m` metres forward along the direction of travel,
  - people and vehicles in the saved photo are blurred (ml/privacy.py).
An annotated video with boxes and track ids is written so you can check the result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detect import Detector, YoloDetector
from .geo import Track, move
from .privacy import anonymise
from .severity import CameraModel, estimate

Progress = Callable[[float, str], None]


@dataclass
class PipelineConfig:
    sample_fps: float = 15.0    # 10 fps loses close potholes at city speeds (they jump past their own height)
    min_frames: int = 3
    conf: float = 0.2
    imgsz: int = 512
    track_high: float = 0.3
    track_low: float = 0.1
    new_track: float = 0.35
    track_buffer: int = 15      # sampled frames a lost pothole stays alive (1.5 s at 10 fps)
    match_thresh: float = 0.8
    lookahead_m: float = 4.0
    stitch_gap_s: float = 0.6   # join a track that ends with one that starts this soon after, nearby
    crop_context: float = 0.8   # extra space around the pothole in the saved photo
    annotate: bool = True
    blur_privacy: bool = True
    camera: CameraModel = field(default_factory=CameraModel)
    measure_width: bool = True  # False when the flat-road camera model does not fit the footage


@dataclass
class Observation:
    t: float
    box: tuple[float, float, float, float]
    conf: float
    sharpness: float


@dataclass
class PotholeRecord:
    track_id: int
    first_t: float
    last_t: float
    frames: int
    max_conf: float
    best_t: float
    best_box: tuple[float, float, float, float]
    severity: str
    width_m: float | None
    image: str
    lat: float | None = None
    lon: float | None = None
    speed_kmh: float | None = None


@dataclass
class PipelineResult:
    video: str
    duration_s: float
    frames_read: int
    frames_processed: int
    raw_detections: int
    tracks_total: int
    tracks_joined: int
    potholes: list[PotholeRecord]
    width: int
    height: int
    annotated_video: str | None
    route: dict | None
    distance_km: float | None
    offset_s: float
    privacy_blurred: int
    seconds: float

    def to_json(self) -> dict:
        d = asdict(self)
        d["potholes"] = [asdict(p) for p in self.potholes]
        return d


def make_tracker(cfg: PipelineConfig):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    args = IterableSimpleNamespace(tracker_type="bytetrack", track_high_thresh=cfg.track_high,
                                   track_low_thresh=cfg.track_low, new_track_thresh=cfg.new_track,
                                   track_buffer=cfg.track_buffer, match_thresh=cfg.match_thresh, fuse_score=True)
    return BYTETracker(args)


def sharpness(frame: np.ndarray, box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def stitch(tracks: dict[int, list[Observation]], max_gap_s: float, frame_w: int) -> tuple[dict[int, list[Observation]], int]:
    """Join broken tracks of the same pothole.

    ByteTrack matches a box to the next frame by overlap. A small, far pothole can jump
    further than its own height between frames, so the tracker drops it and starts a new
    id: one pothole would then be counted twice. Here, track B is joined onto track A when
      - B starts within `max_gap_s` after A ends (they never exist at the same time),
      - B's first box is close to where A was heading (A's last position + A's velocity),
      - B is not much smaller than A's last box (we drive towards potholes, so they grow).
    Returns the joined tracks and how many joins were made.
    """
    order = sorted(tracks, key=lambda k: tracks[k][0].t)
    merged = {k: list(v) for k, v in tracks.items()}
    joins = 0
    for b in order:
        if b not in merged:
            continue
        ob = merged[b]
        tb, bb = ob[0].t, ob[0].box
        best, best_d = None, None
        for a, oa in merged.items():
            if a == b or oa[-1].t >= tb or tb - oa[-1].t > max_gap_s:
                continue
            last = oa[-1]
            vx = vy = 0.0
            if len(oa) >= 2:
                prev = oa[-2]
                dt = max(1e-3, last.t - prev.t)
                (x1, y1), (x0, y0) = _center(last.box), _center(prev.box)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
            px, py = _center(last.box)
            px, py = px + vx * (tb - last.t), py + vy * (tb - last.t)
            bx, by = _center(bb)
            d = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
            size_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
            size_a = (last.box[2] - last.box[0]) * (last.box[3] - last.box[1])
            diag = ((bb[2] - bb[0]) ** 2 + (bb[3] - bb[1]) ** 2) ** 0.5
            if d <= max(diag, 0.05 * frame_w) and size_b >= 0.5 * size_a and (best_d is None or d < best_d):
                best, best_d = a, d
        if best is not None:
            merged[best].extend(ob)
            del merged[b]
            joins += 1
    return merged, joins


def best_observation(obs: list[Observation], w: int, h: int) -> Observation:
    def score(o: Observation) -> float:
        x1, y1, x2, y2 = o.box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        cut = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
        return o.conf * np.sqrt(area) * np.log1p(o.sharpness) * (0.4 if cut else 1.0)
    return max(obs, key=score)


def crop_with_context(frame: np.ndarray, box: tuple[float, float, float, float], ctx: float,
                      min_side: int = 220) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    half = max(bw, bh) * (1 + ctx) / 2
    half = max(half, min_side / 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - half * 1.3)), int(max(0, cy - half))
    c, d = int(min(w, cx + half * 1.3)), int(min(h, cy + half))
    return frame[b:d, a:c].copy(), (int(x1) - a, int(y1) - b, int(x2) - a, int(y2) - b)


def draw(frame: np.ndarray, tracks: np.ndarray, t: float, counted: int) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, tid, score, *_ in tracks:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (176, 72, 255), 2)
        label = f"#{int(tid)} {score:.2f}"
        cv2.putText(out, label, (int(x1), max(14, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (int(x1), max(14, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    banner = f"PotSpot  t={t:6.1f}s  potholes tracked: {counted}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(out, banner, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def to_h264(src: Path) -> Path:
    """Re-encode to H.264 so browsers can play it (OpenCV writes mp4v)."""
    if not shutil.which("ffmpeg"):
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=False)
    if r.returncode == 0 and dst.exists():
        src.unlink(missing_ok=True)
        dst.rename(src)
    return src


def run(video: str | Path, out_dir: str | Path, gpx: str | Path | None = None, offset_s: float = 0.0,
        cfg: PipelineConfig | None = None, detector: Detector | None = None,
        privacy: Detector | None = None, progress: Progress | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    started = time.time()
    video, out = Path(video), Path(out_dir)
    (out / "potholes").mkdir(parents=True, exist_ok=True)
    det = detector or YoloDetector(conf=cfg.conf, imgsz=cfg.imgsz)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / cfg.sample_fps))
    tracker = make_tracker(cfg)
    from ultralytics.engine.results import Boxes

    writer = None
    annotated = out / "annotated.mp4"
    if cfg.annotate:
        writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))

    tracks: dict[int, list[Observation]] = {}
    best_crops: dict[int, tuple] = {}  # track id -> (crop, pothole box inside crop, observation)
    raw = processed = idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        dets = det(frame)
        raw += len(dets)
        out_tr = tracker.update(Boxes(dets, (h, w)), frame) if len(dets) else tracker.update(
            Boxes(np.zeros((0, 6), np.float32), (h, w)), frame)
        for x1, y1, x2, y2, tid, score, *_ in out_tr:
            box = (float(x1), float(y1), float(x2), float(y2))
            o = Observation(t, box, float(score), sharpness(frame, box))
            tracks.setdefault(int(tid), []).append(o)
            # keep only a crop of the best sighting so far for each track (memory stays small)
            if best_observation(tracks[int(tid)], w, h) is o:
                best_crops[int(tid)] = (*crop_with_context(frame, box, cfg.crop_context), o)
        if writer is not None:
            counted = sum(1 for v in tracks.values() if len(v) >= cfg.min_frames)
            writer.write(draw(frame, out_tr, t, counted))
        processed += 1
        idx += 1
        if progress and total and processed % 10 == 0:
            progress(min(0.95, idx / total), f"Analysed {t:.0f}s of {total / fps:.0f}s")
    cap.release()
    if writer is not None:
        writer.release()
        to_h264(annotated)
    duration = (total or idx) / fps

    tracks, joins = stitch(tracks, cfg.stitch_gap_s, w)
    # a stitched track keeps the best crop of whichever part had the best sighting
    for tid, obs in tracks.items():
        best = best_observation(obs, w, h)
        for other, crop in list(best_crops.items()):
            if other != tid and crop[2] is best:
                best_crops[tid] = crop
    track = Track.from_file(gpx, offset_s) if gpx else None
    records: list[PotholeRecord] = []
    blurred = 0
    for tid, obs in sorted(tracks.items()):
        if len(obs) < cfg.min_frames:
            continue
        best = best_observation(obs, w, h)
        sev, width = estimate([o.box for o in obs], w, h, cfg.camera, cfg.measure_width)
        crop, local_box, _ = best_crops[tid]
        if cfg.blur_privacy:
            crop, n = anonymise(crop, privacy, protect=[local_box])
            blurred += n
        name = f"pothole_{tid:04d}.jpg"
        cv2.imwrite(str(out / "potholes" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec = PotholeRecord(tid, round(obs[0].t, 2), round(obs[-1].t, 2), len(obs), round(max(o.conf for o in obs), 3),
                            round(best.t, 2), tuple(round(v, 1) for v in best.box), sev, width, f"potholes/{name}")
        if track:
            pos = track.at(obs[-1].t)
            if pos:
                lat, lon, speed, heading = pos
                rec.lat, rec.lon = move(lat, lon, heading, cfg.lookahead_m)
                rec.lat, rec.lon = round(rec.lat, 7), round(rec.lon, 7)
                rec.speed_kmh = round(speed * 3.6, 1)
        records.append(rec)
    best_crops.clear()

    route = track.geojson_line(0, duration) if track else None
    dist = None
    if track:
        from .geo import haversine_m
        pos = [p for s in np.arange(0, duration + 1e-6, 1.0) if (p := track.at(float(s)))]
        dist = round(sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(pos, pos[1:])) / 1000, 3)
    result = PipelineResult(str(video.name), round(duration, 2), idx, processed, raw, len(tracks) + joins, joins, records, w, h,
                            "annotated.mp4" if cfg.annotate else None, route, dist, offset_s, blurred,
                            round(time.time() - started, 1))
    (out / "result.json").write_text(json.dumps(result.to_json(), indent=1))
    if progress:
        progress(1.0, f"Found {len(records)} potholes")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--gpx")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds after the GPX start that the video began")
    ap.add_argument("--out", default="data/trips/cli")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--no-annotate", action="store_true")
    a = ap.parse_args()
    cfg = PipelineConfig(sample_fps=a.fps, min_frames=a.min_frames, annotate=not a.no_annotate)
    r = run(a.video, a.out, a.gpx, a.offset, cfg, progress=lambda p, m: print(f"{p:5.0%} {m}", flush=True))
    print(f"{r.raw_detections} raw detections -> {r.tracks_total} tracks -> {len(r.potholes)} potholes "
          f"in {r.seconds}s. Results: {Path(a.out) / 'result.json'}")


if __name__ == "__main__":
    main()
