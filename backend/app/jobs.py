"""Background jobs: run the video pipeline for a trip, then merge its potholes into the map."""
from __future__ import annotations

import logging
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from ml.geo import Track, estimate_offset, offset_from_metadata
from ml.merge import Cluster, Sighting as MSighting, merge
from ml.pipeline import PipelineConfig, PipelineResult, run
from ml.severity import CameraModel

from . import geocode
from .config import settings
from .db import Pothole, Sighting, Trip, get_session

log = logging.getLogger("potspot.jobs")
# One video at a time: the model already uses every CPU core.
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="potspot-job")
PipelineFn = Callable[..., PipelineResult]
pipeline_fn: PipelineFn = run  # tests replace this with a fake


def _update(trip_id: str, **fields) -> None:
    with get_session() as s:
        t = s.get(Trip, trip_id)
        if t is None:
            return
        for k, v in fields.items():
            setattr(t, k, v)
        s.commit()


def choose_offset(video: Path, gpx: Path, mode: str, manual: float) -> tuple[float, str, dict]:
    """Decide how far into the GPX the video started. Returns (offset, method, details)."""
    track = Track.from_file(gpx)
    if mode == "manual":
        return manual, "manual", {}
    import cv2
    cap = cv2.VideoCapture(str(video))
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    meta = offset_from_metadata(video, track, duration)
    candidates = [meta["if_start"], meta["if_end"]] if meta else []
    from ml.motion import video_motion
    mt, motion = video_motion(video)
    lo = min([-60.0] + [c - 30 for c in candidates])
    hi = max([min(600.0, track.duration_s)] + [c + 30 for c in candidates])
    est = estimate_offset(mt, motion, track, search=(lo, hi), candidates=candidates)
    detail = {"motion": est, "metadata": meta}
    if est["confidence"] in ("high", "medium"):
        return est["offset_s"], f"motion match ({est['confidence']})", detail
    if meta and abs(meta["if_start"]) < track.duration_s:
        return meta["if_start"], "video creation time", detail
    return manual, "manual (auto-sync was not confident)", detail


def process_trip(trip_id: str, video: Path, gpx: Path, offset_mode: str, offset_s: float) -> None:
    trip_dir = settings.trips_dir / trip_id
    try:
        _update(trip_id, status="running", progress=0.01, message="Lining up video and GPS clocks")
        offset, method, detail = choose_offset(video, gpx, offset_mode, offset_s)
        track = Track.from_file(gpx, offset)
        _update(trip_id, offset_s=offset, offset_method=method, offset_detail=detail,
                started_at=track.start + timedelta(seconds=offset), message="Finding potholes")
        cfg = PipelineConfig(sample_fps=settings.sample_fps,
                             camera=CameraModel(height_m=settings.camera_height_m, horizon_frac=settings.horizon_frac))

        def progress(p: float, msg: str) -> None:
            _update(trip_id, progress=round(0.05 + 0.85 * p, 3), message=msg)

        result = pipeline_fn(video, trip_dir, gpx, offset, cfg, progress=progress)
        _update(trip_id, progress=0.92, message="Adding potholes to the map")
        placed = add_to_map(trip_id, result, track)
        _update(trip_id, status="done", progress=1.0, duration_s=result.duration_s, distance_km=result.distance_km,
                raw_detections=result.raw_detections, tracks=result.tracks_total, potholes_found=len(result.potholes),
                unplaced=len(result.potholes) - placed, route=result.route,
                annotated_video=f"{trip_id}/{result.annotated_video}" if result.annotated_video else None,
                message=f"Found {len(result.potholes)} potholes; {placed} placed on the map")
        threading.Thread(target=fill_road_names, daemon=True).start()
    except Exception as exc:
        log.error("trip %s failed: %s", trip_id, traceback.format_exc(limit=3))
        _update(trip_id, status="failed", message=f"Processing failed: {exc}")


def add_to_map(trip_id: str, result: PipelineResult, track: Track) -> int:
    """Turn this trip's potholes into sightings and merge them with what is already on the map."""
    new: list[MSighting] = []
    by_key = {}
    for p in result.potholes:
        if p.lat is None:
            continue  # seen outside the GPX time range: cannot place it
        key = f"{trip_id}:{p.track_id}"
        new.append(MSighting(p.lat, p.lon, p.severity, p.max_conf, trip_id=trip_id, key=key, width_m=p.width_m))
        by_key[key] = p
    with get_session() as s:
        existing = []
        for ph in s.scalars(select(Pothole)):
            c = Cluster(ph.lat, ph.lon, key=ph.id)
            c.sightings = [MSighting(x.lat, x.lon, x.severity, x.confidence, trip_id=x.trip_id, key=str(x.id))
                           for x in ph.sightings]
            existing.append(c)
        clusters = merge(new, settings.merge_radius_m, existing)
        for c in clusters:
            fresh = [x for x in c.sightings if x.key in by_key]
            if not fresh:
                continue
            ph = s.get(Pothole, c.key) if c.key else None
            if ph is None:
                ph = Pothole(lat=c.lat, lon=c.lon, severity=c.severity, confidence=c.confidence)
                s.add(ph)
                s.flush()
            for x in fresh:
                rec = by_key[x.key]
                seen = track.start + timedelta(seconds=track.offset_s + rec.last_t)
                s.add(Sighting(pothole_id=ph.id, trip_id=trip_id, lat=x.lat, lon=x.lon, seen_at=seen,
                               video_t=rec.best_t, severity=rec.severity, width_m=rec.width_m, confidence=rec.max_conf,
                               frames=rec.frames, speed_kmh=rec.speed_kmh, image=f"{trip_id}/{rec.image}"))
            s.flush()
            s.refresh(ph)
            refresh_pothole(ph)
        s.commit()
    return len(new)


def refresh_pothole(ph: Pothole) -> None:
    """Recompute a pothole's summary from its sightings."""
    sights = ph.sightings
    if not sights:
        return
    order = ("small", "medium", "large")
    ph.lat = sum(x.lat for x in sights) / len(sights)
    ph.lon = sum(x.lon for x in sights) / len(sights)
    ph.severity = max((x.severity for x in sights), key=order.index)
    widths = [x.width_m for x in sights if x.width_m is not None]
    ph.width_m = round(max(widths), 2) if widths else None
    ph.confidence = max(x.confidence for x in sights)
    ph.sightings_count = len(sights)
    ph.trips_count = len({x.trip_id for x in sights})
    times = [x.seen_at for x in sights if x.seen_at]
    ph.first_seen, ph.last_seen = (min(times), max(times)) if times else (None, None)
    best = max(sights, key=lambda x: (order.index(x.severity), x.confidence))
    ph.image = best.image


def remove_trip(trip_id: str) -> bool:
    with get_session() as s:
        t = s.get(Trip, trip_id)
        if t is None:
            return False
        touched = {x.pothole_id for x in t.sightings}
        s.delete(t)
        s.flush()
        for pid in touched:
            ph = s.get(Pothole, pid)
            if ph is None:
                continue
            s.refresh(ph)
            if not ph.sightings:
                s.delete(ph)
            else:
                refresh_pothole(ph)
        s.commit()
    shutil.rmtree(settings.trips_dir / trip_id, ignore_errors=True)
    return True


_fill_lock = threading.Lock()


def fill_road_names(limit: int = 200) -> int:
    """Look up road names for potholes that have none (1 request/second, cached)."""
    if settings.offline or not _fill_lock.acquire(blocking=False):
        return 0
    try:
        with get_session() as s:
            todo = [(p.id, p.lat, p.lon) for p in s.scalars(select(Pothole).where(Pothole.road.is_(None)).limit(limit))]
        n = 0
        for pid, lat, lon in todo:
            road, area = geocode.reverse(lat, lon)
            name = road or (f"Near {area}" if area else None)
            if name:
                with get_session() as s:
                    ph = s.get(Pothole, pid)
                    if ph:
                        ph.road = name
                        s.commit()
                        n += 1
        return n
    finally:
        _fill_lock.release()
