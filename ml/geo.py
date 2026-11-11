"""Phase 4: GPS. Parse GPX, interpolate a position for any video time, and line up the clocks.

Clock rule used everywhere: the video started `offset_s` seconds after the first GPX point.
So a moment `t` seconds into the video happened at  gpx_start + offset_s + t.

How to get offset_s:
  1. You know it (e.g. you pressed record 4 s after starting the GPS app): pass it in.
  2. `offset_from_metadata`: compare the video's creation time with the GPX start.
     Phones write this time in different ways, so treat it as a first guess.
  3. `estimate_offset`: compare how fast the road moves in the video (optical flow)
     with the GPS speed over time, and pick the shift where the two curves line up best.
"""
from __future__ import annotations

import json
import math
import subprocess
import xml.etree.ElementTree as ET
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

EARTH_R = 6_371_008.8


@dataclass
class TrackPoint:
    time: datetime  # timezone-aware UTC
    lat: float
    lon: float
    ele: float | None = None
    speed_mps: float = 0.0


class GPXError(ValueError):
    pass


def _parse_time(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def move(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    """Point `dist_m` metres from (lat, lon) in direction `bearing` degrees."""
    d, b = dist_m / EARTH_R, math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def parse_gpx(data: str | bytes) -> list[TrackPoint]:
    """Read every <trkpt> (or <rtept>) with a time. Namespaces and GPX 1.0/1.1 both work."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise GPXError(f"Not a valid GPX file: {exc}") from exc
    pts: list[TrackPoint] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag not in ("trkpt", "rtept"):
            continue
        t_el = next((c for c in el if c.tag.rsplit("}", 1)[-1] == "time"), None)
        e_el = next((c for c in el if c.tag.rsplit("}", 1)[-1] == "ele"), None)
        if t_el is None or not (t_el.text or "").strip():
            continue
        try:
            pts.append(TrackPoint(_parse_time(t_el.text), float(el.attrib["lat"]), float(el.attrib["lon"]),
                                  float(e_el.text) if e_el is not None and e_el.text else None))
        except (KeyError, ValueError):
            continue
    if len(pts) < 2:
        raise GPXError("The GPX file needs at least two points with timestamps.")
    pts.sort(key=lambda p: p.time)
    # drop duplicate timestamps (some loggers write the same second twice)
    clean = [pts[0]]
    for p in pts[1:]:
        if p.time > clean[-1].time:
            clean.append(p)
    compute_speeds(clean)
    return clean


def compute_speeds(pts: list[TrackPoint], window: int = 2) -> None:
    """Speed at each point from the distance to neighbours `window` points away (smooths GPS jitter)."""
    for i, p in enumerate(pts):
        a, b = pts[max(0, i - window)], pts[min(len(pts) - 1, i + window)]
        dt = (b.time - a.time).total_seconds()
        p.speed_mps = haversine_m(a.lat, a.lon, b.lat, b.lon) / dt if dt > 0 else 0.0


class Track:
    """A GPX track you can ask "where was I at video time t?"."""

    def __init__(self, points: list[TrackPoint], offset_s: float = 0.0):
        self.points = points
        self.offset_s = offset_s
        self.start = points[0].time
        self._secs = [(p.time - self.start).total_seconds() for p in points]

    @classmethod
    def from_file(cls, path: str | Path, offset_s: float = 0.0) -> "Track":
        return cls(parse_gpx(Path(path).read_bytes()), offset_s)

    @property
    def duration_s(self) -> float:
        return self._secs[-1]

    def length_m(self) -> float:
        return sum(haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(self.points, self.points[1:]))

    def at(self, video_t: float) -> tuple[float, float, float, float] | None:
        """(lat, lon, speed m/s, bearing) at `video_t` seconds into the video, or None if outside the track."""
        s = self.offset_s + video_t
        if s < 0 or s > self._secs[-1]:
            return None
        i = max(1, bisect_left(self._secs, s))
        a, b = self.points[i - 1], self.points[i]
        sa, sb = self._secs[i - 1], self._secs[i]
        f = 0.0 if sb == sa else (s - sa) / (sb - sa)
        lat = a.lat + (b.lat - a.lat) * f
        lon = a.lon + (b.lon - a.lon) * f
        speed = a.speed_mps + (b.speed_mps - a.speed_mps) * f
        return lat, lon, speed, self.heading_near(i)

    def heading_near(self, i: int, min_dist: float = 5.0) -> float:
        """Direction of travel around point i, using points far enough apart to be stable."""
        j, k = i - 1, i
        while k < len(self.points) - 1 and haversine_m(self.points[j].lat, self.points[j].lon,
                                                       self.points[k].lat, self.points[k].lon) < min_dist:
            k += 1
        while j > 0 and haversine_m(self.points[j].lat, self.points[j].lon,
                                    self.points[k].lat, self.points[k].lon) < min_dist:
            j -= 1
        return bearing_deg(self.points[j].lat, self.points[j].lon, self.points[k].lat, self.points[k].lon)

    def speed_series(self, step: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """GPS speed resampled every `step` seconds from the GPX start."""
        t = np.arange(0, self._secs[-1], step)
        return t, np.interp(t, self._secs, [p.speed_mps for p in self.points])

    def geojson_line(self, start_t: float = 0.0, end_t: float | None = None) -> dict:
        """The part of the route driven during the video, as a GeoJSON LineString."""
        lo = self.offset_s + start_t
        hi = self._secs[-1] if end_t is None else self.offset_s + end_t
        coords = [[p.lon, p.lat] for p, s in zip(self.points, self._secs) if lo - 1 <= s <= hi + 1]
        return {"type": "LineString", "coordinates": coords}


def video_creation_time(video: str | Path) -> datetime | None:
    """creation_time from the video's metadata, via ffprobe (if installed)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video)],
                             capture_output=True, text=True, timeout=30, check=False).stdout
        tag = json.loads(out or "{}").get("format", {}).get("tags", {}).get("creation_time")
        return _parse_time(tag) if tag else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def offset_from_metadata(video: str | Path, track: Track, video_duration_s: float) -> dict | None:
    """Two guesses, because phones store either the start or the end of recording as creation_time."""
    ct = video_creation_time(video)
    if ct is None:
        return None
    as_start = (ct - track.start).total_seconds()
    as_end = (ct - timedelta(seconds=video_duration_s) - track.start).total_seconds()
    return {"creation_time": ct.isoformat(), "if_start": round(as_start, 1), "if_end": round(as_end, 1)}


def estimate_offset(motion_t: np.ndarray, motion: np.ndarray, track: Track,
                    search: tuple[float, float] = (-60.0, 300.0), step: float = 0.5,
                    candidates: list[float] | None = None) -> dict:
    """Find the offset where video motion and GPS speed rise and fall together.

    Uses Pearson correlation between the two curves (scale does not matter, only shape).
    Returns the best offset, its correlation, and how clearly it beats the runner-up.
    A flat ride at constant speed has no shape to match, so `confidence` will be low:
    then use the manual offset instead.
    """
    gt, gs = track.speed_series(1.0)
    if len(motion) < 10 or len(gt) < 10:
        return {"offset_s": 0.0, "correlation": 0.0, "confidence": "low", "reason": "too little data"}
    m = (motion - motion.mean()) / (motion.std() + 1e-9)
    scores: list[tuple[float, float]] = []
    offsets = list(np.arange(search[0], search[1] + step, step)) + list(candidates or [])
    for off in offsets:
        s = np.interp(motion_t + off, gt, gs, left=np.nan, right=np.nan)
        ok = ~np.isnan(s)
        if ok.sum() < max(10, 0.6 * len(m)):
            continue
        sv = s[ok]
        if sv.std() < 1e-6:
            continue
        r = float(np.corrcoef(m[ok], (sv - sv.mean()) / sv.std())[0, 1])
        scores.append((float(off), r))
    if not scores:
        return {"offset_s": 0.0, "correlation": 0.0, "confidence": "low", "reason": "no overlap between video and GPX"}
    scores.sort(key=lambda x: x[1], reverse=True)
    best_off, best_r = scores[0]
    runner = next((r for o, r in scores if abs(o - best_off) > 10), 0.0)
    conf = "high" if best_r > 0.6 and best_r - runner > 0.15 else "medium" if best_r > 0.4 else "low"
    return {"offset_s": round(best_off, 1), "correlation": round(best_r, 3), "runner_up": round(runner, 3),
            "confidence": conf}
