"""Road names from OpenStreetMap's free Nominatim service, politely.

Nominatim's usage policy: at most 1 request per second, a real User-Agent, and cache
results. We round coordinates to about 10 m and keep every answer in SQLite, so the same
spot is never asked twice. In offline mode (or if the service is unreachable) the road
name is simply left empty and the report groups those potholes under "Unnamed road".
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

from .config import settings
from .db import GeocodeCache, get_session

log = logging.getLogger("potspot.geocode")
URL = "https://nominatim.openstreetmap.org/reverse"
_lock = threading.Lock()
_last_call = 0.0


def _key(lat: float, lon: float) -> str:
    return f"{lat:.4f},{lon:.4f}"


def cached(lat: float, lon: float) -> tuple[str | None, str | None] | None:
    with get_session() as s:
        row = s.get(GeocodeCache, _key(lat, lon))
        return (row.road, row.area) if row else None


def reverse(lat: float, lon: float, client: httpx.Client | None = None) -> tuple[str | None, str | None]:
    """(road name, area) for a point. Uses the cache first; waits to respect 1 request/second."""
    global _last_call
    hit = cached(lat, lon)
    if hit is not None:
        return hit
    if settings.offline:
        return None, None
    with _lock:
        wait = 1.05 - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            c = client or httpx.Client(timeout=10)
            r = c.get(URL, params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 17, "addressdetails": 1},
                      headers={"User-Agent": f"PotSpot/1.0 ({settings.contact})", "Accept-Language": "en"})
            r.raise_for_status()
            addr = r.json().get("address", {})
        except Exception as exc:  # network down, rate-limited, bad JSON: do not cache, try again later
            log.warning("reverse geocode failed: %s", type(exc).__name__)
            return None, None
    road = addr.get("road") or addr.get("pedestrian") or addr.get("residential") or addr.get("highway")
    area = addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district") or addr.get("city") or addr.get("town")
    with get_session() as s:
        s.merge(GeocodeCache(key=_key(lat, lon), road=road, area=area))
        s.commit()
    return road, area
