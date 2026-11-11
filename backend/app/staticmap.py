"""A static map image (PNG) for the PDF report: route lines and pothole dots on OSM tiles.

Tiles are fetched from tile.openstreetmap.org with a proper User-Agent and cached on disk
(OSM tile policy). If tiles cannot be loaded (offline), a plain background with a scale bar
is drawn instead, so the report still works.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from .config import settings

TILE = 256
COLORS = {"small": (233, 179, 0), "medium": (244, 112, 32), "large": (205, 38, 48)}
INK = (24, 24, 32)


def _xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x * TILE, y * TILE


def _tile(z: int, x: int, y: int, client: httpx.Client | None) -> Image.Image | None:
    if settings.offline or client is None:
        return None
    path = settings.cache_dir / "tiles" / f"{z}_{x}_{y}.png"
    if path.exists():
        return Image.open(path).convert("RGB")
    try:
        r = client.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                       headers={"User-Agent": f"PotSpot/1.0 ({settings.contact})"})
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def render(points: list[tuple[float, float, str]], routes: list[list[tuple[float, float]]],
           size: tuple[int, int] = (1000, 620)) -> tuple[bytes, bool]:
    """Returns PNG bytes and whether real map tiles were used."""
    W, H = size
    lats = [p[0] for p in points] + [c[0] for r in routes for c in r]
    lons = [p[1] for p in points] + [c[1] for r in routes for c in r]
    if not lats:
        lats, lons = [12.87], [74.84]
    lat0, lat1, lon0, lon1 = min(lats), max(lats), min(lons), max(lons)
    z = 18
    while z > 3:
        x0, y1 = _xy(lat0, lon0, z)
        x1, y0 = _xy(lat1, lon1, z)
        if x1 - x0 < W * 0.8 and y1 - y0 < H * 0.8:
            break
        z -= 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = cx - W / 2, cy - H / 2
    img = Image.new("RGB", (W, H), (236, 233, 224))
    used_tiles = False
    client = None if settings.offline else httpx.Client(timeout=8)
    try:
        failures = 0
        for tx in range(int(left // TILE), int((left + W) // TILE) + 1):
            for ty in range(int(top // TILE), int((top + H) // TILE) + 1):
                t = _tile(z, tx, ty, client) if failures < 2 else None
                if t is None:
                    failures += 1
                    continue
                img.paste(t, (int(tx * TILE - left), int(ty * TILE - top)))
                used_tiles = True
    finally:
        if client:
            client.close()
    if used_tiles:  # soften the base map so markers stand out
        img = Image.blend(img, Image.new("RGB", (W, H), (245, 245, 245)), 0.35)
    d = ImageDraw.Draw(img)
    for r in routes:
        pts = [(_xy(la, lo, z)[0] - left, _xy(la, lo, z)[1] - top) for la, lo in r]
        if len(pts) > 1:
            d.line(pts, fill=(43, 59, 214), width=4, joint="curve")
    order = {"small": 0, "medium": 1, "large": 2}
    for la, lo, sev in sorted(points, key=lambda p: order.get(p[2], 0)):
        x, y = _xy(la, lo, z)
        x, y = x - left, y - top
        rad = 6 + 2 * order.get(sev, 0)
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=COLORS.get(sev, INK), outline=(255, 255, 255), width=2)
    # scale bar
    m_per_px = 156543.03392 * math.cos(math.radians((lat0 + lat1) / 2)) / 2 ** z
    for nice in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if nice / m_per_px > 90:
            break
    px = nice / m_per_px
    d.rectangle((14, H - 34, 14 + px + 70, H - 10), fill=(255, 255, 255))
    d.line((20, H - 18, 20 + px, H - 18), fill=INK, width=3)
    font = ImageFont.load_default()
    d.text((26 + px, H - 26), f"{nice if nice < 1000 else nice // 1000} {'m' if nice < 1000 else 'km'}", fill=INK, font=font)
    note = "Map data (c) OpenStreetMap contributors" if used_tiles else "Map tiles unavailable (offline)"
    tw = d.textlength(note, font=font)
    d.rectangle((W - tw - 16, H - 22, W, H), fill=(255, 255, 255))
    d.text((W - tw - 8, H - 18), note, fill=INK, font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), used_tiles
