import math

import numpy as np
import pytest

from ml.geo import GPXError, Track, estimate_offset, haversine_m, move, parse_gpx
from tests.synthetic import gpx_line


def test_parse_and_interpolate(tmp_path):
    tr = Track.from_file(gpx_line(tmp_path / "a.gpx", speed_mps=10, seconds=30))
    assert len(tr.points) == 31 and tr.duration_s == 30
    assert tr.length_m() == pytest.approx(300, rel=0.01)
    lat, lon, speed, heading = tr.at(12.5)
    a = tr.points[0]
    assert haversine_m(a.lat, a.lon, lat, lon) == pytest.approx(125, rel=0.01)
    assert speed == pytest.approx(10, rel=0.02)
    assert heading == pytest.approx(90, abs=1)
    assert tr.at(-1) is None and tr.at(31) is None


def test_offset_shifts_the_clock(tmp_path):
    p = gpx_line(tmp_path / "a.gpx", speed_mps=10, seconds=60)
    plain, shifted = Track.from_file(p), Track.from_file(p, offset_s=20)
    assert shifted.at(0)[:2] == pytest.approx(plain.at(20)[:2])


def test_timezone_offsets_are_converted(tmp_path):
    utc = parse_gpx(gpx_line(tmp_path / "u.gpx").read_text())
    # same instants written in IST (+05:30) local time
    from datetime import datetime, timezone, timedelta
    ist = gpx_line(tmp_path / "i.gpx", t0=datetime(2026, 11, 1, 12, 0), tz_suffix="+05:30")
    assert parse_gpx(ist.read_text())[0].time == utc[0].time


def test_bad_gpx_is_rejected():
    with pytest.raises(GPXError):
        parse_gpx(b"<gpx><trk></trk></gpx>")
    with pytest.raises(GPXError):
        parse_gpx(b"not xml")


def test_move_and_haversine_agree():
    la, lo = move(12.87, 74.84, 45, 1000)
    assert haversine_m(12.87, 74.84, la, lo) == pytest.approx(1000, rel=1e-4)


def test_estimate_offset_finds_the_shift(tmp_path):
    # GPS speed with stops and starts; the video "saw" the same pattern starting 37 s later
    from tests.synthetic import gpx_line as _  # noqa: F401
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)
    speeds = np.clip(6 + 5 * np.sin(np.arange(240) / 9) + 3 * np.sin(np.arange(240) / 3.7), 0, None)
    pts, lat, lon = [], 12.87, 74.84
    for s, v in enumerate(speeds):
        pts.append(f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}"><time>{(t0 + timedelta(seconds=s)).isoformat()}</time></trkpt>')
        lat, lon = move(lat, lon, 10, float(v))
    p = tmp_path / "v.gpx"
    p.write_text(f'<gpx><trk><trkseg>{"".join(pts)}</trkseg></trk></gpx>')
    tr = Track.from_file(p)
    mt = np.arange(0, 120, 0.5)
    motion = np.interp(mt + 37, np.arange(240), speeds) * 3.1 + np.random.default_rng(1).normal(0, 0.4, len(mt))
    res = estimate_offset(mt, motion, tr, search=(0, 100))
    assert res["offset_s"] == pytest.approx(37, abs=1.0)
    assert res["confidence"] in ("high", "medium")
