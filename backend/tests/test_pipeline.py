import json

import cv2
import numpy as np

from ml.pipeline import Observation, PipelineConfig, run, stitch
from ml.privacy import blur_regions
from tests.synthetic import BlobDetector, NoPrivacy, gpx_line, road_clip


def test_one_pothole_one_record_and_flicker_dropped(tmp_path):
    # a 0.3 m pothole 0.6 m left, then a 0.9 m pothole 1 m right, plus a one-frame false blob
    clip = road_clip(tmp_path / "c.mp4", [(0.2, -0.6, 0.3), (1.4, 1.0, 0.9)], seconds=4.5, flicker_at=1.0)
    r = run(clip, tmp_path / "out", cfg=PipelineConfig(sample_fps=15, min_frames=3),
            detector=BlobDetector(), privacy=NoPrivacy())
    assert r.raw_detections > 25               # seen in many frames...
    assert len(r.potholes) == 2                # ...but counted once each; the flicker is dropped
    first, second = sorted(r.potholes, key=lambda p: p.first_t)
    assert first.frames >= 10 and second.frames >= 10
    # flat-road size estimate matches the true widths when the camera model matches
    assert first.severity == "small" and second.severity == "large"
    assert first.width_m == __import__("pytest").approx(0.3, rel=0.25)
    assert second.width_m == __import__("pytest").approx(0.9, rel=0.25)
    for p in r.potholes:
        assert (tmp_path / "out" / p.image).exists()
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 1000
    saved = json.loads((tmp_path / "out" / "result.json").read_text())
    assert len(saved["potholes"]) == 2


def test_geotagging_uses_last_sighting_plus_lookahead(tmp_path):
    from ml.geo import Track, haversine_m
    clip = road_clip(tmp_path / "c.mp4", [(0.2, 0.5, 0.6)], seconds=3)
    gpx = gpx_line(tmp_path / "t.gpx", speed_mps=10, seconds=60)
    r = run(clip, tmp_path / "out", gpx=gpx, offset_s=20, cfg=PipelineConfig(lookahead_m=4.0),
            detector=BlobDetector(), privacy=NoPrivacy())
    p = r.potholes[0]
    tr = Track.from_file(gpx, 20)
    lat, lon, *_ = tr.at(p.last_t)
    assert haversine_m(lat, lon, p.lat, p.lon) == __import__("pytest").approx(4.0, abs=0.2)
    assert p.speed_kmh == __import__("pytest").approx(36, abs=1)
    assert r.distance_km == __import__('pytest').approx(0.03, abs=0.002)  # 3 s at 10 m/s
    assert r.route["type"] == "LineString"


def test_privacy_blur_keeps_pothole_sharp():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    out, n = blur_regions(img, np.array([[0, 0, 300, 200, 0.9, 2]], np.float32), protect=[(100, 80, 160, 140)])
    assert n == 1
    assert np.array_equal(out[80:140, 100:160], img[80:140, 100:160])     # pothole untouched
    assert out[:60].std() < img[:60].std() / 3                             # the rest is blurred


def test_stitch_joins_a_broken_track_but_not_neighbours():
    o = lambda t, x, y, s: Observation(t, (x, y, x + s, y + s / 2), 0.8, 10.0)
    a = [o(0.0, 100, 100, 10), o(0.1, 100, 110, 12)]            # moving down 100 px/s
    b = [o(0.3, 100, 130, 16), o(0.4, 100, 140, 18)]            # continues where a was heading
    c = [o(0.3, 400, 120, 16), o(0.4, 400, 130, 18)]            # a different pothole on the right
    joined, n = stitch({1: a, 2: b, 3: c}, max_gap_s=0.6, frame_w=640)
    assert n == 1 and set(joined) == {1, 3} and len(joined[1]) == 4
