import pytest

from ml.geo import move
from ml.merge import Sighting, merge
from ml.severity import CameraModel, estimate, width_m

W, H = 1280, 720
CAM = CameraModel(height_m=1.2, horizon_frac=0.45)


def test_same_pothole_same_width_near_and_far():
    # A 0.5 m pothole: far away it is small and high, close up it is big and low.
    horizon = 0.45 * H
    far_below, near_below = 60, 300
    far = (600, horizon + far_below - 10, 600 + 0.5 / 1.2 * far_below, horizon + far_below)
    near = (500, horizon + near_below - 50, 500 + 0.5 / 1.2 * near_below, horizon + near_below)
    assert width_m(far, W, H, CAM) == pytest.approx(0.5, rel=0.01)
    assert width_m(near, W, H, CAM) == pytest.approx(0.5, rel=0.01)


def test_cut_off_and_horizon_boxes_are_skipped():
    assert width_m((0, 500, 200, 600), W, H, CAM) is None        # touches left edge
    assert width_m((400, 600, 600, H), W, H, CAM) is None        # touches bottom
    assert width_m((400, 320, 420, 330), W, H, CAM) is None      # almost at the horizon


def test_severity_labels():
    horizon = 0.45 * H
    def box(width_m_, below=200):
        return (500, horizon + below - 20, 500 + width_m_ / 1.2 * below, horizon + below)
    assert estimate([box(0.2)], W, H, CAM) == ("small", 0.2)
    assert estimate([box(0.5)], W, H, CAM)[0] == "medium"
    assert estimate([box(1.1), box(1.0), box(0.9)], W, H, CAM) == ("large", 1.0)  # median


def _s(lat, lon, trip, sev="small"):
    return Sighting(lat, lon, sev, 0.8, trip_id=trip)


def test_merge_across_trips_within_radius():
    a = (12.8700, 74.8430)
    b = move(*a, 30, 5)    # 5 m away: same pothole, other trip
    c = move(*a, 90, 30)   # 30 m away: a different pothole
    clusters = merge([_s(*a, "t1"), _s(*b, "t2", "large"), _s(*c, "t2")], radius_m=8)
    assert len(clusters) == 2
    big = max(clusters, key=lambda c: c.count)
    assert big.count == 2 and big.trips == 2 and big.severity == "large"


def test_same_trip_never_merges():
    a = (12.8700, 74.8430)
    b = move(*a, 0, 3)
    assert len(merge([_s(*a, "t1"), _s(*b, "t1")], radius_m=8)) == 2


def test_no_chaining_along_a_bad_road():
    pts = [move(12.87, 74.84, 90, 6 * i) for i in range(6)]  # potholes every 6 m over 30 m
    clusters = merge([_s(la, lo, f"t{i}") for i, (la, lo) in enumerate(pts)], radius_m=8)
    assert len(clusters) >= 3


def test_merge_into_existing():
    a = (12.8700, 74.8430)
    first = merge([_s(*a, "t1")])
    again = merge([_s(*move(*a, 0, 2), "t2")], existing=first)
    assert len(again) == 1 and again[0].count == 2
