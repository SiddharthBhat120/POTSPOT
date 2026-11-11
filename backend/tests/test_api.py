"""API tests with the pipeline replaced by a fake (no model, no video decoding)."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml.geo import Track, move
from ml.pipeline import PipelineResult, PotholeRecord
from tests.synthetic import gpx_line

from app import jobs
from app.main import app

client = TestClient(app)


def fake_pipeline(positions_m):
    """A pipeline stand-in that 'finds' potholes at given metres along the GPX line."""
    def run(video, out_dir, gpx, offset_s, cfg, progress=None, **_):
        out = Path(out_dir)
        (out / "potholes").mkdir(parents=True, exist_ok=True)
        track = Track.from_file(gpx, offset_s)
        recs = []
        for i, (metres, sev) in enumerate(positions_m):
            t = metres / 10.0 - offset_s  # gpx_line default speed 10 m/s set below
            if i == 0:
                import cv2, numpy as np
                cv2.imwrite(str(out / "potholes" / f"p{i}.jpg"), np.full((60, 80, 3), 90, np.uint8))
            else:
                (out / "potholes" / f"p{i}.jpg").write_bytes(b"\xff\xd8\xff")  # a broken file must not break the report
            lat, lon, speed, heading = track.at(t)
            recs.append(PotholeRecord(i + 1, t - 1, t, 12, 0.8, t - 0.5, (1, 2, 3, 4), sev, 0.5, f"potholes/p{i}.jpg",
                                      lat, lon, speed * 3.6))
        if progress:
            progress(0.5, "halfway")
        return PipelineResult("v.mp4", 60.0, 900, 900, 400, len(recs) + 3, 1, recs, 1280, 720, None,
                              track.geojson_line(0, 60), 0.6, offset_s, 0, 1.0)
    return run


def wait(trip_id):
    for _ in range(100):
        t = client.get(f"/api/trips/{trip_id}").json()
        if t["status"] in ("done", "failed"):
            return t
        time.sleep(0.05)
    raise AssertionError("trip did not finish")


def upload(tmp_path, name, offset=0.0):
    gpx = gpx_line(tmp_path / f"{name}.gpx", speed_mps=10, seconds=120)
    files = {"video": ("ride.mp4", b"not really a video", "video/mp4"),
             "gpx": ("ride.gpx", gpx.read_bytes(), "application/gpx+xml")}
    r = client.post("/api/trips", files=files, data={"name": name, "offset_mode": "manual", "offset_s": str(offset)})
    assert r.status_code == 202, r.text
    return r.json()


def test_upload_process_list_and_merge_across_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "pipeline_fn", fake_pipeline([(100, "small"), (300, "large")]))
    t1 = wait(upload(tmp_path, "first ride")["id"])
    assert t1["status"] == "done" and t1["potholes_found"] == 2 and t1["distance_km"] == 0.6
    # second ride over the same road: one pothole again (now medium), one new
    monkeypatch.setattr(jobs, "pipeline_fn", fake_pipeline([(102, "medium"), (500, "small")]))
    t2 = wait(upload(tmp_path, "second ride")["id"])
    assert t2["status"] == "done"

    ps = client.get("/api/potholes").json()
    assert len(ps) == 3
    twice = [p for p in ps if p["sightings"] == 2]
    assert len(twice) == 1 and twice[0]["trips"] == 2 and twice[0]["severity"] == "medium"

    st = client.get("/api/stats").json()
    assert st["potholes"] == 3 and st["trips"] == 2 and st["km"] == 1.2 and st["seen_on_multiple_trips"] == 1
    assert client.get("/api/potholes?severity=large").json()[0]["severity"] == "large"

    detail = client.get(f"/api/potholes/{twice[0]['id']}").json()
    assert [s["trip_name"] for s in detail["sightings_list"]] == ["first ride", "second ride"]

    # bbox filter: only potholes inside a small box around the first one
    lat, lon = twice[0]["lat"], twice[0]["lon"]
    box = f"{lon - 0.0002},{lat - 0.0002},{lon + 0.0002},{lat + 0.0002}"
    assert len(client.get(f"/api/potholes?bbox={box}").json()) == 1

    routes = client.get("/api/routes").json()
    assert len(routes["features"]) == 2

    pdf = client.get("/api/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    # deleting the second trip removes its new pothole and its sighting of the shared one
    assert client.delete(f"/api/trips/{t2['id']}").json()["deleted"]
    ps = client.get("/api/potholes").json()
    assert len(ps) == 2 and all(p["sightings"] == 1 for p in ps)
    assert next(p for p in ps if p["id"] == twice[0]["id"])["severity"] == "small"
    client.delete(f"/api/trips/{t1['id']}")
    assert client.get("/api/potholes").json() == []


def test_validation_errors(tmp_path):
    gpx = gpx_line(tmp_path / "g.gpx").read_bytes()
    bad_video = {"video": ("ride.txt", b"x", "text/plain"), "gpx": ("g.gpx", gpx, "application/gpx+xml")}
    assert client.post("/api/trips", files=bad_video).status_code == 415
    bad_gpx = {"video": ("ride.mp4", b"x", "video/mp4"), "gpx": ("g.gpx", b"<gpx/>", "application/gpx+xml")}
    assert client.post("/api/trips", files=bad_gpx).status_code == 422
    assert client.get("/api/potholes?bbox=1,2,3").status_code == 400
    assert client.get("/api/trips/nope").status_code == 404
    # the client collapses "../" before sending, so also try the encoded form the server really sees
    assert "root:" not in client.get("/media/../../etc/passwd").text
    assert client.get("/media/..%2F..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_failed_pipeline_is_reported(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model file missing")
    monkeypatch.setattr(jobs, "pipeline_fn", boom)
    t = wait(upload(tmp_path, "broken")["id"])
    assert t["status"] == "failed" and "model file missing" in t["message"]
    client.delete(f"/api/trips/{t['id']}")
