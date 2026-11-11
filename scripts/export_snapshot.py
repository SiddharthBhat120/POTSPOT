"""Export the current map as a static, read-only demo site (for GitHub Pages).

    python scripts/export_snapshot.py --out site

Copies frontend/dist and adds snapshot/: potholes, details, routes, trips, the photos they
use, and a PDF report. The frontend notices there is no API and reads the snapshot instead.
Run it with the server's data folder (POTSPOT_DATA) pointing at the trips you want to show.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    a = ap.parse_args()
    c = TestClient(app)
    potholes = c.get("/api/potholes").json()
    if not potholes:
        sys.exit("No potholes to export. Process a ride first.")
    trips = [t for t in c.get("/api/trips").json() if t["status"] == "done"]
    details = {p["id"]: c.get(f"/api/potholes/{p['id']}").json() for p in potholes}
    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "frontend" / "dist", out)
    snap = out / "snapshot"
    (snap / "media").mkdir(parents=True)
    used = {p["image"] for p in potholes if p["image"]} | {s["image"] for d in details.values() for s in d["sightings_list"]}
    for rel in used:
        dst = snap / "media" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(settings.trips_dir / rel, dst)
    (snap / "snapshot.json").write_text(json.dumps({
        "potholes": potholes, "details": details, "routes": c.get("/api/routes").json(), "trips": trips,
    }))
    (snap / "report.pdf").write_bytes(c.get("/api/report.pdf").content)
    (out / ".nojekyll").write_text("")
    print(f"Static demo written to {out} ({len(potholes)} potholes, {len(used)} photos)")


if __name__ == "__main__":
    main()
