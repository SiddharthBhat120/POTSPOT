"""PotSpot API. Run with:  python -m uvicorn app.main:app --app-dir backend --port 8002"""
from __future__ import annotations

import contextlib
import logging
import mimetypes
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func, select

from ml.geo import GPXError, parse_gpx

from . import jobs, report
from .config import ROOT, settings
from .db import Pothole, Sighting, Trip, get_session

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("video/mp4", ".mp4")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.WARNING)

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    _recover()
    yield


app = FastAPI(title="PotSpot", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5175", "http://127.0.0.1:5175"],
                   allow_methods=["*"], allow_headers=["*"])
DIST = ROOT / "frontend" / "dist"
DEMO = ROOT / "demo"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".webm"}
settings.trips_dir.mkdir(parents=True, exist_ok=True)


# ---------- response models ----------
class TripOut(BaseModel):
    id: str
    name: str
    video_name: str
    created_at: datetime
    started_at: datetime | None
    status: str
    progress: float
    message: str
    offset_s: float
    offset_method: str
    duration_s: float | None
    distance_km: float | None
    raw_detections: int | None
    tracks: int | None
    potholes_found: int | None
    unplaced: int | None
    annotated_video: str | None
    is_demo: bool

    model_config = {"from_attributes": True}


class PotholeOut(BaseModel):
    id: int
    lat: float
    lon: float
    severity: str
    width_m: float | None
    confidence: float
    sightings: int
    trips: int
    first_seen: datetime | None
    last_seen: datetime | None
    image: str | None
    road: str | None


class SightingOut(BaseModel):
    trip_id: str
    trip_name: str
    seen_at: datetime | None
    video_t: float
    severity: str
    width_m: float | None
    confidence: float
    frames: int
    speed_kmh: float | None
    image: str
    lat: float
    lon: float


class PotholeDetail(PotholeOut):
    sightings_list: list[SightingOut]


def pothole_out(p: Pothole) -> PotholeOut:
    return PotholeOut(id=p.id, lat=p.lat, lon=p.lon, severity=p.severity, width_m=p.width_m,
                      confidence=round(p.confidence, 3), sightings=p.sightings_count, trips=p.trips_count,
                      first_seen=p.first_seen, last_seen=p.last_seen, image=p.image, road=p.road)


# ---------- filtering shared by list, stats and report ----------
def query_potholes(severity: str | None, bbox: str | None, since: str | None, until: str | None,
                   trip: str | None = None) -> list[Pothole]:
    with get_session() as s:
        q = select(Pothole)
        if severity:
            sev = [x for x in severity.split(",") if x in ("small", "medium", "large")]
            if sev:
                q = q.where(Pothole.severity.in_(sev))
        if bbox:
            try:
                w, so, e, n = (float(x) for x in bbox.split(","))
            except ValueError as exc:
                raise HTTPException(400, "bbox must be minLon,minLat,maxLon,maxLat") from exc
            q = q.where(Pothole.lon >= w, Pothole.lon <= e, Pothole.lat >= so, Pothole.lat <= n)
        if since:
            q = q.where(Pothole.last_seen >= datetime.fromisoformat(since))
        if until:
            q = q.where(Pothole.first_seen <= datetime.fromisoformat(until))
        if trip:
            q = q.where(Pothole.id.in_(select(Sighting.pothole_id).where(Sighting.trip_id == trip)))
        rows = list(s.scalars(q.order_by(Pothole.id)))
        for p in rows:
            _ = p.sightings  # load before the session closes
        return rows


def done_trips(trip: str | None = None) -> list[Trip]:
    with get_session() as s:
        q = select(Trip).where(Trip.status == "done")
        if trip:
            q = q.where(Trip.id == trip)
        return list(s.scalars(q))


# ---------- endpoints ----------
@app.get("/api/health")
def health() -> dict:
    model = ROOT / "models" / "pothole.pt"
    return {"ok": True, "model": model.exists(), "demo_available": (DEMO / "demo_trip.mp4").exists(),
            "offline": settings.offline, "merge_radius_m": settings.merge_radius_m}


def _start(trip: Trip, video: Path, gpx: Path, offset_mode: str, offset_s: float) -> None:
    with get_session() as s:
        s.add(trip)
        s.commit()
    jobs.executor.submit(jobs.process_trip, trip.id, video, gpx, offset_mode, offset_s)


@app.post("/api/trips", response_model=TripOut, status_code=202)
async def upload_trip(video: UploadFile = File(...), gpx: UploadFile = File(...), name: str = Form(""),
                      offset_mode: str = Form("auto"), offset_s: float = Form(0.0)) -> TripOut:
    """Upload a road video and the GPX track recorded on the same ride. Processing runs in the background."""
    vext = Path(video.filename or "").suffix.lower()
    if vext not in VIDEO_EXT:
        raise HTTPException(415, f"Video must be one of: {', '.join(sorted(VIDEO_EXT))}")
    gpx_bytes = await gpx.read()
    try:
        pts = parse_gpx(gpx_bytes)
    except GPXError as exc:
        raise HTTPException(422, str(exc)) from exc
    if offset_mode not in ("auto", "manual"):
        raise HTTPException(422, "offset_mode must be 'auto' or 'manual'")
    trip_id = uuid.uuid4().hex[:10]
    d = settings.trips_dir / trip_id
    d.mkdir(parents=True)
    vpath = d / f"video{vext}"
    size = 0
    with vpath.open("wb") as fh:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb * (1 << 20):
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"Video is larger than {settings.max_upload_mb} MB.")
            fh.write(chunk)
    gpath = d / "track.gpx"
    gpath.write_bytes(gpx_bytes)
    trip = Trip(id=trip_id, name=name.strip() or f"Ride on {pts[0].time:%d %b %Y}", video_name=video.filename or "video",
                gpx_name=gpx.filename or "track.gpx", offset_s=offset_s, offset_method=offset_mode)
    _start(trip, vpath, gpath, offset_mode, offset_s)
    return TripOut.model_validate(trip)


@app.post("/api/demo", response_model=TripOut, status_code=202)
def run_demo() -> TripOut:
    """Process the bundled demo ride (simulated camera motion over held-out test photos, simulated GPX)."""
    if not (DEMO / "demo_trip.mp4").exists():
        raise HTTPException(404, "Demo files are missing. Run `python demo/make_demo_trip.py`.")
    trip_id = uuid.uuid4().hex[:10]
    d = settings.trips_dir / trip_id
    d.mkdir(parents=True)
    shutil.copy2(DEMO / "demo_trip.mp4", d / "video.mp4")
    shutil.copy2(DEMO / "demo_trip.gpx", d / "track.gpx")
    trip = Trip(id=trip_id, name="Demo ride (simulated)", video_name="demo_trip.mp4", gpx_name="demo_trip.gpx",
                offset_s=0.0, offset_method="manual", is_demo=1)
    _start(trip, d / "video.mp4", d / "track.gpx", "manual", 0.0)
    return TripOut.model_validate(trip)


@app.get("/api/trips", response_model=list[TripOut])
def list_trips() -> list[TripOut]:
    with get_session() as s:
        return [TripOut.model_validate(t) for t in s.scalars(select(Trip).order_by(Trip.created_at.desc()))]


@app.get("/api/trips/{trip_id}", response_model=TripOut)
def get_trip(trip_id: str) -> TripOut:
    with get_session() as s:
        t = s.get(Trip, trip_id)
        if not t:
            raise HTTPException(404, "Trip not found.")
        return TripOut.model_validate(t)


@app.delete("/api/trips/{trip_id}")
def delete_trip(trip_id: str) -> dict:
    with get_session() as s:
        t = s.get(Trip, trip_id)
        if t and t.status in ("queued", "running"):
            raise HTTPException(409, "This trip is still being processed.")
    return {"deleted": jobs.remove_trip(trip_id)}


@app.get("/api/routes")
def routes(trip: str | None = None) -> dict:
    """Driven routes as a GeoJSON FeatureCollection."""
    feats = [{"type": "Feature", "properties": {"trip": t.id, "name": t.name, "demo": bool(t.is_demo)},
              "geometry": t.route} for t in done_trips(trip) if t.route]
    return {"type": "FeatureCollection", "features": feats}


@app.get("/api/potholes", response_model=list[PotholeOut])
def list_potholes(severity: str | None = None, bbox: str | None = None, since: str | None = None,
                  until: str | None = None, trip: str | None = None) -> list[PotholeOut]:
    return [pothole_out(p) for p in query_potholes(severity, bbox, since, until, trip)]


@app.get("/api/potholes/{pid}", response_model=PotholeDetail)
def pothole_detail(pid: int) -> PotholeDetail:
    with get_session() as s:
        p = s.get(Pothole, pid)
        if not p:
            raise HTTPException(404, "Pothole not found.")
        sights = [SightingOut(trip_id=x.trip_id, trip_name=x.trip.name, seen_at=x.seen_at, video_t=x.video_t,
                              severity=x.severity, width_m=x.width_m, confidence=round(x.confidence, 3), frames=x.frames,
                              speed_kmh=x.speed_kmh, image=x.image, lat=x.lat, lon=x.lon)
                  for x in sorted(p.sightings, key=lambda x: x.seen_at or datetime.min)]
        return PotholeDetail(**pothole_out(p).model_dump(), sightings_list=sights)


@app.get("/api/stats")
def stats(severity: str | None = None, bbox: str | None = None, since: str | None = None,
          until: str | None = None, trip: str | None = None) -> dict:
    rows = query_potholes(severity, bbox, since, until, trip)
    trips = done_trips(trip)
    km = round(sum(t.distance_km or 0 for t in trips), 2)
    by = {k: sum(1 for p in rows if p.severity == k) for k in ("small", "medium", "large")}
    return {"potholes": len(rows), "by_severity": by, "km": km,
            "per_km": round(len(rows) / km, 2) if km else None, "trips": len(trips),
            "seen_on_multiple_trips": sum(1 for p in rows if p.trips_count > 1)}


@app.get("/api/report.pdf")
def pdf_report(severity: str | None = None, bbox: str | None = None, since: str | None = None,
               until: str | None = None, trip: str | None = None) -> Response:
    rows = query_potholes(severity, bbox, since, until, trip)
    if not rows:
        raise HTTPException(404, "No potholes match these filters, so there is nothing to report.")
    parts = [f"Severity: {severity}" if severity else "All severities"]
    if since or until:
        parts.append(f"Seen between {since or 'start'} and {until or 'now'}")
    if bbox:
        parts.append("Map area filter applied")
    pdf = report.build(rows, done_trips(trip), ". ".join(parts) + ".")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="potspot-report-{datetime.now():%Y%m%d}.pdf"'})


@app.get("/media/{path:path}", include_in_schema=False)
def media(path: str) -> FileResponse:
    target = (settings.trips_dir / path).resolve()
    if settings.trips_dir.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found.")
    if target.suffix.lower() not in (".jpg", ".png", ".mp4"):
        raise HTTPException(403, "Not available.")  # never serve the uploaded raw video or GPX
    if target.name.startswith("video."):
        raise HTTPException(403, "Not available.")
    return FileResponse(target)


def _recover() -> None:
    """Trips that were running when the server stopped are marked failed, so the UI does not wait forever."""
    with get_session() as s:
        for t in s.scalars(select(Trip).where(Trip.status.in_(["queued", "running"]))):
            t.status, t.message = "failed", "The server restarted while this trip was processing. Upload it again."
        s.commit()


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
