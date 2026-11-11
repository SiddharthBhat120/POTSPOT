"""SQLite tables via SQLAlchemy 2.0."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import settings


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Trip(Base):
    __tablename__ = "trips"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    video_name: Mapped[str] = mapped_column(String(300))
    gpx_name: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # when the ride began (GPX time)
    status: Mapped[str] = mapped_column(String(16), default="queued")             # queued, running, done, failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="Waiting to start")
    offset_s: Mapped[float] = mapped_column(Float, default=0.0)
    offset_method: Mapped[str] = mapped_column(String(32), default="manual")
    offset_detail: Mapped[dict | None] = mapped_column(JSON)
    duration_s: Mapped[float | None] = mapped_column(Float)
    distance_km: Mapped[float | None] = mapped_column(Float)
    raw_detections: Mapped[int | None] = mapped_column(Integer)
    tracks: Mapped[int | None] = mapped_column(Integer)
    potholes_found: Mapped[int | None] = mapped_column(Integer)
    unplaced: Mapped[int | None] = mapped_column(Integer)        # potholes seen outside the GPX time range
    route: Mapped[dict | None] = mapped_column(JSON)
    annotated_video: Mapped[str | None] = mapped_column(String(300))
    is_demo: Mapped[int] = mapped_column(Integer, default=0)
    sightings: Mapped[list["Sighting"]] = relationship(back_populates="trip", cascade="all, delete-orphan")


class Pothole(Base):
    __tablename__ = "potholes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(8))
    width_m: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    sightings_count: Mapped[int] = mapped_column(Integer, default=0)
    trips_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    image: Mapped[str | None] = mapped_column(String(300))
    road: Mapped[str | None] = mapped_column(String(300))
    sightings: Mapped[list["Sighting"]] = relationship(back_populates="pothole")
    __table_args__ = (Index("ix_potholes_latlon", "lat", "lon"),)


class Sighting(Base):
    __tablename__ = "sightings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pothole_id: Mapped[int] = mapped_column(ForeignKey("potholes.id", ondelete="CASCADE"), index=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    video_t: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(8))
    width_m: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    frames: Mapped[int] = mapped_column(Integer)
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    image: Mapped[str] = mapped_column(String(300))
    pothole: Mapped[Pothole] = relationship(back_populates="sightings")
    trip: Mapped[Trip] = relationship(back_populates="sightings")


class GeocodeCache(Base):
    __tablename__ = "geocode_cache"
    key: Mapped[str] = mapped_column(String(32), primary_key=True)   # lat,lon rounded to ~10 m
    road: Mapped[str | None] = mapped_column(String(300))
    area: Mapped[str | None] = mapped_column(String(300))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def make_engine(url: str | None = None):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url or settings.db_url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _):  # noqa: ANN001
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    Base.metadata.create_all(engine)
    return engine


engine = make_engine()
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_session() -> Session:
    return SessionLocal()
