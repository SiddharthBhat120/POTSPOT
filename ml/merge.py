"""Phase 5b: one pothole, one record, across trips.

Each trip reports its own potholes. If two trips see the same pothole, their GPS
positions differ by a few metres (GPS error plus the detection timing). Sightings
closer than `radius_m` to an existing pothole's average position are merged into it.

Greedy clustering against the running centre (not chaining): A near B and B near C
does not merge A and C if they are far apart. This stops a long row of potholes on a
bad road from collapsing into one record.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geo import haversine_m
from .severity import SEVERITIES

DEFAULT_RADIUS_M = 8.0


@dataclass
class Sighting:
    lat: float
    lon: float
    severity: str
    confidence: float
    trip_id: str = ""
    key: str = ""  # caller's id for this sighting
    width_m: float | None = None
    time: str = ""


@dataclass
class Cluster:
    lat: float
    lon: float
    sightings: list[Sighting] = field(default_factory=list)
    key: int | None = None  # database id when the cluster is an existing pothole

    @property
    def count(self) -> int:
        return len(self.sightings)

    @property
    def trips(self) -> int:
        return len({s.trip_id for s in self.sightings})

    @property
    def severity(self) -> str:
        """Worst severity seen. A pothole only grows; a smaller later reading is usually a worse angle."""
        return max((s.severity for s in self.sightings), key=SEVERITIES.index)

    @property
    def confidence(self) -> float:
        return max(s.confidence for s in self.sightings)

    def add(self, s: Sighting) -> None:
        self.sightings.append(s)
        n = len(self.sightings)
        self.lat += (s.lat - self.lat) / n
        self.lon += (s.lon - self.lon) / n


def merge(sightings: list[Sighting], radius_m: float = DEFAULT_RADIUS_M,
          existing: list[Cluster] | None = None) -> list[Cluster]:
    clusters = list(existing or [])
    for s in sightings:
        best, best_d = None, radius_m
        for c in clusters:
            # Two sightings from the same trip are different potholes (tracking already merged repeats).
            if s.trip_id and any(x.trip_id == s.trip_id for x in c.sightings):
                continue
            d = haversine_m(s.lat, s.lon, c.lat, c.lon)
            if d <= best_d:
                best, best_d = c, d
        if best is None:
            best = Cluster(s.lat, s.lon)
            clusters.append(best)
        best.add(s)
    return clusters
