"""Phase 8: the PDF you can send to the city corporation."""
from __future__ import annotations

import io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from . import geocode, staticmap
from .config import settings
from .db import Pothole, Trip

INK = colors.HexColor("#1B1D24")
MUTED = colors.HexColor("#5C6070")
SEV_COL = {"large": colors.HexColor("#CD2630"), "medium": colors.HexColor("#F47020"), "small": colors.HexColor("#C99700")}
ORDER = {"large": 0, "medium": 1, "small": 2}


def photo(path: Path | None, style: ParagraphStyle):
    """Thumbnail for the table, or a note if the file is missing or not a readable image."""
    if path and path.exists():
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                im.verify()
            return RLImage(str(path), width=34 * mm, height=24 * mm, kind="proportional")
        except Exception:
            pass
    return Paragraph("no photo", style)


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def build(potholes: list[Pothole], trips: list[Trip], filters: str, top_n: int = 60) -> bytes:
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=20, textColor=INK, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=INK, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13, textColor=INK)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=MUTED)
    cell = ParagraphStyle("c", parent=body, fontSize=8.5, leading=11)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="PotSpot pothole report", author="PotSpot")
    km = sum(t.distance_km or 0 for t in trips)
    counts = {k: sum(1 for p in potholes if p.severity == k) for k in ORDER}
    story = [
        Paragraph("Pothole report", h1),
        Paragraph(escape(f"Generated {datetime.now(timezone.utc):%d %b %Y, %H:%M} UTC by PotSpot. {filters}"), small),
        Spacer(1, 6),
    ]
    summary = [["Potholes mapped", "Large", "Medium", "Small", "Road covered", "Per km", "Trips"],
               [str(len(potholes)), str(counts["large"]), str(counts["medium"]), str(counts["small"]),
                f"{km:.1f} km", f"{len(potholes) / km:.1f}" if km else "-", str(len(trips))]]
    t = Table(summary, colWidths=[30 * mm, 18 * mm, 20 * mm, 18 * mm, 28 * mm, 18 * mm, 18 * mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica", 7.5), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 15), ("TEXTCOLOR", (1, 1), (1, 1), SEV_COL["large"]),
        ("TEXTCOLOR", (2, 1), (2, 1), SEV_COL["medium"]), ("TEXTCOLOR", (3, 1), (3, 1), SEV_COL["small"]),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, INK), ("BOTTOMPADDING", (0, 1), (-1, 1), 6)]))
    story += [t, Spacer(1, 8)]

    png, tiles = staticmap.render([(p.lat, p.lon, p.severity) for p in potholes],
                                  [[(c[1], c[0]) for c in (tr.route or {}).get("coordinates", [])] for tr in trips])
    story.append(RLImage(io.BytesIO(png), width=180 * mm, height=180 * mm * 620 / 1000))
    story.append(Paragraph("Blue: routes driven. Dots: potholes, coloured by severity (red large, orange medium, "
                           "yellow small).", small))

    # group by road (cached reverse geocoding; fills in missing names politely)
    for p in potholes:
        if not p.road and not settings.offline:
            road, area = geocode.reverse(p.lat, p.lon)
            p.road = road or (f"Near {area}" if area else None)
    worst = sorted(potholes, key=lambda p: (ORDER[p.severity], -p.sightings_count, -p.confidence))[:top_n]
    groups: dict[str, list[Pothole]] = defaultdict(list)
    for p in worst:
        groups[p.road or "Unnamed road"].append(p)
    story += [PageBreak(), Paragraph(f"Worst potholes by road (top {len(worst)})", h2),
              Paragraph("Each row links to the exact spot on Google Maps. Photos have people and vehicles blurred.", small)]
    rank = 0
    for road, items in sorted(groups.items(), key=lambda kv: (min(ORDER[p.severity] for p in kv[1]), -len(kv[1]))):
        rows = [[Paragraph(f"<b>{escape(road)}</b> ({len(items)})", cell), "", "", ""]]
        for p in items:
            rank += 1
            img = photo(settings.trips_dir / p.image if p.image else None, small)
            sev = f'<font color="{SEV_COL[p.severity].hexval()}"><b>{p.severity.upper()}</b></font>'
            width = f"about {p.width_m:.1f} m wide" if p.width_m else "size unknown"
            seen = f"seen {p.sightings_count}x on {p.trips_count} trip{'s' if p.trips_count != 1 else ''}"
            when = p.last_seen.strftime("%d %b %Y") if p.last_seen else ""
            rows.append([Paragraph(f"#{rank}", cell), img,
                         Paragraph(f"{sev}<br/>{width}<br/>{seen}<br/>last {when}", cell),
                         Paragraph(f"{p.lat:.6f}, {p.lon:.6f}<br/>"
                                   f'<link href="{maps_link(p.lat, p.lon)}" color="#2B3BD6">Open in Google Maps</link><br/>'
                                   f"confidence {p.confidence:.0%}", cell)])
        tbl = Table(rows, colWidths=[12 * mm, 38 * mm, 52 * mm, 78 * mm])
        tbl.setStyle(TableStyle([
            ("SPAN", (0, 0), (-1, 0)), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF0F6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#D5D8E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story += [KeepTogether([tbl]), Spacer(1, 6)]

    story += [Paragraph("How to read this report", h2), Paragraph(
        "Potholes were found automatically in phone video by an AI model and placed using the phone's GPS. "
        "Positions can be off by 5 to 15 metres. Severity comes from the pothole's apparent width on a flat road; "
        "the camera cannot see depth, so a 'small' pothole can still be deep. Please verify on site before work is "
        "ordered. A pothole seen on several trips is more certain than one seen once.", body)]
    if not tiles:
        story.append(Paragraph("Map tiles could not be loaded when this report was made, so the map has no streets.", small))
    doc.build(story)
    return buf.getvalue()
