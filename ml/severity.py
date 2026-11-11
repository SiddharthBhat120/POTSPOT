"""Phase 5a: estimate how big a pothole is from one camera, and turn that into small / medium / large.

The idea (flat-road pinhole camera):
  A pothole lying on the road looks smaller the further away it is. On a flat road, how
  far away it is shows up as how far below the horizon its bottom edge sits. So:

      estimated width (m)  ~=  camera_height_m  x  box_width_px / (box_bottom_px - horizon_px)

  This cancels the camera's focal length, so it works for any phone, if the road is flat
  and the horizon line is roughly right.

Honest limits:
  - It measures width only. Depth (how deep it is), which matters most for damage, is
    invisible to a single camera.
  - Camera height and horizon are assumptions (defaults: 1.2 m, 45% down the frame).
    A phone tilted down or mounted high shifts every estimate.
  - Slopes, speed bumps and a bouncing bike break the flat-road assumption for a moment.
  - Boxes touching the bottom or side of the frame are cut off, so they are skipped.
  - Water-filled potholes and patches of shadow can look bigger or smaller than they are.
  So severity is a rough triage label, not a measurement. The report says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

SEVERITIES = ("small", "medium", "large")


@dataclass(frozen=True)
class CameraModel:
    height_m: float = 1.2       # camera above the road (bike handlebar ~1.1, car dashboard ~1.3)
    horizon_frac: float = 0.45  # horizon line, as a fraction of frame height from the top
    small_below_m: float = 0.30
    large_above_m: float = 0.70


def width_m(box: tuple[float, float, float, float], frame_w: int, frame_h: int, cam: CameraModel) -> float | None:
    """Estimated real width of one detection, or None if the box cannot be trusted."""
    x1, y1, x2, y2 = box
    horizon = cam.horizon_frac * frame_h
    below = y2 - horizon
    edge = 2
    if below < 0.08 * frame_h:        # too close to the horizon: tiny and unreliable
        return None
    if y2 >= frame_h - edge or x1 <= edge or x2 >= frame_w - edge:  # cut off by the frame
        return None
    return cam.height_m * (x2 - x1) / below


def estimate(observations: list[tuple[float, float, float, float]], frame_w: int, frame_h: int,
             cam: CameraModel = CameraModel()) -> tuple[str, float | None]:
    """Severity and width from all sightings of one pothole (median of the usable ones)."""
    widths = [w for b in observations if (w := width_m(b, frame_w, frame_h, cam)) is not None]
    if not widths:
        # Fall back on the largest box: better than nothing, but flagged by width=None
        x1, y1, x2, y2 = max(observations, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
        return ("large" if frac > 0.06 else "medium" if frac > 0.015 else "small"), None
    w = median(widths)
    return label(w, cam), round(w, 2)


def label(width: float, cam: CameraModel = CameraModel()) -> str:
    if width < cam.small_below_m:
        return "small"
    if width > cam.large_above_m:
        return "large"
    return "medium"
