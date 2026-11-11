"""Score the whole video pipeline on the demo trip, where the true pothole count is known.

    python -m ml.track_eval

Each pothole the pipeline reports is mapped back onto the photo it came from (we know the
virtual camera's zoom at every frame) and matched to the hand-drawn boxes:
  found          labelled potholes the pipeline reported
  double counted extra records for a pothole already reported (tracking broke)
  false records  records that match no labelled pothole
It also reports raw detections vs final records, which shows what tracking saves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from .evaluate import iou  # noqa: E402
from .pipeline import PipelineConfig, run  # noqa: E402


def main(out_dir: Path = ROOT / "data" / "trips" / "_track_eval") -> dict:
    from make_demo_trip import to_photo  # noqa: E402

    gt = json.loads((ROOT / "demo" / "demo_ground_truth.json").read_text())
    res = run(ROOT / "demo" / "demo_trip.mp4", out_dir, ROOT / "demo" / "demo_trip.gpx", gt["offset_s"],
              PipelineConfig(), progress=lambda p, m: print(f"\r{p:4.0%} {m}", end="", flush=True))
    print()
    fpp = gt["frames_per_photo"]
    fps = gt["fps"]
    # all labelled boxes that were visible long enough, as (photo index, box xyxy in photo pixels)
    wanted: dict[tuple[int, int], bool] = {}
    for item in gt["potholes"]:
        pi = int(round(item["photo_start_s"] * fps)) // fpp
        ph = gt["photos"][pi]
        j = ph["boxes"].index(item["box"])
        wanted[(pi, j)] = False
    found = double = false = 0
    details = []
    for p in res.potholes:
        frame = int(round(p.best_t * fps))
        pi = min(len(gt["photos"]) - 1, frame // fpp)
        ph = gt["photos"][pi]
        k = (frame - ph["start_frame"]) / max(1, fpp - 1)
        box = np.array([to_photo(p.best_box, ph["w"], ph["h"], k)], np.float32)
        gtb = np.array([[(cx - bw / 2) * ph["w"], (cy - bh / 2) * ph["h"], (cx + bw / 2) * ph["w"], (cy + bh / 2) * ph["h"]]
                        for cx, cy, bw, bh in ph["boxes"]], np.float32)
        m = iou(box, gtb)[0] if len(gtb) else np.array([])
        j = int(np.argmax(m)) if len(m) else -1
        # a looser overlap than detection scoring: the best frame is often a close-up of one part
        if j >= 0 and m[j] >= 0.2:
            key = (pi, j)
            if key in wanted and not wanted[key]:
                wanted[key] = True
                found += 1
                verdict = "found"
            elif key in wanted:
                double += 1
                verdict = "double"
            else:
                found += 0
                verdict = "found (short-visible label)"
        else:
            false += 1
            verdict = "false"
        details.append({"track": p.track_id, "t": p.best_t, "photo": ph["photo"], "iou": round(float(m[j]) if j >= 0 else 0, 2),
                        "verdict": verdict, "severity": p.severity})
    total = len(wanted)
    report = {
        "video_s": res.duration_s, "frames_processed": res.frames_processed,
        "raw_detections": res.raw_detections, "tracks": res.tracks_total, "tracks_joined": res.tracks_joined,
        "records": len(res.potholes), "labelled_potholes": total, "found": found,
        "recall": round(found / max(1, total), 3), "double_counted": double, "false_records": false,
        "processing_s": res.seconds, "privacy_blurred": res.privacy_blurred, "details": details,
    }
    (ROOT / "docs" / "eval").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "eval" / "tracking.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=1))
    return report


if __name__ == "__main__":
    main()
