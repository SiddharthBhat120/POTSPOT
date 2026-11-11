"""Phase 2b: measure the model on the held-out test split, honestly.

    python -m ml.evaluate                       # models/pothole.pt on data/yolo test split
    python -m ml.evaluate --weights runs/pothole/weights/best.pt

What the numbers mean (simple words):
  precision  Of the boxes the model drew, how many were real potholes.
             Low precision = false alarms (shadows, patches, manholes marked as potholes).
  recall     Of the real potholes, how many the model found.
             Low recall = missed potholes.
  mAP50      Average precision when a box counts as correct if it overlaps the true box
             by at least 50%. It sums up precision across all confidence levels. 1.0 is perfect.
  mAP50-95   The same, averaged over stricter overlaps (50% to 95%). It rewards tight
             boxes. For mapping potholes, mAP50 matters more: we need to find them,
             not outline them to the pixel.

It also saves (to docs/eval/):
  metrics.json and metrics.md   the table for the README
  failures_missed.jpg            real potholes the model did not find
  failures_false.jpg             boxes the model drew where nothing was labelled
  examples.jpg                   a mix of predictions next to the labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
OUT = ROOT / "docs" / "eval"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_gt(img_path: Path, split: str) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    rows = []
    lbl = DATA / "labels" / split / f"{img_path.stem}.txt"
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            cx, cy, bw, bh = (float(v) for v in p[1:])
            rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, np.float32).reshape(-1, 4)


def match(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> tuple[list[int], list[int], list[int]]:
    """Greedy matching by confidence. Returns indices of TP preds, FP preds, missed GT."""
    if len(pred) == 0:
        return [], [], list(range(len(gt)))
    order = np.argsort(-pred[:, 4])
    m = iou(pred[order, :4], gt)
    used, tp, fp = set(), [], []
    for row, pi in enumerate(order):
        j = int(np.argmax(m[row])) if m.shape[1] else -1
        if j >= 0 and m[row, j] >= thr and j not in used:
            used.add(j)
            tp.append(int(pi))
        else:
            fp.append(int(pi))
    return tp, fp, [j for j in range(len(gt)) if j not in used]


def tile(crops: list[np.ndarray], cols: int = 4, cell: int = 300) -> np.ndarray:
    if not crops:
        return np.full((cell, cell, 3), 245, np.uint8)
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 245, np.uint8)
    for k, c in enumerate(crops):
        s = cell / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
        r, q = divmod(k, cols)
        oy, ox = r * cell + (cell - c.shape[0]) // 2, q * cell + (cell - c.shape[1]) // 2
        canvas[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return canvas


def draw(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = img.copy()
    t = max(2, img.shape[1] // 300)
    for b in gt:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 200, 60), t)
    for b in pred:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (176, 72, 255), t)
        cv2.putText(out, f"{b[4]:.2f}", (int(b[0]), max(12, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * t / 2, (176, 72, 255), max(1, t // 2), cv2.LINE_AA)
    return out


def evaluate(weights: Path, split: str = "test", conf: float = 0.25, imgsz: int = 512) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    v = model.val(data=str(DATA / "dataset.yaml"), split=split, imgsz=imgsz, batch=8, device="cpu",
                  plots=False, verbose=False, project=str(ROOT / "runs"), name=f"eval_{split}", exist_ok=True)
    metrics = {
        "split": split, "images": 0, "labelled_potholes": 0,
        "precision": round(float(v.box.mp), 3), "recall": round(float(v.box.mr), 3),
        "mAP50": round(float(v.box.map50), 3), "mAP50_95": round(float(v.box.map), 3),
    }
    # Our own count at one working threshold (the one the app uses), for plain-language numbers.
    tp_n = fp_n = fn_n = 0
    missed, false, examples = [], [], []
    by_size = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [found, total]
    imgs = sorted((DATA / "images" / split).glob("*.jpg"))
    for k, p in enumerate(imgs):
        img = cv2.imread(str(p))
        gt = load_gt(p, split)
        r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        pred = r.boxes.data.cpu().numpy()[:, :5] if r.boxes is not None and len(r.boxes) else np.zeros((0, 5))
        tp, fp, fn = match(pred, gt)
        tp_n, fp_n, fn_n = tp_n + len(tp), fp_n + len(fp), fn_n + len(fn)
        area = img.shape[0] * img.shape[1]
        matched_gt = set(range(len(gt))) - set(fn)
        for j, b in enumerate(gt):
            a = (b[2] - b[0]) * (b[3] - b[1]) / area
            size = "small" if a < 0.01 else "medium" if a < 0.10 else "large"
            by_size[size][1] += 1
            by_size[size][0] += j in matched_gt
        for j in fn[:1]:
            b = gt[j].astype(int)
            pad = int(max(b[2] - b[0], b[3] - b[1]) * 0.6)
            crop = draw(img, gt[j:j + 1], pred)[max(0, b[1] - pad):b[3] + pad, max(0, b[0] - pad):b[2] + pad]
            missed.append(crop)
        for i in sorted(fp, key=lambda i: -pred[i, 4])[:1]:
            if pred[i, 4] < 0.4:
                continue
            b = pred[i, :4].astype(int)
            pad = int(max(b[2] - b[0], b[3] - b[1]) * 0.6)
            crop = draw(img, gt, pred[i:i + 1])[max(0, b[1] - pad):b[3] + pad, max(0, b[0] - pad):b[2] + pad]
            false.append((float(pred[i, 4]), crop))
        if k % max(1, len(imgs) // 12) == 0 and len(examples) < 12:
            examples.append(draw(img, gt, pred))
    metrics.update({
        "images": len(imgs), "labelled_potholes": tp_n + fn_n, "working_threshold": conf,
        "found": tp_n, "missed": fn_n, "false_alarms": fp_n,
        "precision_at_threshold": round(tp_n / max(1, tp_n + fp_n), 3),
        "recall_at_threshold": round(tp_n / max(1, tp_n + fn_n), 3),
        "recall_by_size": {k: {"found": f, "total": t, "recall": round(f / t, 3) if t else None}
                           for k, (f, t) in by_size.items()},
    })
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "failures_missed.jpg"), tile(missed[:16]))
    cv2.imwrite(str(OUT / "failures_false.jpg"), tile([c for _, c in sorted(false, key=lambda x: -x[0])][:16]))
    cv2.imwrite(str(OUT / "examples.jpg"), tile(examples, cols=4, cell=360))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    md = ["| Metric | Test split |", "|---|---|",
          f"| Images / labelled potholes | {metrics['images']} / {metrics['labelled_potholes']} |",
          f"| Precision | {metrics['precision']} |", f"| Recall | {metrics['recall']} |",
          f"| mAP50 | {metrics['mAP50']} |", f"| mAP50-95 | {metrics['mAP50_95']} |",
          f"| At confidence {conf}: found / missed / false alarms | {tp_n} / {fn_n} / {fp_n} |"]
    md += [f"| Recall on {k} potholes ({v['total']}) | {v['recall']} |" for k, v in metrics["recall_by_size"].items()]
    (OUT / "metrics.md").write_text("\n".join(md) + "\n")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "pothole.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    a = ap.parse_args()
    print(json.dumps(evaluate(Path(a.weights), a.split, a.conf, a.imgsz), indent=2))
