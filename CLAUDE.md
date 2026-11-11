# PotSpot (spec name: RoadScan)

Detects potholes in phone dashcam video with a fine-tuned YOLO model, tracks each one across frames
(ByteTrack + stitching) so it is counted once, geotags it from a GPX track, merges repeat sightings across
rides, and shows everything on a Leaflet map with a PDF report for the city.

## Stack
Python, Ultralytics YOLO11, OpenCV, FastAPI, SQLite (SQLAlchemy 2), ReportLab; React + Vite + Leaflet
(markercluster, heat). Training notebook for Colab in notebooks/.

## Rules
- Type hints; pytest (`python -m pytest backend/tests -q`); config in .env (see .env.example).
- Tests never need the model: the pipeline takes any `Detector` (see backend/tests/synthetic.py).
- Times: the video started `offset_s` after the first GPX point; all stored times are UTC.
- Severity is an estimate from a flat-road camera model. Never present it as a measurement.
- Saved photos are crops with people and vehicles blurred (ml/privacy.py). Never serve raw uploads.
- Nominatim: 1 request/second, User-Agent with contact, cached in SQLite.
- Dataset images are downloaded by script, not committed. Keep test images out of training.

## Layout
ml/ (data/, train, evaluate, detect, pipeline, geo, severity, merge, privacy, motion, track_eval),
backend/app (config, db, jobs, geocode, staticmap, report, main), backend/tests, frontend/ (dist committed),
demo/ (demo ride generator + files), notebooks/, scripts/, docs/.
