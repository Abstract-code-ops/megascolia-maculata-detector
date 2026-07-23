# Wasp Detection App — Field Ledger build

## What's in here
- `app.py` — Flask backend: `/detect-image`, `/detect-video`, `/api/stats`, `/health`
- `db.py` — SQLite logging + pandas-based dashboard analytics (`get_dashboard_stats`)
- `templates/index.html` — Field-Ledger themed UI (Detect tab + Ledger/analytics tab)
- `analytics.py` — standalone script (not a Flask route) for offline pandas/sklearn analysis
- `CPU_SPEED_FIX.py` — reference notes on CPU inference speed, not meant to be run
- `requirements_extras.txt` — pandas / scikit-learn / matplotlib / numpy

## Setup
```bash
pip install -r requirements.txt          # your existing app deps (flask, ultralytics, opencv, torch, etc.)
pip install -r requirements_extras.txt   # pandas + sklearn + matplotlib
python app.py
```
Drop your trained weights in as `best.pt` (or set `MODEL_PATH` env var to point elsewhere,
including an exported OpenVINO/ONNX folder) and visit `http://localhost:5000`.

## Still on you, in priority order
1. **Rotate the two exposed secrets** (Comet ML API key, ngrok auth token) if those notebooks were
   ever pushed anywhere. This wasn't part of this package — it's a standing action item from
   the earlier review.
2. **CPU speed** — confirm `yolo26s`/nano vs `yolo26m` tradeoff and `torch.set_num_threads` are
   set the way you want (see `CPU_SPEED_FIX.py`). If you've since exported to OpenVINO, the app's
   `run_tracked_inference` already has a safe fallback for formats that don't support `.track()`.
3. **Populate the Ledger tab** — it renders an empty state cleanly until you run a few detections;
   nothing to configure, it reads live from `wasp_detections.db`.

## Notes on this rebuild
- Fixed a bug in the histogram computation in `db.py` (a leftover unused intermediate that could've
  produced misleading bucket counts) — it now computes `confidence_histogram` directly from one
  `pd.cut(...).value_counts()` call.
- Changed `if last_computed_result and len(...)` to `is not None` in the video loop in `app.py` —
  truthiness checks on Ultralytics `Results` objects can throw or behave unexpectedly; an explicit
  `is not None` is safer.
- Everything else matches what was scoped: SQLite logging wired into both detect routes, a
  `/api/stats` endpoint backing the Ledger tab, and the teal/forest color system applied throughout.
