"""
db.py — SQLite detection logging + dashboard analytics for the wasp app.

Every /detect-image and /detect-video call is written to a local SQLite
database (see log_detection). get_dashboard_stats() reads it back with
pandas and returns everything the Ledger tab in the UI needs in one call.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "wasp_detections.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the detections table if it doesn't already exist.
    Call once at app startup."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                source_type     TEXT    NOT NULL,   -- 'image' or 'video'
                filename        TEXT,
                detection_count INTEGER NOT NULL,
                avg_confidence  REAL,
                max_confidence  REAL,
                min_confidence  REAL,
                inference_ms    REAL,
                conf_threshold  REAL,
                iou_threshold   REAL,
                extra_json      TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
            ON detections(timestamp)
        """)


def log_detection(source_type, filename, count, confidences,
                   inference_ms, conf_threshold, iou_threshold, extra=None):
    """Insert one row per detection request. Never raises — a logging
    failure should never take down a real inference response."""
    try:
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        max_conf = max(confidences) if confidences else None
        min_conf = min(confidences) if confidences else None

        with get_conn() as conn:
            conn.execute("""
                INSERT INTO detections
                    (timestamp, source_type, filename, detection_count,
                     avg_confidence, max_confidence, min_confidence,
                     inference_ms, conf_threshold, iou_threshold, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                source_type, filename, count, avg_conf, max_conf, min_conf,
                inference_ms, conf_threshold, iou_threshold,
                json.dumps(extra) if extra else None,
            ))
    except Exception as e:
        print(f"[WARNING] Failed to log detection to DB: {e}")


def get_dashboard_stats(days: int = 30) -> dict:
    """Aggregate everything the Ledger/analytics tab needs into one dict,
    ready to be returned as JSON. Returns zeroed/empty structures (not an
    error) when the table has no rows yet, so the UI can render an empty
    state cleanly."""
    empty = {
        "total_records": 0, "image_count": 0, "video_count": 0,
        "total_wasps_found": 0, "avg_confidence": None,
        "avg_inference_ms": None, "zero_hit_rate": 0,
        "daily_series": [], "confidence_histogram": [],
        "inference_series": [], "source_breakdown": {},
        "recent_records": [],
    }

    if not DB_PATH.exists():
        return empty

    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM detections"
        if days:
            query += f" WHERE timestamp >= datetime('now', '-{int(days)} days')"
        df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    finally:
        conn.close()

    if df.empty:
        return empty

    # ── KPIs ──────────────────────────────────────────────────────────────
    total_records = len(df)
    image_count = int((df["source_type"] == "image").sum())
    video_count = int((df["source_type"] == "video").sum())
    total_wasps = int(df["detection_count"].sum())
    avg_confidence = df["avg_confidence"].mean()
    avg_inference_ms = df["inference_ms"].mean()
    zero_hits = int((df["detection_count"] == 0).sum())
    zero_hit_rate = round(zero_hits / total_records, 4) if total_records else 0

    # ── Daily time series (requests + avg confidence per day) ──────────────
    daily = (
        df.set_index("timestamp")
          .resample("D")
          .agg(requests=("id", "count"),
               wasps_found=("detection_count", "sum"),
               avg_confidence=("avg_confidence", "mean"))
          .reset_index()
    )
    daily_series = [
        {
            "date": row.timestamp.strftime("%Y-%m-%d"),
            "requests": int(row.requests),
            "wasps_found": int(row.wasps_found) if pd.notna(row.wasps_found) else 0,
            "avg_confidence": round(row.avg_confidence, 3) if pd.notna(row.avg_confidence) else None,
        }
        for row in daily.itertuples()
    ]

    # ── Confidence histogram (10 fixed bins from 0 to 1) ────────────────────
    conf_series = df["avg_confidence"].dropna()
    bins = [i / 10 for i in range(11)]
    confidence_histogram = [
        {"bucket": f"{bins[i]:.1f}\u2013{bins[i+1]:.1f}", "count": int(c)}
        for i, c in enumerate(pd.cut(conf_series, bins=bins, include_lowest=True)
                               .value_counts(sort=False).values)
    ] if len(conf_series) else []

    # ── Inference time — most recent 50 requests, chronological ────────────
    recent_for_series = df.sort_values("timestamp").tail(50)
    inference_series = [
        {"timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
         "inference_ms": round(row.inference_ms, 1) if pd.notna(row.inference_ms) else None}
        for row in recent_for_series.itertuples()
    ]

    # ── Source type breakdown (total wasps found, by request type) ─────────
    source_breakdown = (
        df.groupby("source_type")["detection_count"].sum().to_dict()
    )
    source_breakdown = {k: int(v) for k, v in source_breakdown.items()}

    # ── Recent records table (specimen-ledger style, newest first) ─────────
    recent = df.sort_values("timestamp", ascending=False).head(25)
    recent_records = [
        {
            "id": int(row.id),
            "specimen_tag": f"MW\u00b7{row.timestamp.strftime('%Y')}\u00b7{int(row.id):04d}",
            "timestamp": row.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source_type": row.source_type,
            "filename": row.filename or "\u2014",
            "detection_count": int(row.detection_count),
            "avg_confidence": round(row.avg_confidence, 3) if pd.notna(row.avg_confidence) else None,
            "inference_ms": round(row.inference_ms, 1) if pd.notna(row.inference_ms) else None,
        }
        for row in recent.itertuples()
    ]

    return {
        "total_records": total_records,
        "image_count": image_count,
        "video_count": video_count,
        "total_wasps_found": total_wasps,
        "avg_confidence": round(avg_confidence, 3) if pd.notna(avg_confidence) else None,
        "avg_inference_ms": round(avg_inference_ms, 1) if pd.notna(avg_inference_ms) else None,
        "zero_hit_rate": zero_hit_rate,
        "daily_series": daily_series,
        "confidence_histogram": confidence_histogram,
        "inference_series": inference_series,
        "source_breakdown": source_breakdown,
        "recent_records": recent_records,
    }
