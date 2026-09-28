"""
SQLite persistence for prediction history (class, confidence, Top-3, inference time),
per the project's "rich output and storage" objective.
"""
import json
import sqlite3
from datetime import datetime

from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            top3 TEXT NOT NULL,
            inference_time_ms REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(mode, predicted_class, confidence, top3, inference_time_ms):
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions
           (timestamp, mode, predicted_class, confidence, top3, inference_time_ms)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            mode,
            predicted_class,
            confidence,
            json.dumps(top3),
            inference_time_ms,
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "mode": r["mode"],
            "predicted_class": r["predicted_class"],
            "confidence": r["confidence"],
            "top3": json.loads(r["top3"]),
            "inference_time_ms": r["inference_time_ms"],
        }
        for r in rows
    ]
