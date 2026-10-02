import sqlite3
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            top3 TEXT,
            inference_time_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(
    mode,
    predicted_class,
    confidence,
    top3,
    inference_time_ms
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO predictions (
            mode,
            predicted_class,
            confidence,
            top3,
            inference_time_ms
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            mode,
            predicted_class,
            confidence,
            json.dumps(top3),
            inference_time_ms
        )
    )

    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            id,
            mode,
            predicted_class,
            confidence,
            top3,
            inference_time_ms,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    conn.close()

    history = []

    for row in rows:
        item = dict(row)

        try:
            item["top3"] = json.loads(item["top3"])
        except (TypeError, json.JSONDecodeError):
            item["top3"] = []

        history.append(item)

    return history