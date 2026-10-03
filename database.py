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

"""
SQLite persistence.

Three tables:
  users              - one row per signed-in Google account
  predictions        - single digit/character predictions (existing feature),
                        now scoped to the user who made them
  documents           - whole PDF/photo predictions (new feature): the
                        reconstructed text plus a JSON breakdown of every
                        detected character, also scoped per-user

Every read function requires a user_id and only ever returns that user's own
rows — this is what keeps each person's prediction history private.
"""
import json
import sqlite3
from datetime import datetime

from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT NOT NULL,
            picture TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            top3 TEXT NOT NULL,
            inference_time_ms REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            mode TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            recognized_text TEXT NOT NULL,
            avg_confidence REAL NOT NULL,
            processing_time_ms REAL NOT NULL,
            details TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)")
    conn.commit()
    conn.close()


def upsert_user(user_id, email, name, picture):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO users (id, email, name, picture, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET email=excluded.email, name=excluded.name, picture=excluded.picture
        """,
        (user_id, email, name, picture, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


def log_prediction(user_id, mode, predicted_class, confidence, top3, inference_time_ms):
    conn = get_connection()
    conn.execute(
        """INSERT INTO predictions
           (user_id, timestamp, mode, predicted_class, confidence, top3, inference_time_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
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


def get_history(user_id, limit=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
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


def log_document(user_id, source_type, filename, mode, page_count,
                  recognized_text, avg_confidence, processing_time_ms, details):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO documents
           (user_id, timestamp, source_type, filename, mode, page_count,
            recognized_text, avg_confidence, processing_time_ms, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            datetime.utcnow().isoformat(),
            source_type,
            filename,
            mode,
            page_count,
            recognized_text,
            avg_confidence,
            processing_time_ms,
            json.dumps(details),
        ),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return doc_id


def get_document_history(user_id, limit=10):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "source_type": r["source_type"],
            "filename": r["filename"],
            "mode": r["mode"],
            "page_count": r["page_count"],
            "recognized_text": r["recognized_text"],
            "avg_confidence": r["avg_confidence"],
            "processing_time_ms": r["processing_time_ms"],
            "details": json.loads(r["details"]),
        }
        for r in rows
    ]


def get_document_by_id(user_id, document_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "source_type": row["source_type"],
        "filename": row["filename"],
        "mode": row["mode"],
        "page_count": row["page_count"],
        "recognized_text": row["recognized_text"],
        "avg_confidence": row["avg_confidence"],
        "processing_time_ms": row["processing_time_ms"],
        "details": json.loads(row["details"]),
    }