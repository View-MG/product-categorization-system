from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from compute_drift import (
        calculate_drift_from_prediction_ids,
        get_conn,
        insert_drift_event,
        load_reference_embeddings,
        load_reference_stats,
    )
except ImportError:
    from scripts.compute_drift import (
        calculate_drift_from_prediction_ids,
        get_conn,
        insert_drift_event,
        load_reference_embeddings,
        load_reference_stats,
    )


DB_PATH = "monitoring.db"
REFERENCE_STATS_PATH = "reference_stats.json"
REFERENCE_EMBEDDINGS_PATH = "reference_embeddings.npz"

WINDOW_SIZE = 50
COOLDOWN_MINUTES = 30

STATE_KEY_LAST_PROCESSED_ID = "last_processed_prediction_id"


def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def parse_dt(value: str) -> datetime:
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass

    return datetime.utcnow()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                predicted_class  TEXT NOT NULL,
                confidence       FLOAT NOT NULL,
                latency_ms       FLOAT,
                brightness       FLOAT,
                blur_var         FLOAT,
                width            INTEGER,
                height           INTEGER,
                quality_warnings TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS human_feedback (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id  INTEGER NOT NULL REFERENCES prediction_events(id),
                true_label     TEXT NOT NULL,
                labeled_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS drift_events (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp          DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_score    FLOAT,
                confidence_score   FLOAT,
                class_score        FLOAT,
                is_drift           BOOLEAN DEFAULT 0,
                embedding_drifted  BOOLEAN DEFAULT 0,
                confidence_drifted BOOLEAN DEFAULT 0,
                class_drifted      BOOLEAN DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type  TEXT NOT NULL,
                message     TEXT,
                resolved    BOOLEAN DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS prediction_embeddings (
                prediction_id INTEGER PRIMARY KEY REFERENCES prediction_events(id) ON DELETE CASCADE,
                embedding_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orchestrator_state (
                state_key   TEXT PRIMARY KEY,
                state_value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_prediction_events_id
            ON prediction_events(id);

            CREATE INDEX IF NOT EXISTS idx_alerts_type_timestamp
            ON alerts(alert_type, timestamp);
            """
        )
        conn.commit()


def get_state_int(conn: sqlite3.Connection, state_key: str, default: int = 0) -> int:
    row = conn.execute(
        """
        SELECT state_value
        FROM orchestrator_state
        WHERE state_key = ?
        """,
        (state_key,),
    ).fetchone()

    if row is None:
        return default

    try:
        return int(row["state_value"])
    except Exception:
        return default


def set_state_int(conn: sqlite3.Connection, state_key: str, value: int) -> None:
    conn.execute(
        """
        INSERT INTO orchestrator_state (state_key, state_value)
        VALUES (?, ?)
        ON CONFLICT(state_key) DO UPDATE SET state_value = excluded.state_value
        """,
        (state_key, str(int(value))),
    )


def insert_prediction_event(conn: sqlite3.Connection, result: Dict[str, Any]) -> int:
    timestamp = result.get("timestamp", now_str())
    predicted_class = str(result["predicted_class"])
    confidence = float(result["confidence"])
    latency_ms = None if result.get("latency_ms") is None else float(result["latency_ms"])
    brightness = None if result.get("brightness") is None else float(result["brightness"])
    blur_var = None if result.get("blur_var") is None else float(result["blur_var"])
    width = None if result.get("width") is None else int(result["width"])
    height = None if result.get("height") is None else int(result["height"])
    quality_warnings = json.dumps(result.get("quality_warnings", []), ensure_ascii=False)

    cur = conn.execute(
        """
        INSERT INTO prediction_events (
            timestamp,
            predicted_class,
            confidence,
            latency_ms,
            brightness,
            blur_var,
            width,
            height,
            quality_warnings
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            predicted_class,
            confidence,
            latency_ms,
            brightness,
            blur_var,
            width,
            height,
            quality_warnings,
        ),
    )
    return int(cur.lastrowid)


def insert_prediction_embedding(conn: sqlite3.Connection, prediction_id: int, embedding: Any) -> None:
    embedding_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    embedding_json = json.dumps(embedding_arr.tolist(), ensure_ascii=False)

    conn.execute(
        """
        INSERT OR REPLACE INTO prediction_embeddings (prediction_id, embedding_json)
        VALUES (?, ?)
        """,
        (prediction_id, embedding_json),
    )


def fetch_next_window_prediction_ids(
    conn: sqlite3.Connection,
    start_after_id: int,
    window_size: int,
) -> List[int]:
    rows = conn.execute(
        """
        SELECT id
        FROM prediction_events
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (start_after_id, window_size),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def get_last_alert_time(conn: sqlite3.Connection, alert_type: str) -> Optional[datetime]:
    row = conn.execute(
        """
        SELECT timestamp
        FROM alerts
        WHERE alert_type = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (alert_type,),
    ).fetchone()

    if row is None or row["timestamp"] is None:
        return None

    return parse_dt(str(row["timestamp"]))


def is_in_cooldown(conn: sqlite3.Connection, alert_type: str, cooldown_minutes: int) -> bool:
    last_alert_time = get_last_alert_time(conn, alert_type)
    if last_alert_time is None:
        return False
    return datetime.utcnow() - last_alert_time < timedelta(minutes=cooldown_minutes)


def insert_alert(conn: sqlite3.Connection, alert_type: str, message: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO alerts (timestamp, alert_type, message, resolved)
        VALUES (?, ?, ?, 0)
        """,
        (now_str(), alert_type, message),
    )
    return int(cur.lastrowid)


def build_alert_message(
    drift_result: Dict[str, Any],
    window_prediction_ids: List[int],
) -> str:
    parts = []

    if drift_result["embedding_drifted"]:
        parts.append(f"embedding={drift_result['embedding_score']:.4f}")
    if drift_result["confidence_drifted"]:
        parts.append(f"confidence={drift_result['confidence_score']:.4f}")
    if drift_result["class_drifted"]:
        parts.append(f"class={drift_result['class_score']:.4f}")

    detail = ", ".join(parts) if parts else "unknown"
    return (
        f"Drift detected for prediction window "
        f"{window_prediction_ids[0]}-{window_prediction_ids[-1]}: {detail}"
    )


class DriftOrchestrator:
    def __init__(
        self,
        db_path: str = DB_PATH,
        reference_stats_path: str = REFERENCE_STATS_PATH,
        reference_embeddings_path: str = REFERENCE_EMBEDDINGS_PATH,
        window_size: int = WINDOW_SIZE,
        cooldown_minutes: int = COOLDOWN_MINUTES,
    ) -> None:
        init_db(db_path)
        self.db_path = db_path
        self.reference_stats = load_reference_stats(reference_stats_path)
        self.reference_embeddings = load_reference_embeddings(reference_embeddings_path)
        self.window_size = int(window_size)
        self.cooldown_minutes = int(cooldown_minutes)
        self.alert_type = "drift"

    def process_pending_windows(self, conn: sqlite3.Connection) -> Dict[str, List[int]]:
        drift_event_ids: List[int] = []
        alert_ids: List[int] = []

        while True:
            last_processed_id = get_state_int(conn, STATE_KEY_LAST_PROCESSED_ID, 0)
            prediction_ids = fetch_next_window_prediction_ids(conn, last_processed_id, self.window_size)

            if len(prediction_ids) < self.window_size:
                break

            drift_result = calculate_drift_from_prediction_ids(
                conn=conn,
                prediction_ids=prediction_ids,
                ref_stats=self.reference_stats,
                ref_embeddings=self.reference_embeddings,
            )

            drift_event_id = insert_drift_event(conn, drift_result)
            drift_event_ids.append(drift_event_id)

            set_state_int(conn, STATE_KEY_LAST_PROCESSED_ID, prediction_ids[-1])

            if drift_result["is_drift"] and not is_in_cooldown(conn, self.alert_type, self.cooldown_minutes):
                message = build_alert_message(drift_result, prediction_ids)
                alert_id = insert_alert(conn, self.alert_type, message)
                alert_ids.append(alert_id)

        return {
            "drift_event_ids": drift_event_ids,
            "alert_ids": alert_ids,
        }

    def handle_inference(self, result: Dict[str, Any]) -> Dict[str, Any]:
        with get_conn(self.db_path) as conn:
            prediction_id = insert_prediction_event(conn, result)

            if result.get("embedding") is not None:
                insert_prediction_embedding(conn, prediction_id, result["embedding"])

            processed = self.process_pending_windows(conn)
            conn.commit()

        return {
            "prediction_id": prediction_id,
            "drift_event_ids": processed["drift_event_ids"],
            "alert_ids": processed["alert_ids"],
        }


if __name__ == "__main__":
    orchestrator = DriftOrchestrator(
        db_path=DB_PATH,
        reference_stats_path=REFERENCE_STATS_PATH,
        reference_embeddings_path=REFERENCE_EMBEDDINGS_PATH,
        window_size=50,
        cooldown_minutes=30,
    )

    embedding_dim = int(orchestrator.reference_embeddings.shape[1])

    sample_result = {
        "predicted_class": "beverages",
        "confidence": 0.91,
        "latency_ms": 38.4,
        "brightness": 126.7,
        "blur_var": 215.2,
        "width": 224,
        "height": 224,
        "quality_warnings": [],
        "embedding": np.random.rand(embedding_dim).astype(np.float32).tolist(),
    }

    out = orchestrator.handle_inference(sample_result)
    print(json.dumps(out, ensure_ascii=False, indent=2))