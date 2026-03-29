import sys
import sqlite3
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitoring.store import (
    DB_PATH,
    init_db,
    insert_alert,
    insert_drift_event,
)
from scripts.compute_drift import (
    WINDOW_SIZE,
    BATCH_SIZE,
    EMBEDDING_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    CLASS_THRESHOLD,
    build_model,
    build_transform,
    compute_class_ratio_drift,
    compute_confidence_drift,
    compute_embedding_drift,
    infer_recent_embeddings,
    load_reference_embedding_mean,
    load_reference_stats,
)

COOLDOWN_MINUTES = 60

_RUNTIME_CACHE = {
    "loaded": False,
    "ref_stats": None,
    "ref_mean": None,
    "class_names": None,
    "input_size": None,
    "mean": None,
    "std": None,
    "num_classes": None,
    "device": None,
    "model": None,
    "transform": None,
}


def ensure_system_state_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS system_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_last_drift_prediction_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM system_state WHERE key = 'last_drift_prediction_id'"
    ).fetchone()
    return int(row[0]) if row else 0


def set_last_drift_prediction_id(conn: sqlite3.Connection, prediction_id: int) -> None:
    conn.execute(
        """
        INSERT INTO system_state (key, value)
        VALUES ('last_drift_prediction_id', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(prediction_id),),
    )


def get_latest_ready_prediction_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT MAX(id)
        FROM prediction_events
        WHERE image_data_url IS NOT NULL
          AND TRIM(image_data_url) != ''
        """
    ).fetchone()

    if row is None or row[0] is None:
        return 0

    return int(row[0])


def get_runtime_components():
    if _RUNTIME_CACHE["loaded"]:
        return _RUNTIME_CACHE

    ref_stats = load_reference_stats()
    ref_mean = load_reference_embedding_mean(ref_stats)

    class_names = ref_stats["model_info"]["classes"]
    input_size = int(ref_stats["model_info"]["input_size"])
    mean = ref_stats["model_info"]["norm_mean"]
    std = ref_stats["model_info"]["norm_std"]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes).to(device)
    transform = build_transform(input_size=input_size, mean=mean, std=std)

    _RUNTIME_CACHE.update({
        "loaded": True,
        "ref_stats": ref_stats,
        "ref_mean": ref_mean,
        "class_names": class_names,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
        "device": device,
        "model": model,
        "transform": transform,
    })
    return _RUNTIME_CACHE


def normalize_class_name(name: str, allowed_classes: list[str]) -> str:
    raw = str(name).strip()
    raw_lower = raw.lower()

    alias_map = {
        "beverage": "beverage",
        "beverages": "beverage",
        "snack": "snack",
        "snacks": "snack",
    }

    if raw in allowed_classes:
        return raw

    if raw_lower in alias_map:
        normalized = alias_map[raw_lower]
        for allowed in allowed_classes:
            if allowed.lower() == normalized:
                return allowed

    for allowed in allowed_classes:
        if raw_lower == allowed.lower():
            return allowed

    raise ValueError(f"invalid predicted_class={name!r}, allowed={allowed_classes}")


def count_new_ready_predictions(conn: sqlite3.Connection, last_drift_prediction_id: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM prediction_events
        WHERE id > ?
          AND image_data_url IS NOT NULL
          AND TRIM(image_data_url) != ''
        """,
        (last_drift_prediction_id,),
    ).fetchone()
    return int(row[0])


def load_latest_window_from_db(
    conn: sqlite3.Connection,
    window_size: int,
    allowed_classes: list[str],
):
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT id, predicted_class, confidence, image_data_url
        FROM prediction_events
        WHERE image_data_url IS NOT NULL
          AND TRIM(image_data_url) != ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (window_size,),
    ).fetchall()

    if len(rows) < window_size:
        raise ValueError(
            f"not enough ready prediction rows: need {window_size}, got {len(rows)}"
        )

    rows = rows[::-1]

    ids = [int(r["id"]) for r in rows]
    classes = [normalize_class_name(r["predicted_class"], allowed_classes) for r in rows]
    confidences = np.array([float(r["confidence"]) for r in rows], dtype=np.float32)
    image_data_urls = [str(r["image_data_url"]) for r in rows]

    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidence must be between 0 and 1")

    return ids, classes, confidences, image_data_urls


def compute_drift_for_latest_window() -> dict:
    runtime = get_runtime_components()

    conn = sqlite3.connect(DB_PATH)
    ids, recent_classes, recent_confidences, image_data_urls = load_latest_window_from_db(
        conn=conn,
        window_size=WINDOW_SIZE,
        allowed_classes=runtime["class_names"],
    )
    conn.close()

    recent_embeddings = infer_recent_embeddings(
        model=runtime["model"],
        image_data_urls=image_data_urls,
        transform=runtime["transform"],
        device=runtime["device"],
        batch_size=BATCH_SIZE,
    )

    embedding_score = compute_embedding_drift(runtime["ref_mean"], recent_embeddings)
    confidence_score = compute_confidence_drift(runtime["ref_stats"], recent_confidences)
    class_score = compute_class_ratio_drift(runtime["ref_stats"], recent_classes)

    embedding_drifted = embedding_score > EMBEDDING_THRESHOLD
    confidence_drifted = confidence_score > CONFIDENCE_THRESHOLD
    class_drifted = class_score > CLASS_THRESHOLD
    is_drift = embedding_drifted or confidence_drifted or class_drifted

    return {
        "prediction_ids": ids,
        "window_size": WINDOW_SIZE,
        "embedding_score": float(embedding_score),
        "confidence_score": float(confidence_score),
        "class_score": float(class_score),
        "embedding_drifted": bool(embedding_drifted),
        "confidence_drifted": bool(confidence_drifted),
        "class_drifted": bool(class_drifted),
        "is_drift": bool(is_drift),
    }


def in_alert_cooldown(conn: sqlite3.Connection, cooldown_minutes: int) -> bool:
    row = conn.execute(
        """
        SELECT id
        FROM alerts
        WHERE timestamp >= datetime('now', ?)
        ORDER BY id DESC
        LIMIT 1
        """,
        (f"-{cooldown_minutes} minutes",),
    ).fetchone()
    return row is not None


def build_alert_message(result: dict) -> str:
    parts = []

    if result["embedding_drifted"]:
        parts.append(f"embedding={result['embedding_score']:.4f}")
    if result["confidence_drifted"]:
        parts.append(f"confidence={result['confidence_score']:.4f}")
    if result["class_drifted"]:
        parts.append(f"class={result['class_score']:.4f}")

    if not parts:
        parts.append("threshold exceeded")

    return "Drift detected: " + ", ".join(parts)


def run_orchestrator_from_db(cooldown_minutes: int = COOLDOWN_MINUTES) -> dict:
    init_db()
    ensure_system_state_table()
    get_runtime_components()

    conn = sqlite3.connect(DB_PATH)
    last_drift_prediction_id = get_last_drift_prediction_id(conn)
    latest_prediction_id = get_latest_ready_prediction_id(conn)
    new_count = count_new_ready_predictions(conn, last_drift_prediction_id)
    conn.close()

    if latest_prediction_id == 0:
        return {
            "drift_checked": False,
            "reason": "no ready predictions in prediction_events",
            "alert_created": False,
        }

    if new_count < WINDOW_SIZE:
        return {
            "drift_checked": False,
            "latest_prediction_id": latest_prediction_id,
            "reason": f"waiting for more data ({new_count}/{WINDOW_SIZE})",
            "alert_created": False,
        }

    drift_result = compute_drift_for_latest_window()

    insert_drift_event(
        embedding_score=drift_result["embedding_score"],
        confidence_score=drift_result["confidence_score"],
        class_score=drift_result["class_score"],
        is_drift=drift_result["is_drift"],
        embedding_drifted=drift_result["embedding_drifted"],
        confidence_drifted=drift_result["confidence_drifted"],
        class_drifted=drift_result["class_drifted"],
    )

    conn = sqlite3.connect(DB_PATH)
    should_create_alert = drift_result["is_drift"] and not in_alert_cooldown(conn, cooldown_minutes)

    set_last_drift_prediction_id(conn, drift_result["prediction_ids"][-1])
    conn.commit()
    conn.close()

    alert_created = False
    if should_create_alert:
        insert_alert(
            alert_type="drift",
            message=build_alert_message(drift_result),
        )
        alert_created = True

    return {
        "drift_checked": True,
        "window_start_id": drift_result["prediction_ids"][0],
        "window_end_id": drift_result["prediction_ids"][-1],
        "embedding_score": drift_result["embedding_score"],
        "confidence_score": drift_result["confidence_score"],
        "class_score": drift_result["class_score"],
        "embedding_drifted": drift_result["embedding_drifted"],
        "confidence_drifted": drift_result["confidence_drifted"],
        "class_drifted": drift_result["class_drifted"],
        "is_drift": drift_result["is_drift"],
        "alert_created": alert_created,
    }


if __name__ == "__main__":
    result = run_orchestrator_from_db()
    print(result)