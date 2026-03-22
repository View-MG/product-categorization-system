from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List

import numpy as np

DB_PATH = "monitoring.db"
REFERENCE_STATS_PATH = "reference_stats.json"
REFERENCE_EMBEDDINGS_PATH = "reference_embeddings.npz"
LATEST_N = 100
EPS = 1e-8


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def load_reference_stats(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_embeddings(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    if "embeddings" in data:
        embeddings = data["embeddings"]
    elif "sample_embeddings" in data:
        embeddings = data["sample_embeddings"]
    else:
        first_key = list(data.files)[0]
        embeddings = data[first_key]

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("reference embeddings must be 2D")
    return embeddings


def fetch_latest_prediction_ids(conn: sqlite3.Connection, n: int) -> List[int]:
    rows = conn.execute(
        """
        SELECT id
        FROM prediction_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    ids = [int(row["id"]) for row in rows]
    ids.reverse()
    return ids


def fetch_prediction_rows_by_ids(conn: sqlite3.Connection, prediction_ids: List[int]) -> List[sqlite3.Row]:
    if not prediction_ids:
        return []

    placeholders = ",".join("?" for _ in prediction_ids)
    rows = conn.execute(
        f"""
        SELECT id, predicted_class, confidence, timestamp
        FROM prediction_events
        WHERE id IN ({placeholders})
        ORDER BY id ASC
        """,
        prediction_ids,
    ).fetchall()
    return rows


def parse_embedding_json(embedding_json: str) -> np.ndarray:
    arr = np.asarray(json.loads(embedding_json), dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("embedding must be a 1D vector")
    return arr


def fetch_embeddings_for_prediction_ids(conn: sqlite3.Connection, prediction_ids: List[int]) -> np.ndarray:
    if not prediction_ids:
        raise ValueError("prediction_ids is empty")

    embedding_map: Dict[int, np.ndarray] = {}

    if table_exists(conn, "prediction_embeddings"):
        placeholders = ",".join("?" for _ in prediction_ids)
        rows = conn.execute(
            f"""
            SELECT prediction_id, embedding_json
            FROM prediction_embeddings
            WHERE prediction_id IN ({placeholders})
            """,
            prediction_ids,
        ).fetchall()

        for row in rows:
            embedding_map[int(row["prediction_id"])] = parse_embedding_json(row["embedding_json"])

    elif "embedding_json" in get_table_columns(conn, "prediction_events"):
        placeholders = ",".join("?" for _ in prediction_ids)
        rows = conn.execute(
            f"""
            SELECT id, embedding_json
            FROM prediction_events
            WHERE id IN ({placeholders})
            """,
            prediction_ids,
        ).fetchall()

        for row in rows:
            if row["embedding_json"] is not None:
                embedding_map[int(row["id"])] = parse_embedding_json(row["embedding_json"])
    else:
        raise ValueError(
            "No embedding storage found. Need either prediction_embeddings table "
            "or prediction_events.embedding_json column."
        )

    ordered = []
    for prediction_id in prediction_ids:
        if prediction_id not in embedding_map:
            raise ValueError(f"missing embedding for prediction_id={prediction_id}")
        ordered.append(embedding_map[prediction_id])

    embeddings = np.stack(ordered, axis=0).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("current embeddings must be 2D")
    return embeddings


def compute_hist_probs(values: List[float], bin_edges: List[float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=np.float64))
    probs = hist.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    probs = np.clip(probs, EPS, None)
    probs = probs / probs.sum()
    return probs


def compute_psi(current_values: List[float], ref_bin_edges: List[float], ref_bin_probs: List[float]) -> float:
    current_probs = compute_hist_probs(current_values, ref_bin_edges)
    ref_probs = np.asarray(ref_bin_probs, dtype=np.float64)
    ref_probs = np.clip(ref_probs, EPS, None)
    ref_probs = ref_probs / ref_probs.sum()

    if len(ref_probs) != len(current_probs):
        raise ValueError("reference confidence bin_probs length mismatch")

    psi = np.sum((ref_probs - current_probs) * np.log(ref_probs / current_probs))
    return float(psi)


def class_probs_from_labels(classes: List[str], labels: List[str]) -> np.ndarray:
    counts = {label: 0 for label in labels}
    for c in classes:
        if c in counts:
            counts[c] += 1

    probs = np.asarray([counts[label] for label in labels], dtype=np.float64)
    probs = probs / max(probs.sum(), 1.0)
    probs = np.clip(probs, EPS, None)
    probs = probs / probs.sum()
    return probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_jsd(current_classes: List[str], labels: List[str], ref_probs_dict: Dict[str, float]) -> float:
    p = np.asarray([float(ref_probs_dict.get(label, 0.0)) for label in labels], dtype=np.float64)
    p = np.clip(p, EPS, None)
    p = p / p.sum()

    q = class_probs_from_labels(current_classes, labels)
    m = 0.5 * (p + q)

    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(jsd)


def compute_embedding_score(
    current_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    ref_stats: Dict[str, Any],
) -> float:
    if current_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("embedding dimension mismatch")

    ref_center = reference_embeddings.mean(axis=0)
    ref_dists = np.linalg.norm(reference_embeddings - ref_center, axis=1)
    cur_dists = np.linalg.norm(current_embeddings - ref_center, axis=1)

    ref_dist_mean = float(ref_stats.get("embedding", {}).get("distance_mean", ref_dists.mean()))
    ref_dist_std = float(ref_stats.get("embedding", {}).get("distance_std", ref_dists.std()))
    ref_dist_std = max(ref_dist_std, EPS)

    score = abs(float(cur_dists.mean()) - ref_dist_mean) / ref_dist_std
    return float(score)


def build_result(
    embedding_score: float,
    confidence_score: float,
    class_score: float,
    ref_stats: Dict[str, Any],
) -> Dict[str, Any]:
    embedding_threshold = float(ref_stats["embedding"]["threshold"])
    confidence_threshold = float(ref_stats["confidence"]["psi_threshold"])
    class_threshold = float(ref_stats["class_ratio"]["jsd_threshold"])

    embedding_drifted = embedding_score > embedding_threshold
    confidence_drifted = confidence_score > confidence_threshold
    class_drifted = class_score > class_threshold
    is_drift = embedding_drifted or confidence_drifted or class_drifted

    return {
        "embedding_score": round(float(embedding_score), 6),
        "confidence_score": round(float(confidence_score), 6),
        "class_score": round(float(class_score), 6),
        "is_drift": bool(is_drift),
        "embedding_drifted": bool(embedding_drifted),
        "confidence_drifted": bool(confidence_drifted),
        "class_drifted": bool(class_drifted),
    }


def calculate_drift_from_prediction_ids(
    conn: sqlite3.Connection,
    prediction_ids: List[int],
    ref_stats: Dict[str, Any],
    ref_embeddings: np.ndarray,
) -> Dict[str, Any]:
    rows = fetch_prediction_rows_by_ids(conn, prediction_ids)
    if not rows:
        raise ValueError("no prediction rows found")

    confidences = [float(row["confidence"]) for row in rows]
    classes = [str(row["predicted_class"]) for row in rows]
    current_embeddings = fetch_embeddings_for_prediction_ids(conn, prediction_ids)

    confidence_score = compute_psi(
        current_values=confidences,
        ref_bin_edges=ref_stats["confidence"]["bin_edges"],
        ref_bin_probs=ref_stats["confidence"]["bin_probs"],
    )

    class_score = compute_jsd(
        current_classes=classes,
        labels=ref_stats["labels"],
        ref_probs_dict=ref_stats["class_ratio"]["reference_probs"],
    )

    embedding_score = compute_embedding_score(
        current_embeddings=current_embeddings,
        reference_embeddings=ref_embeddings,
        ref_stats=ref_stats,
    )

    return build_result(
        embedding_score=embedding_score,
        confidence_score=confidence_score,
        class_score=class_score,
        ref_stats=ref_stats,
    )


def insert_drift_event(conn: sqlite3.Connection, result: Dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO drift_events (
            embedding_score,
            confidence_score,
            class_score,
            is_drift,
            embedding_drifted,
            confidence_drifted,
            class_drifted
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["embedding_score"],
            result["confidence_score"],
            result["class_score"],
            int(result["is_drift"]),
            int(result["embedding_drifted"]),
            int(result["confidence_drifted"]),
            int(result["class_drifted"]),
        ),
    )
    return int(cur.lastrowid)


def compute_and_store_latest_window(
    db_path: str = DB_PATH,
    reference_stats_path: str = REFERENCE_STATS_PATH,
    reference_embeddings_path: str = REFERENCE_EMBEDDINGS_PATH,
    latest_n: int = LATEST_N,
) -> Dict[str, Any]:
    ref_stats = load_reference_stats(reference_stats_path)
    ref_embeddings = load_reference_embeddings(reference_embeddings_path)

    with get_conn(db_path) as conn:
        prediction_ids = fetch_latest_prediction_ids(conn, latest_n)
        if len(prediction_ids) < latest_n:
            raise ValueError(f"not enough prediction rows: need {latest_n}, got {len(prediction_ids)}")

        result = calculate_drift_from_prediction_ids(conn, prediction_ids, ref_stats, ref_embeddings)
        drift_event_id = insert_drift_event(conn, result)
        conn.commit()

    return {
        "prediction_ids": prediction_ids,
        "drift_event_id": drift_event_id,
        **result,
    }


def main() -> None:
    result = compute_and_store_latest_window()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()