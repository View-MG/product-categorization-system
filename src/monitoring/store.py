# src/monitoring/store.py

import sqlite3
import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "monitoring.db"

def init_db(db_path: str = DB_PATH):
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
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
            quality_warnings TEXT DEFAULT '[]',
            image_data_url   TEXT

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
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database created at {db_path}")


def insert_prediction(
    db_path: str = DB_PATH,
    predicted_class: str = None,
    confidence: float = None,
    latency_ms: float = None,
    brightness: float = None,
    blur_var: float = None,
    width: int = None,
    height: int = None,
    quality_warnings: list = [],
    image_data_url: str = None,
) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_events (
            predicted_class, confidence, latency_ms,
            brightness, blur_var, width, height, quality_warnings, image_data_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        predicted_class,
        confidence,
        latency_ms,
        brightness,
        blur_var,
        width,
        height,
        json.dumps(quality_warnings),
        image_data_url,   
    ))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id  


def insert_feedback(
    prediction_id: int,
    true_label: str,
    db_path: str = DB_PATH
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO human_feedback (prediction_id, true_label)
        VALUES (?, ?)
    """, (prediction_id, true_label))

    conn.commit()
    conn.close()


def insert_drift_event(
    embedding_score: float,
    confidence_score: float,
    class_score: float,
    is_drift: bool,
    embedding_drifted: bool,
    confidence_drifted: bool,
    class_drifted: bool,
    db_path: str = DB_PATH
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO drift_events (
            embedding_score, confidence_score, class_score,
            is_drift, embedding_drifted, confidence_drifted, class_drifted
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        embedding_score, confidence_score, class_score,
        is_drift, embedding_drifted, confidence_drifted, class_drifted
    ))

    conn.commit()
    conn.close()


def insert_alert(
    alert_type: str,
    message: str,
    db_path: str = DB_PATH
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO alerts (alert_type, message)
        VALUES (?, ?)
    """, (alert_type, message))

    conn.commit()
    conn.close()


# ทดสอบว่าใช้งานได้
if __name__ == "__main__":
    # สร้าง database
    init_db()

    # ทดลอง insert ข้อมูลตัวอย่าง
    pid = insert_prediction(
        predicted_class="beverages",
        confidence=0.45,
        latency_ms=120,
        brightness=30.1,
        blur_var=12.4,
        width=640,
        height=480,
        quality_warnings=["low_brightness", "low_blur"],
        image_data_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWAAAADFCAYAAABqxyilAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAAK3SURBVHhe7dQxAQAgDMCwgX/P8COAPslZAV0zcwaA7/YbAPjDgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAiAEDRAwYIGLAABEDBogYMEDEgAEiBgwQMWCAyAUSjgKJIlXmJAAAAABJRU5ErkJggg=="
    )
    print(f"✅ Inserted prediction id = {pid}")

    insert_feedback(
        prediction_id=pid,
        true_label="snacks"
    )
    print(f"✅ Inserted feedback for prediction {pid}")