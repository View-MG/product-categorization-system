from pathlib import Path
import sqlite3
import time

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "data" / "monitoring.db"
ARCHIVE_DB_PATH = BASE_DIR / "data" / "monitoring_archive.db"


def init_archive_db(archive_db_path: str = ARCHIVE_DB_PATH):
    Path(archive_db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(archive_db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS prediction_events_archive (
            id               INTEGER PRIMARY KEY,
            timestamp        DATETIME,
            predicted_class  TEXT NOT NULL,
            confidence       FLOAT NOT NULL,
            latency_ms       FLOAT,
            brightness       FLOAT,
            blur_var         FLOAT,
            width            INTEGER,
            height           INTEGER,
            quality_warnings TEXT DEFAULT '[]',
            archived_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS human_feedback_archive (
            id             INTEGER PRIMARY KEY,
            prediction_id  INTEGER NOT NULL,
            true_label     TEXT NOT NULL,
            labeled_at     DATETIME,
            archived_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS drift_events_archive (
            id                 INTEGER PRIMARY KEY,
            timestamp          DATETIME,
            embedding_score    FLOAT,
            confidence_score   FLOAT,
            class_score        FLOAT,
            is_drift           BOOLEAN DEFAULT 0,
            embedding_drifted  BOOLEAN DEFAULT 0,
            confidence_drifted BOOLEAN DEFAULT 0,
            class_drifted      BOOLEAN DEFAULT 0,
            archived_at        DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alerts_archive (
            id          INTEGER PRIMARY KEY,
            timestamp   DATETIME,
            alert_type  TEXT NOT NULL,
            message     TEXT,
            resolved    BOOLEAN DEFAULT 0,
            archived_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print(f"✅ Archive database created at {archive_db_path}")


def archive_and_cleanup_data(
    db_path: str = DB_PATH,
    archive_db_path: str = ARCHIVE_DB_PATH,
    retention_years: int = 5
):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    cutoff = f"-{retention_years} years"

    try:
        cursor.execute("ATTACH DATABASE ? AS archive_db", (str(archive_db_path),))
        cursor.execute("BEGIN;")

        cursor.execute("""
            CREATE TEMP TABLE old_prediction_ids AS
            SELECT id
            FROM prediction_events
            WHERE timestamp < datetime('now', ?)
        """, (cutoff,))

        cursor.execute("""
            CREATE TEMP TABLE old_drift_event_ids AS
            SELECT id
            FROM drift_events
            WHERE timestamp < datetime('now', ?)
        """, (cutoff,))

        cursor.execute("""
            CREATE TEMP TABLE old_alert_ids AS
            SELECT id
            FROM alerts
            WHERE timestamp < datetime('now', ?)
        """, (cutoff,))

        cursor.execute("""
            INSERT OR IGNORE INTO archive_db.prediction_events_archive (
                id, timestamp, predicted_class, confidence, latency_ms,
                brightness, blur_var, width, height, quality_warnings, archived_at
            )
            SELECT
                id, timestamp, predicted_class, confidence, latency_ms,
                brightness, blur_var, width, height, quality_warnings, CURRENT_TIMESTAMP
            FROM prediction_events
            WHERE id IN (SELECT id FROM old_prediction_ids)
        """)

        cursor.execute("""
            INSERT OR IGNORE INTO archive_db.human_feedback_archive (
                id, prediction_id, true_label, labeled_at, archived_at
            )
            SELECT
                id, prediction_id, true_label, labeled_at, CURRENT_TIMESTAMP
            FROM human_feedback
            WHERE prediction_id IN (SELECT id FROM old_prediction_ids)
               OR labeled_at < datetime('now', ?)
        """, (cutoff,))

        cursor.execute("""
            INSERT OR IGNORE INTO archive_db.drift_events_archive (
                id, timestamp, embedding_score, confidence_score, class_score,
                is_drift, embedding_drifted, confidence_drifted, class_drifted, archived_at
            )
            SELECT
                id, timestamp, embedding_score, confidence_score, class_score,
                is_drift, embedding_drifted, confidence_drifted, class_drifted, CURRENT_TIMESTAMP
            FROM drift_events
            WHERE id IN (SELECT id FROM old_drift_event_ids)
        """)

        cursor.execute("""
            INSERT OR IGNORE INTO archive_db.alerts_archive (
                id, timestamp, alert_type, message, resolved, archived_at
            )
            SELECT
                id, timestamp, alert_type, message, resolved, CURRENT_TIMESTAMP
            FROM alerts
            WHERE id IN (SELECT id FROM old_alert_ids)
        """)

        cursor.execute("""
            DELETE FROM human_feedback
            WHERE prediction_id IN (SELECT id FROM old_prediction_ids)
               OR labeled_at < datetime('now', ?)
        """, (cutoff,))

        cursor.execute("""
            DELETE FROM prediction_events
            WHERE id IN (SELECT id FROM old_prediction_ids)
        """)

        cursor.execute("""
            DELETE FROM drift_events
            WHERE id IN (SELECT id FROM old_drift_event_ids)
        """)

        cursor.execute("""
            DELETE FROM alerts
            WHERE id IN (SELECT id FROM old_alert_ids)
        """)

        cursor.execute("DROP TABLE old_prediction_ids")
        cursor.execute("DROP TABLE old_drift_event_ids")
        cursor.execute("DROP TABLE old_alert_ids")

        cursor.execute("COMMIT;")
        print("✅ Archive and cleanup completed successfully")

    except Exception as e:
        cursor.execute("ROLLBACK;")
        print(f"❌ Error during cleanup: {e}")

    finally:
        try:
            cursor.execute("DETACH DATABASE archive_db;")
        except:
            pass
        conn.close()


def main():
    init_archive_db()
    while True:
        print("Running archive cleanup job...")
        archive_and_cleanup_data()
        print("Waiting 1 minute for next run...\n")
        time.sleep(60)


if __name__ == "__main__":
    main()