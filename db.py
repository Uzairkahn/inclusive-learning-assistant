import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "users.db"
STATS_ROW_ID = 1
STAT_COLUMNS = {"summaries", "uploads", "audio", "translations"}


def get_db_connection():
    """
    Create a SQLite connection for shared app data.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_stats_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY,
            summaries INTEGER DEFAULT 0,
            uploads INTEGER DEFAULT 0,
            audio INTEGER DEFAULT 0,
            translations INTEGER DEFAULT 0
        )
        """
    )


def _ensure_stats_row(cursor):
    cursor.execute(
        """
        INSERT OR IGNORE INTO stats (id, summaries, uploads, audio, translations)
        VALUES (?, 0, 0, 0, 0)
        """,
        (STATS_ROW_ID,),
    )


def _row_to_stats(row):
    if row is None:
        return {
            "summaries": 0,
            "uploads": 0,
            "audio": 0,
            "translations": 0,
        }

    return {
        "summaries": int(row["summaries"]),
        "uploads": int(row["uploads"]),
        "audio": int(row["audio"]),
        "translations": int(row["translations"]),
    }


def _fetch_stats_row(cursor):
    cursor.execute(
        """
        SELECT summaries, uploads, audio, translations
        FROM stats
        WHERE id = ?
        """,
        (STATS_ROW_ID,),
    )
    return cursor.fetchone()


def init_stats_table():
    """
    Create the stats table and its singleton row if they do not exist yet.
    """
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        _ensure_stats_table(cursor)
        _ensure_stats_row(cursor)
        connection.commit()
        print(f"[stats] Initialized stats table at {DB_PATH}")
    finally:
        connection.close()


def get_stats():
    """
    Return the current dashboard counters from the singleton stats row.
    """
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        _ensure_stats_table(cursor)
        _ensure_stats_row(cursor)
        row = _fetch_stats_row(cursor)
        connection.commit()
        stats = _row_to_stats(row)
        print(f"[stats] Current dashboard stats: {stats}")
        return stats
    finally:
        connection.close()


def _increment_stat(column_name):
    if column_name not in STAT_COLUMNS:
        raise ValueError(f"Unsupported stat column: {column_name}")

    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        _ensure_stats_table(cursor)
        _ensure_stats_row(cursor)
        print(f"[stats] increment_{column_name} called")
        cursor.execute(
            f"""
            UPDATE stats
            SET {column_name} = {column_name} + 1
            WHERE id = ?
            """,
            (STATS_ROW_ID,),
        )
        row = _fetch_stats_row(cursor)
        connection.commit()
        updated_stats = _row_to_stats(row)
        print(f"[stats] Updated dashboard stats: {updated_stats}")
        return updated_stats
    finally:
        connection.close()


def increment_summaries():
    _increment_stat("summaries")


def increment_uploads():
    _increment_stat("uploads")


def increment_audio():
    _increment_stat("audio")


def increment_translations():
    _increment_stat("translations")
