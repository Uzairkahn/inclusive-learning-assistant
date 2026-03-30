# utils/db.py
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Users table - role column hata diya
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT
        )
    """)
    
    
    # Transcripts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password, name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        created_at = datetime.now().isoformat()
        cur.execute(
            "INSERT INTO users (email, password, name, created_at) VALUES (?, ?, ?, ?)",
            (email, hashed_pw, name, created_at)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    hashed_pw = hash_password(password)
    cur.execute("SELECT id, name FROM users WHERE email = ? AND password = ?", (email, hashed_pw))
    user = cur.fetchone()
    conn.close()
    return user  # Returns (id, name) if valid, else None