# utils/db.py
import sqlite3
from pathlib import Path
from datetime import datetime
import bcrypt

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
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

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
    cur.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    
    if user:
        user_id, name, stored_hash = user
        try:
            # Convert stored_hash to bytes if it's a string (for bcrypt compatibility)
            if isinstance(stored_hash, str):
                stored_hash = stored_hash.encode()
            if bcrypt.checkpw(password.encode(), stored_hash):
                return (user_id, name)
        except (ValueError, TypeError):
            pass
    
    return None