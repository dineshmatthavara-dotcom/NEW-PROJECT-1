
import sqlite3
import hashlib
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "predictions.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_users_table():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def register_user(username, password):
    username = username.strip()

    if not username or len(password) < 8:
        return False, "Enter a username and a password of at least 8 characters."

    salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        200000
    ).hex()

    try:
        conn = sqlite3.connect(DB_PATH)

        conn.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt)
        )

        conn.commit()
        conn.close()

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "That username already exists."


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)

    user = conn.execute(
        "SELECT password_hash, salt FROM users WHERE username = ?",
        (username.strip(),)
    ).fetchone()

    conn.close()

    if user is None:
        return False

    stored_hash, salt = user

    entered_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        200000
    ).hex()

    return secrets.compare_digest(stored_hash, entered_hash)


create_users_table()
