import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "reviews.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pnr TEXT UNIQUE,
            review TEXT,
            sentiment TEXT,
            confidence REAL,
            category TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()