import sqlite3
import json
import os

class StendStore:
    """
    SQLite-based Key-Value store for bot configurations and user data.
    Port of legacy PyKV from irispy-client.
    """
    def __init__(self, db_path="stend_store.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def put(self, key, value):
        # Serialize if not string
        if not isinstance(value, str):
            value = json.dumps(value)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)", (key, value))
            conn.commit()
        return True

    def get(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                val = row[0]
                try:
                    return json.loads(val)
                except:
                    return val
        return None

    def delete(self, key):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM store WHERE key = ?", (key,))
            conn.commit()
        return True

    def list_keys(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key FROM store")
            return [row[0] for row in cursor.fetchall()]

    def search_key(self, keyword):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key FROM store WHERE key LIKE ?", (f"%{keyword}%",))
            return [row[0] for row in cursor.fetchall()]
