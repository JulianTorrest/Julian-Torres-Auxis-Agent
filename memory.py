import json
import os
import time
import sqlite3

class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.getenv("AUXIS_DATA_DIR", "observability"), "memory.sqlite")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY,
            mtype TEXT,
            session_id TEXT,
            key TEXT,
            value TEXT,
            checkpoint_id TEXT,
            timestamp REAL
        )""")
        self.conn.commit()

    def save(self, mtype, key, value, session_id="default", checkpoint_id=None):
        now = time.time()
        self.conn.execute("""INSERT OR REPLACE INTO memory (mtype, session_id, key, value, checkpoint_id, timestamp)
            VALUES (?,?,?,?,?,?)""",
            (mtype, session_id, key, json.dumps(value, ensure_ascii=False), checkpoint_id, now))
        self.conn.commit()

    def load(self, mtype, session_id="default", key=None, checkpoint_id=None):
        sql = "SELECT key, value, checkpoint_id, timestamp FROM memory WHERE mtype=? AND session_id=?"
        params = [mtype, session_id]
        if key:
            sql += " AND key=?"
            params.append(key)
        if checkpoint_id:
            sql += " AND checkpoint_id=?"
            params.append(checkpoint_id)
        rows = self.conn.execute(sql, params).fetchall()
        return [{"key": r[0], "value": json.loads(r[1]), "checkpoint_id": r[2], "timestamp": r[3]} for r in rows]

    def checkpoint(self, session_id="default"):
        cp = f"cp_{int(time.time())}"
        meta = {"created": time.time(), "session": session_id}
        self.conn.execute("INSERT INTO memory (mtype, session_id, key, value, checkpoint_id, timestamp) VALUES (?,?,?,?,?,?)",
            ("checkpoint", session_id, cp, json.dumps(meta, ensure_ascii=False), cp, time.time()))
        self.conn.commit()
        return cp

    def list_checkpoints(self, session_id="default"):
        rows = self.conn.execute("SELECT key, timestamp FROM memory WHERE mtype='checkpoint' AND session_id=? ORDER BY timestamp DESC", (session_id,)).fetchall()
        return [{"checkpoint_id": r[0], "timestamp": r[1]} for r in rows]

    def clear(self, session_id="default", mtype=None):
        if mtype:
            self.conn.execute("DELETE FROM memory WHERE session_id=? AND mtype=?", (session_id, mtype))
        else:
            self.conn.execute("DELETE FROM memory WHERE session_id=?", (session_id,))
        self.conn.commit()
