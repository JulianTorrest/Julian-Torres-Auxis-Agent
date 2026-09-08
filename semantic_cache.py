import json
import os
import time
import sqlite3
import hashlib
import numpy as np

class SemanticCache:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.getenv("AUXIS_DATA_DIR", "observability"), "semantic_cache.sqlite")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS cache (
            id TEXT PRIMARY KEY,
            query_hash TEXT,
            query_text TEXT,
            embedding BLOB,
            answer TEXT,
            provider TEXT,
            cache_type TEXT,
            expires_at REAL,
            created_at REAL,
            alerts INTEGER
        )""")
        self.conn.commit()

    def _hash(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def _now(self):
        return time.time()

    def _cleanup(self, cache_type=None):
        now = self._now()
        if cache_type:
            self.conn.execute("DELETE FROM cache WHERE expires_at < ? AND cache_type=?", (now, cache_type))
        else:
            self.conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
        self.conn.commit()

    def get_exact(self, query, cache_type="general"):
        self._cleanup(cache_type)
        h = self._hash(query)
        row = self.conn.execute("SELECT answer, provider, cache_type FROM cache WHERE query_hash=? AND cache_type=?", (h, cache_type)).fetchone()
        if row:
            return {"hit": "exact", "answer": row[0], "provider": row[1], "cache_type": row[2]}
        return None

    def get_semantic(self, query, embed_fn, threshold=0.92, cache_type="general"):
        self._cleanup(cache_type)
        q = embed_fn(query)
        q = q / (np.linalg.norm(q) + 1e-10)
        rows = self.conn.execute("SELECT query_text, embedding, answer, provider, cache_type FROM cache WHERE cache_type=?", (cache_type,)).fetchall()
        best = (0, None, None, None)
        for _, emb_blob, ans, prov, ct in rows:
            emb = np.frombuffer(emb_blob, dtype="float32")
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(q, emb))
            if sim > best[0]:
                best = (sim, ans, prov, ct)
        if best[0] >= threshold:
            return {"hit": "semantic", "similarity": best[0], "answer": best[1], "provider": best[2], "cache_type": best[3]}
        return None

    def put(self, query, answer, provider, embed_fn, cache_type="general", ttl=300, alert=False):
        h = self._hash(query)
        emb = embed_fn(query).astype("float32").tobytes()
        now = self._now()
        self.conn.execute("""INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (h, h, query, emb, answer, provider, cache_type, now + ttl, now, int(alert)))
        self.conn.commit()
        return {"cached": True, "expires_at": now + ttl, "alert": alert}

    def invalidate(self, cache_type=None):
        if cache_type:
            self.conn.execute("DELETE FROM cache WHERE cache_type=?", (cache_type,))
        else:
            self.conn.execute("DELETE FROM cache")
        self.conn.commit()

    def stats(self):
        rows = self.conn.execute("SELECT cache_type, COUNT(*), SUM(alerts) FROM cache GROUP BY cache_type").fetchall()
        return {r[0]: {"count": r[1], "alerts": r[2] or 0} for r in rows}
