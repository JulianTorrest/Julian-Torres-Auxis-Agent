import os
import time
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from summarizer import summarize

class ContextManager:
    def __init__(self, llm=None, token_budget=2000, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.getenv("AUXIS_DATA_DIR", "observability"), "context.sqlite")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            role TEXT,
            content TEXT,
            topic TEXT,
            tokens INTEGER,
            timestamp REAL
        )""")
        self.conn.commit()
        self.llm = llm
        self.budget = token_budget
        self.vectorizer = TfidfVectorizer(max_features=64)

    def add(self, role, content, tokens=0):
        topic = self._topic_signature(content)
        now = time.time()
        self.conn.execute("INSERT INTO messages (role, content, topic, tokens, timestamp) VALUES (?,?,?,?,?)",
            (role, content, topic, tokens, now))
        self.conn.commit()
        self._compress_if_needed()

    def _topic_signature(self, text):
        return " ".join(text.lower().split()[:4])

    def _total_tokens(self):
        row = self.conn.execute("SELECT SUM(tokens) FROM messages").fetchone()
        return row[0] or 0

    def _compress_if_needed(self):
        while self._total_tokens() > self.budget:
            rows = self.conn.execute("SELECT id, role, content FROM messages ORDER BY timestamp LIMIT 2").fetchall()
            if len(rows) < 2:
                break
            a, b = rows
            if a[1] == "user" and b[1] == "assistant":
                pair_text = f"Pregunta: {a[2]}\nRespuesta: {b[2]}"
                summary = summarize(pair_text, self.llm) if self.llm else f"Resumen: {a[2][:80]}..."
                self.conn.execute("DELETE FROM messages WHERE id IN (?,?)", (a[0], b[0]))
                self.conn.execute("INSERT INTO messages (role, content, topic, tokens, timestamp) VALUES (?,?,?,?,?)",
                    ("system", summary, "resumen", 50, time.time()))
                self.conn.commit()
            else:
                break

    def topic_changed(self, query):
        last = self.conn.execute("SELECT topic FROM messages ORDER BY timestamp DESC LIMIT 1").fetchone()
        if not last:
            return False
        return self._topic_signature(query) != last[0]

    def get_relevant(self, query, k=6):
        all_rows = self.conn.execute("SELECT role, content FROM messages ORDER BY timestamp").fetchall()
        if not all_rows:
            return []
        texts = [r[1] for r in all_rows]
        self.vectorizer.fit(texts)
        q_vec = self.vectorizer.transform([query]).toarray()[0]
        q_norm = np.linalg.norm(q_vec) or 1
        ranked = []
        for i, t in enumerate(texts):
            t_vec = self.vectorizer.transform([t]).toarray()[0]
            sim = float(np.dot(q_vec, t_vec) / (q_norm * (np.linalg.norm(t_vec) or 1)))
            ranked.append((sim, all_rows[i]))
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [r for s, r in ranked[:k]]
        return [{"role": r[0], "content": r[1]} for r in selected]

    def build_prompt(self, query, retrieved_context, k=4):
        history = self.get_relevant(query, k)
        parts = []
        if history:
            parts.append("Historial relevante:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in history))
        if retrieved_context:
            parts.append("Contexto recuperado:\n" + "\n".join(f"- {c}" for c in retrieved_context))
        parts.append(f"Pregunta: {query}\nResponde de forma clara y concisa:")
        return "\n\n".join(parts)

    def clear(self):
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()
