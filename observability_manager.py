import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional

try:
    from langsmith import Client as LangSmithClient
    HAS_LANGSMITH = True
except Exception:
    HAS_LANGSMITH = False

try:
    from langfuse import Langfuse
    HAS_LANGFUSE = True
except Exception:
    HAS_LANGFUSE = False

class TraceStore:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.getenv("AUXIS_DATA_DIR", "observability"), "traces.sqlite")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        schema = """
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            user TEXT,
            query TEXT,
            redacted TEXT,
            pii_found TEXT,
            injection INTEGER,
            policy_ok INTEGER,
            violations TEXT,
            context TEXT,
            answer TEXT,
            answer_provider TEXT,
            mode TEXT,
            selected_provider TEXT,
            max_retries INTEGER,
            latency_ms REAL,
            tokens_input INTEGER,
            tokens_output INTEGER,
            rejected INTEGER,
            rejection_reason TEXT,
            tool_used TEXT,
            moderation_flags TEXT
        """
        existing = [c[1] for c in self._conn.execute("PRAGMA table_info(traces)")]
        if existing and len(existing) != 21:
            self._conn.execute("DROP TABLE IF EXISTS traces")
            existing = []
        if not existing:
            self._conn.execute(f"CREATE TABLE traces ({schema})")
            self._conn.commit()

    def _migrate(self):
        existing = {c[1] for c in self._conn.execute("PRAGMA table_info(traces)")}
        cols = [
            ("tokens_input", "INTEGER", 0),
            ("tokens_output", "INTEGER", 0),
            ("rejected", "INTEGER", 0),
            ("rejection_reason", "TEXT", "''"),
            ("tool_used", "TEXT", "''"),
            ("moderation_flags", "TEXT", "'[]'"),
        ]
        for name, ctype, default in cols:
            if name not in existing:
                try:
                    self._conn.execute(f"ALTER TABLE traces ADD COLUMN {name} {ctype} DEFAULT {default}")
                    self._conn.commit()
                except Exception:
                    pass

    def log(self, trace: Dict):
        self._conn.execute("""
            INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trace["id"],
            trace["timestamp"],
            trace["user"],
            trace["query"],
            trace["redacted"],
            json.dumps(trace.get("pii_found", []), ensure_ascii=False),
            int(trace.get("injection", False)),
            int(trace.get("policy_ok", False)),
            json.dumps(trace.get("violations", []), ensure_ascii=False),
            json.dumps(trace.get("context", []), ensure_ascii=False),
            trace.get("answer", ""),
            trace.get("answer_provider", ""),
            trace.get("mode", ""),
            trace.get("selected_provider", ""),
            trace.get("max_retries", 0),
            trace.get("latency_ms", 0.0),
            trace.get("tokens_input", 0),
            trace.get("tokens_output", 0),
            int(trace.get("rejected", False)),
            trace.get("rejection_reason", ""),
            trace.get("tool_used", ""),
            json.dumps(trace.get("moderation_flags", []), ensure_ascii=False),
        ))
        self._conn.commit()
        return trace

    def list(self, user: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        where = "WHERE user = ?" if user else ""
        params = (user,) if user else ()
        rows = self._conn.execute(
            f"SELECT * FROM traces {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + (limit, offset)
        ).fetchall()
        cols = [d[1] for d in self._conn.execute("PRAGMA table_info(traces)")]
        return [dict(zip(cols, r)) for r in rows]

    def get_audit_report(self, user: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict]:
        where = []
        params = []
        if user:
            where.append("user = ?")
            params.append(user)
        if start:
            where.append("timestamp >= ?")
            params.append(start)
        if end:
            where.append("timestamp <= ?")
            params.append(end)
        clause = "WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(f"SELECT * FROM traces {clause} ORDER BY timestamp DESC", params).fetchall()
        cols = [d[1] for d in self._conn.execute("PRAGMA table_info(traces)")]
        return [dict(zip(cols, r)) for r in rows]

    def get_moderation_stats(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict:
        where = []
        params = []
        if start:
            where.append("timestamp >= ?")
            params.append(start)
        if end:
            where.append("timestamp <= ?")
            params.append(end)
        clause = "WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(f"SELECT moderation_flags, rejected, rejection_reason FROM traces {clause}", params).fetchall()
        stats = {"total": len(rows), "rejected": 0, "by_category": {}, "by_rejection_reason": {}}
        for flags, rej, reason in rows:
            try:
                fl = json.loads(flags) if flags else []
            except Exception:
                fl = []
            for f in fl:
                stats["by_category"][f] = stats["by_category"].get(f, 0) + 1
            if rej:
                stats["rejected"] += 1
                r = reason or "unknown"
                stats["by_rejection_reason"][r] = stats["by_rejection_reason"].get(r, 0) + 1
        return stats

class LangSmithObserver:
    def __init__(self):
        key = os.getenv("LANGSMITH_API_KEY")
        if not key:
            raise RuntimeError("LANGSMITH_API_KEY no configurada")
        self.client = LangSmithClient(api_key=key)

    def log(self, trace: Dict):
        try:
            self.client.create_run(
                name="agent_query",
                run_type="chain",
                inputs={"query": trace["query"], "redacted": trace["redacted"]},
                outputs={"answer": trace["answer"], "provider": trace["answer_provider"]},
                start_time=trace["timestamp"],
            )
        except Exception:
            pass

class LangFuseObserver:
    def __init__(self):
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        if not secret or not public:
            raise RuntimeError("LANGFUSE_SECRET_KEY o LANGFUSE_PUBLIC_KEY no configuradas")
        self.langfuse = Langfuse(public_key=public, secret_key=secret, host=host)

    def log(self, trace: Dict):
        try:
            t = self.langfuse.trace(
                name="agent_query",
                user_id=trace["user"],
                input={"query": trace["query"]},
            )
            t.generation(
                name="generate",
                input={"redacted": trace["redacted"]},
                output={"answer": trace["answer"], "provider": trace["answer_provider"]},
            )
        except Exception:
            pass

class ObservabilityManager:
    def __init__(self):
        self.local = TraceStore()
        self.observers = []
        if HAS_LANGSMITH and os.getenv("LANGSMITH_API_KEY"):
            try:
                self.observers.append(LangSmithObserver())
            except Exception:
                pass
        if HAS_LANGFUSE and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                self.observers.append(LangFuseObserver())
            except Exception:
                pass

    def log(self, state: Dict, user: str, latency_ms: float, tokens_input: int = 0, tokens_output: int = 0,
            rejected: bool = False, rejection_reason: str = "", tool_used: str = "", moderation_flags: List[str] = None) -> Dict:
        trace = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "user": user,
            "query": state.get("query", ""),
            "redacted": state.get("redacted", ""),
            "pii_found": state.get("pii_found", []),
            "injection": state.get("injection", False),
            "policy_ok": state.get("policy_ok", False),
            "violations": state.get("violations", []),
            "context": state.get("context", []),
            "answer": state.get("answer", ""),
            "answer_provider": state.get("answer_provider", ""),
            "mode": state.get("mode", ""),
            "selected_provider": state.get("selected_provider", ""),
            "max_retries": state.get("max_retries", 0),
            "latency_ms": round(latency_ms, 2),
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "tool_used": tool_used,
            "moderation_flags": moderation_flags or [],
        }
        self.local.log(trace)
        for obs in self.observers:
            try:
                obs.log(trace)
            except Exception:
                pass
        return trace

    def get_audit_report(self, **kwargs):
        return self.local.get_audit_report(**kwargs)

    def get_moderation_stats(self, **kwargs):
        return self.local.get_moderation_stats(**kwargs)
