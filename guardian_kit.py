import re
import os
import time
import sqlite3
import hashlib
import unicodedata
from typing import List, Dict, Tuple

def _normalize(text):
    return unicodedata.normalize("NFKC", text)

ENCODING_RE = re.compile(r"(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|%[0-9a-fA-F]{2}|&#x[0-9a-fA-F]+;)")

def detect_suspicious_encoding(text: str) -> List[str]:
    reasons = []
    n = _normalize(text)
    if ENCODING_RE.search(n):
        reasons.append("Caracteres escapados/codificados")
    if re.search(r"[\u0400-\u04FF\u0370-\u03FF]", n):
        reasons.append("Caracteres cirilicos/griegos (homoglifos)")
    if "\u200b" in n or "\u200c" in n or "\u200d" in n:
        reasons.append("Caracteres de ancho cero")
    if any(c != n for c, n in zip(text, n)):
        reasons.append("Normalizacion NFKC cambio el texto")
    return reasons

SQL_RE = re.compile(r"\b(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE\s+TABLE|UNION\s+SELECT|INSERT\s+INTO|UPDATE\s+\w+\s+SET|;--|')\b", re.I)

def detect_dangerous_sql(text: str) -> List[str]:
    return [f"SQL peligroso: {m.strip()}" for m in SQL_RE.findall(text)]

PROHIBITED_TOPICS = {
    "malware": ["malware", "virus", "troyano", "ransomware", "rootkit"],
    "ilegal": ["ilegal", "fraude", "hackear", "exploit", "vulnerar", "robar datos"],
    "violencia": ["asesinar", "secuestrar", "bomba", "atentado", "disparar"],
    "autolesion": ["suicidio", "autolesion", "matarme", "morirme"],
    "sexual": ["pornografia", "sexo explicito", "menores", "grooming", "desnudo"],
    "politica_sensible": ["golpe de estado", "derrocar", "extremista"],
}

def detect_prohibited_topics(text: str) -> List[str]:
    t = text.lower()
    reasons = []
    for topic, words in PROHIBITED_TOPICS.items():
        for w in words:
            if w in t:
                reasons.append(f"Topico prohibido '{topic}': {w}")
    return reasons

MODERATION_CATEGORIES = {
    "toxicidad": ["idiota", "estupido", "imbecil", "mierda", "basura", "maldito"],
    "hate_speech": ["odio", "racista", "nazi", "xenofobo", "supremacista"],
    "acoso": ["acosar", "hostigar", "acoso", "perseguir", " Amenazar"],
    "critica": ["pesimo", "horrible", "inutil", "mentiroso", "estafa", "fraude"],
    "autolesion": ["suicidio", "autolesion", "matarme", "cortarme"],
    "contenido_sexual": ["sexo", "desnudo", "pornografia", "erotico"],
    "spam": ["www.", "http", "compra ahora", "oferta", "click aqui", "gratis"],
}

def moderate_content(text: str, whitelist: List[str] = None) -> List[str]:
    t = text.lower()
    flags = []
    for category, words in MODERATION_CATEGORIES.items():
        for w in words:
            if w in t:
                flags.append(category)
                break
    if whitelist:
        tokens = set(re.findall(r"\w+", t))
        allowed = {w.lower() for w in whitelist}
        if tokens and not tokens.issubset(allowed):
            flags.append("whitelist")
    return list(set(flags))

CREDENTIAL_RE = re.compile(r"\b(sk-[a-zA-Z0-9]{20,}|[A-Za-z0-9_]{32,64}|[A-Fa-f0-9]{64})\b")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def detect_credentials(text: str) -> List[str]:
    return [m for m in CREDENTIAL_RE.findall(text)]

def detect_pii(text: str) -> List[str]:
    return list(set(EMAIL_RE.findall(text) + PHONE_RE.findall(text) + SSN_RE.findall(text)))

def output_guard(text: str, whitelist: List[str] = None) -> List[str]:
    reasons = []
    creds = detect_credentials(text)
    if creds:
        reasons.append("credenciales")
    pii = detect_pii(text)
    if pii:
        reasons.append("pii")
    reasons.extend(moderate_content(text, whitelist))
    return reasons

def redact(text: str) -> Tuple[str, List[str]]:
    found = []
    for m in EMAIL_RE.finditer(text):
        found.append(m.group())
    for m in PHONE_RE.finditer(text):
        found.append(m.group())
    for m in SSN_RE.finditer(text):
        found.append(m.group())
    out = text
    for v in found:
        out = out.replace(v, "[REDACTED]")
    return out, found

class RateLimiter:
    def __init__(self, db_path=None, max_requests=10, window_seconds=60):
        if db_path is None:
            db_path = os.path.join(os.getenv("AUXIS_DATA_DIR", "observability"), "rate_limits.sqlite")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS requests (user TEXT, timestamp REAL)")
        self.conn.commit()
        self.max_requests = max_requests
        self.window = window_seconds

    def check(self, user: str) -> Tuple[bool, str]:
        since = time.time() - self.window
        self.conn.execute("DELETE FROM requests WHERE timestamp < ?", (since,))
        self.conn.commit()
        count = self.conn.execute("SELECT COUNT(*) FROM requests WHERE user=?", (user,)).fetchone()[0]
        if count >= self.max_requests:
            return False, f"Rate limit: {count}/{self.max_requests} en {self.window}s"
        self.conn.execute("INSERT INTO requests VALUES (?, ?)", (user, time.time()))
        self.conn.commit()
        return True, ""

class InputGuardian:
    def __init__(self, max_requests=10, window_seconds=60, whitelist=None):
        self.rate = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
        self.whitelist = whitelist or []

    def check(self, user: str, query: str) -> Dict:
        ok, reason = self.rate.check(user)
        result = {"allowed": True, "rejected": False, "reasons": []}
        if not ok:
            result["allowed"] = False
            result["rejected"] = True
            result["reasons"].append(reason)
        result["encoding"] = detect_suspicious_encoding(query)
        result["sql"] = detect_dangerous_sql(query)
        result["topics"] = detect_prohibited_topics(query)
        result["moderation"] = moderate_content(query, self.whitelist)
        if result["encoding"]:
            result["reasons"].append("encoding sospechoso")
        if result["sql"]:
            result["reasons"].append("sql peligroso")
        if result["topics"]:
            result["reasons"].append("topicos prohibidos")
        if result["moderation"]:
            result["reasons"].append(f"moderacion: {', '.join(result['moderation'])}")
        if result["reasons"]:
            result["allowed"] = False
            result["rejected"] = True
        return result
