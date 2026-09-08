import re

_PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "telefono": r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,9}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

def redact_pii(text: str) -> tuple[str, list[str]]:
    matches = []
    for label, pattern in _PII_PATTERNS.items():
        for m in re.finditer(pattern, text):
            matches.append((m.start(), m.end(), label, m.group()))
    matches.sort(key=lambda x: x[0], reverse=True)
    out = text
    found = []
    for start, end, label, value in matches:
        out = out[:start] + f"[{label.upper()}]" + out[end:]
        found.append(f"{label}: {value}")
    return out, found
