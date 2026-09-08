import re

_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all prior",
    r"system prompt",
    r"you are now",
    r"disregard",
    r"jailbreak",
    r"DAN",
    r"\{\{.*?\}\}",
]

def is_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in _INJECTION_PATTERNS)
