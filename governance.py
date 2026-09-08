def check_policy(query: str, has_pii: bool, injection: bool) -> tuple[bool, list[str]]:
    violations = []
    if has_pii:
        violations.append("Datos personales detectados en la consulta.")
    if injection:
        violations.append("Posible prompt injection detectado.")
    blocked_terms = ["malware", "virus", "ilegal", "hack", "fraude"]
    for term in blocked_terms:
        if term in query.lower():
            violations.append(f"Termino bloqueado: {term}")
    return not violations, violations
