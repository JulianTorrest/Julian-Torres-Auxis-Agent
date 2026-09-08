def chunk_text(text: str, size: int = 200, overlap: int = 20) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def chunk_by_sections(text: str, min_size: int = 100) -> list[str]:
    raw = [s.strip() for s in text.split("\n\n") if s.strip()]
    sections = []
    current = ""
    for s in raw:
        if len(current) < min_size:
            current = f"{current}\n\n{s}".strip() if current else s
        else:
            if current:
                sections.append(current)
            current = s
    if current:
        sections.append(current)
    return sections
