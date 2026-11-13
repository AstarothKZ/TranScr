"""Small helpers for services that limit the amount of text per request."""


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text near line/word boundaries while preserving readable line groups."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not isinstance(text, str):
        raise TypeError("Текст должен быть строкой")
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    for line in text.splitlines():
        parts: list[str] = []
        remaining = line
        while len(remaining) > max_chars:
            limit = max_chars
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            parts.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        parts.append(remaining)

        for part in parts:
            candidate = f"{current}\n{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]
