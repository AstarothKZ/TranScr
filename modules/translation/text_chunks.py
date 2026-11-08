"""Small helpers for services that limit the amount of text per request."""


def split_text(text: str, max_chars: int, *, utf8_bytes: bool = False) -> list[str]:
    """Split text near line/word boundaries while preserving readable line groups."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    def size(value: str) -> int:
        return len(value.encode("utf-8")) if utf8_bytes else len(value)

    for line in text.splitlines():
        parts: list[str] = []
        remaining = line
        while size(remaining) > max_chars:
            limit = max_chars
            if utf8_bytes:
                limit = len(
                    remaining.encode("utf-8")[:max_chars].decode(
                        "utf-8", errors="ignore"
                    )
                )
                if not limit:
                    raise ValueError("max_chars cannot fit one UTF-8 character")
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            parts.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip()
        parts.append(remaining)

        for part in parts:
            candidate = f"{current}\n{part}" if current else part
            if size(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

    if current:
        chunks.append(current)
    return chunks
