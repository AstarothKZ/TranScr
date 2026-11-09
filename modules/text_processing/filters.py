"""Small, conservative filters for screen-translation OCR results."""

import re
import unicodedata

# Numeric labels, scores and numeric-only dates need no translation.
_NUMBERS_AND_SEPARATORS = re.compile(r"^[\s\d.,:/\\\-–—+()%$€£¥₽#№]+$")
_COMPACT_CODE = re.compile(r"^[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*$")
_CONTACT = re.compile(r"^(?:https?://|www\.)?[^\s/@]+(?:@[^\s/]+)?\.[A-Za-z]{2,}(?:/\S*)?$", re.IGNORECASE)
_DIMENSIONS = re.compile(r"^\d+\s*[xх×✕*]\s*\d+(?:\s*[xх×✕*]\s*\d+)?(?:\s*(?:px|пкс|пикс))?$", re.IGNORECASE)
_SIZE_LABEL = re.compile(
    r"(?:[BВБ▣□]\s+)?[+-]?\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?\s*"
    r"(?:[kmgtpeкмгтпе]\s*[iи]?\s*[bб6]|[bб]|bytes?|байт(?:а|ов)?)(?:\s*/\s*(?:s|sec|с|сек))?",
    re.IGNORECASE,
)
# Only standalone UI tokens: preserve these words within actual sentences.
_UI_LABELS = {
    "PNG",
    "JPG",
    "JPEG",
    "GIF",
    "BMP",
    "WEBP",
    "TIFF",
    "SVG",
    "PDF",
    "PC",
    "РУС",
    "RUS",
    "PYC",
    "ENG",
    "EN",
    "RU",
}
_LATIN_PHRASE = re.compile(
    r"[A-Za-z]+(?:[-'][A-Za-z]+)*(?:[ \t]+[A-Za-z]+(?:[-'][A-Za-z]+)*)*"
)


def is_contact_address(text: str) -> bool:
    return _CONTACT.fullmatch(text.strip()) is not None


def is_ui_label(text: str) -> bool:
    return text.strip(' [](){}|·•:;.').upper() in _UI_LABELS


def untranslated_fragments(text: str) -> list[tuple[int, int]]:
    """Conservative candidates in a mixed Russian/English service response.

    Preserve capitalized names and abbreviations; this is not language detection.
    """
    if not re.search(r"[А-Яа-яЁё]", text):
        return []
    result = []
    for match in _LATIN_PHRASE.finditer(text):
        words = re.findall(r"[A-Za-z]+", match.group())
        if all(word.islower() for word in words) and (
            len(words) >= 2 or len(words[0]) >= 8
        ):
            result.append(match.span())
    return result


def should_translate(text: str) -> bool:
    """Reject numeric data and identifiers; dates containing words translate.

    This intentionally filters only a whole OCR region. It will not remove a
    code embedded inside a meaningful sentence, such as ``The A-4SU entered``.
    """
    value = " ".join((text or "").split()).strip()
    if is_contact_address(value):
        return False
    # Filename/status labels are often returned as one OCR line by Paddle.
    label = unicodedata.normalize('NFKC', value).strip(' [](){}|·•:;')
    if re.fullmatch(r'(?:[^\s]+\.)?(?:png|jpe?g|gif|webp|bmp|tiff|pdf)(?:\s*[|·•]\s*[\d.,]+\s*(?:kb|mb|gb))?', label, re.IGNORECASE):
        return False
    if not value or not any(c.isalnum() for c in value):
        return False
    if sum(c.isalnum() for c in value) <= 2:
        return False

    if is_ui_label(value):
        return False
    if _NUMBERS_AND_SEPARATORS.fullmatch(value):
        return False
    # OCR may return several adjacent serial/model cells as one line. A row
    # made entirely of numeric identifiers remains data regardless of spacing.
    tokens = value.split()
    if all(_COMPACT_CODE.fullmatch(token) and any(c.isdigit() for c in token)
           for token in tokens):
        return False
    if _DIMENSIONS.fullmatch(value):
        return False
    if _SIZE_LABEL.fullmatch(value.rstrip(" .;:")):
        return False

    return any(char.isalpha() for char in value)


def _comparison_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    # Ignore whitespace and punctuation differences: a result that only fixes
    # spacing or punctuation is not useful enough to paint on the screen.
    return "".join(char for char in normalized if char.isalnum())


def translation_changed(source: str, translated: str) -> bool:
    """True only if the provider returned non-empty, meaningfully changed text."""
    if not translated or not translated.strip():
        return False
    source_key = _comparison_key(source)
    translated_key = _comparison_key(translated)
    return bool(translated_key) and source_key != translated_key
