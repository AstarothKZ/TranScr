"""Small, conservative filters for screen-translation OCR results."""

import re
import unicodedata
from urllib.parse import urlsplit

# Numeric dates remain data; dates with a written-out month are translatable.
_NUMBERS_AND_SEPARATORS = re.compile(r"^[\s\d.,:/\\\-–—+()%$€£¥₽#№]+$")
_NUMERIC_DATE = re.compile(r"^(?:\d{1,4}[./-]){1,2}\d{1,4}$|^\d{1,2}:\d{2}(?::\d{2})?$")
_NUMBER_LABEL = re.compile(r"(?:no\.?|#|№)\s*\d+(?:[-./]\d+)*", re.IGNORECASE)
_MONTH_DATE = re.compile(
    r"^(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\.?\s+\d{1,2}(?:,?\s+\d{2,4})?|"
    r"\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s*\d{0,4})$",
    re.IGNORECASE,
)
_FULL_MONTH = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)
_COMPACT_CODE = re.compile(r"[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*")
_CODE_DASHES = str.maketrans("‐‑‒–—−", "------")
_HOST = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
_EMAIL = re.compile(r"[^\s@<>]+@" + _HOST, re.IGNORECASE)
_DOMAIN_ADDRESS = re.compile(_HOST + r"(?::\d{1,5})?(?:[/?#]\S*)?", re.IGNORECASE)
_FILE_LABEL = re.compile(
    r"(?:[^\s]+\.)?(?:png|jpe?g|gif|webp|bmp|tiff|pdf)"
    r"(?:\s*[|·•]\s*[\d.,]+\s*(?:kb|mb|gb))?",
    re.IGNORECASE,
)
_DIMENSIONS = re.compile(
    r"^\d+\s*[xх×✕*]\s*\d+(?:\s*[xх×✕*]\s*\d+)?(?:\s*(?:px|пкс|пикс))?$", re.IGNORECASE
)
_LATIN_LINE = re.compile(r"[A-Za-z\s.,;:'\"!?()\[\]{}<>/\\|~`^*+_=\-]+")
_VOWELS = frozenset("aeiou")
_LETTERS_OR_DIGITS = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")
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
# Explicit non-translatable technical field labels, not a general abbreviation
# guesser. Match the entire source only; prose mentioning these still translates.
_TECHNICAL_LABELS = {"buno", "c/n", "s/n", "p/n"}
_TABLE_CODE = re.compile(
    r"(?:[A-Za-z]{1,4}(?:[-_/][A-Za-z]{1,4})+|[A-Z]{1,4}|[a-z]{3}|[A-Za-z])"
)
_LATIN_PHRASE = re.compile(
    r"[A-Za-z]+(?:[-'][A-Za-z]+)*(?:[ \t]+[A-Za-z]+(?:[-'][A-Za-z]+)*)*"
)


def _bare_token(text: str) -> str:
    """Ignore enclosing punctuation for classification, preserving source text."""
    value = text.strip().rstrip(".,;!?:")
    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
        "<": ">",
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
    }
    while len(value) >= 2 and pairs.get(value[0]) == value[-1]:
        value = value[1:-1].strip().rstrip(".,;!?:")
    return value


def is_contact_address(text: str) -> bool:
    value = _bare_token(text)
    if not value or any(char.isspace() for char in value):
        return False
    if value.lower().startswith("mailto:"):
        return _EMAIL.fullmatch(value[7:].split("?", 1)[0]) is not None
    if _EMAIL.fullmatch(value):
        return True
    if _DOMAIN_ADDRESS.fullmatch(value):
        return True
    if value.lower().startswith(("http://", "https://", "ftp://")):
        try:
            address = urlsplit(value)
            return bool(address.hostname) and (
                address.port is None or 0 < address.port <= 65535
            )
        except ValueError:
            return False
    return False


def _is_address_part(text: str) -> bool:
    """Avoid reading a sentence of abbreviations as a list of bare domains."""
    if not is_contact_address(text):
        return False
    value = _bare_token(text)
    if "@" in value or "://" in value or value.lower().startswith("www."):
        return True
    # BK.Black Kn.ts is OCR-damaged prose, not a trustworthy address list.
    # A complete standalone address is still protected regardless of its case.
    host = re.split(r"[/:?#]", value, maxsplit=1)[0]
    suffix = host.rsplit(".", 1)[-1]
    return suffix.islower() or suffix.isupper()


def is_ui_label(text: str) -> bool:
    return text.strip(" [](){}|·•:;.").upper() in _UI_LABELS


def _is_identifier(text: str) -> bool:
    value = _bare_token(text).translate(_CODE_DASHES)
    return bool(_COMPACT_CODE.fullmatch(value)) and any(
        char.isdigit() for char in value
    )


def _is_letter_noise(value: str) -> bool:
    if not _LATIN_LINE.fullmatch(value):
        return False
    letters = [char.lower() for char in value if char.isalpha()]
    if len(letters) < 3:
        return False
    distinct = set(letters)
    if distinct <= _VOWELS:
        return True
    # Y can carry the vowel in myths/rhythms/crypt. A lexical dictionary is
    # deliberately not loaded; this is a conservative structural check only.
    if not distinct & _VOWELS:
        return "y" not in distinct or len(distinct) == 1
    return False


def table_identifier_indices(lines) -> set[int]:
    """Preserve OCR-damaged IDs only in a corroborated numeric table column.

    Keep every input line as layout context. This selection does not infer
    column meaning or reject a short word merely because it occurs in a table.
    """
    reasons = [exclusion_reason(line.text) for line in lines]
    anchors = [
        line
        for line, reason in zip(lines, reasons)
        if reason in {"numeric", "identifier", "no_letters", "short"}
        and any(char.isdigit() for char in line.text)
    ]
    rejected = set()
    for index in sorted(range(len(lines)), key=lambda i: (lines[i].top, lines[i].left)):
        line = lines[index]
        if reasons[index] is not None:
            continue
        tokens = [
            _bare_token(token).translate(_CODE_DASHES) for token in line.text.split()
        ]
        if not tokens or not all(
            _TABLE_CODE.fullmatch(token) or _is_identifier(token) for token in tokens
        ):
            continue
        height = max(1, line.bottom - line.top)
        aligned, row_columns = 0, []
        for anchor in anchors:
            other_height = max(1, anchor.bottom - anchor.top)
            if (
                abs(anchor.left - line.left) <= max(4, height * 0.6)
                and abs(anchor.top - line.top) <= max(height, other_height) * 8
            ):
                aligned += 1
            if (
                abs(anchor.top - line.top) <= max(height, other_height) * 1.5
                and (anchor.right <= line.left or anchor.left >= line.right)
                and all(
                    abs(anchor.left - left) > max(4, height * 0.6)
                    for left in row_columns
                )
            ):
                row_columns.append(anchor.left)
        if aligned >= 2 and len(row_columns) >= 2:
            rejected.add(index)
            # Confirmed cells can bridge a run of OCR-damaged identifiers.
            anchors.append(line)
    return rejected


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
    """Filter entire OCR lines/regions; keep identifiers inside ordinary prose."""
    return exclusion_reason(text) is None


def exclusion_reason(text: str) -> str | None:
    """Return the first rejection reason; None means eligible for translation."""
    value = " ".join(unicodedata.normalize("NFKC", text or "").split())
    if not value:
        return "empty"
    if not _LETTERS_OR_DIGITS.search(value):
        return "unsupported_characters"
    if len([char for char in value if char.isalnum()]) <= 2:
        return "short"
    if is_ui_label(value):
        return "ui_label"
    if _bare_token(value).casefold() in _TECHNICAL_LABELS:
        return "technical_label"
    # Filename/status labels are often returned as one OCR line by Paddle.
    label = _bare_token(value).strip(" [](){}|·•:;")
    if _FILE_LABEL.fullmatch(label):
        return "file_label"
    if is_contact_address(value):
        return "contact"
    if (
        _NUMBERS_AND_SEPARATORS.fullmatch(label)
        or _NUMERIC_DATE.fullmatch(label)
        or _NUMBER_LABEL.fullmatch(label)
    ):
        return "numeric"
    if _DIMENSIONS.fullmatch(label):
        return "dimensions"
    if _SIZE_LABEL.fullmatch(label.rstrip(" .;:")):
        return "file_size"
    if _MONTH_DATE.fullmatch(label) and not _FULL_MONTH.search(label):
        return "abbreviated_date"
    # OCR may return several adjacent serial/model cells as one line. A row
    # made entirely of numeric identifiers remains data regardless of spacing.
    tokens = value.split()
    if all(_is_identifier(token) for token in tokens):
        return "identifier"
    if all(_is_address_part(token) or _is_identifier(token) for token in tokens):
        return "non_text_block"
    if _is_letter_noise(value):
        return "letter_noise"
    if not any(char.isalpha() for char in value):
        return "no_letters"
    return None


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
