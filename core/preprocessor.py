"""Safe normalization and filtering for loaded project documents."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


_EXCESSIVE_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_newlines(text: str) -> str:
    """Convert Windows and old-Mac line endings to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(text: str) -> str:
    """Remove non-semantic whitespace while preserving code indentation.

    Internal spaces and tabs are intentionally preserved because collapsing them
    can change source code, string literals, YAML indentation, or formatted data.
    """
    normalized = text.replace("\u00a0", " ").replace("\ufeff", "")
    normalized = normalize_newlines(normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _EXCESSIVE_BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def preprocess_content(text: str) -> str:
    """Normalize Unicode, control characters, newlines and extra whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = _remove_unsafe_control_characters(normalized)
    return normalize_whitespace(normalized)


def preprocess_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of one loader result."""
    result = dict(record)
    content = preprocess_content(str(result.get("content") or ""))
    result["content"] = content

    if result.get("status") == "loaded" and not content:
        result["status"] = "skipped"
        result["error"] = "Nội dung rỗng sau tiền xử lý."
    return result


def preprocess_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a batch of records without mutating the input objects."""
    return [preprocess_record(record) for record in records]


def _remove_unsafe_control_characters(text: str) -> str:
    """Remove control characters except tab and line separators."""
    return "".join(
        character
        for character in text
        if character in {"\n", "\t"}
        or not unicodedata.category(character).startswith("C")
    )
