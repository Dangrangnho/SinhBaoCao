"""Shared text helpers used across pipeline stages."""

from __future__ import annotations

from typing import Any


NOT_ENOUGH_DATA = "Không đủ dữ liệu để xác định."


def as_string_list(value: Any) -> list[str]:
    """Convert a model list field to clean non-empty strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def unique_strings(values: Any) -> list[str]:
    """Return stripped strings in first-seen order, deduplicated case-insensitively."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
