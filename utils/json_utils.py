"""Utilities for extracting JSON from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


class JSONExtractionError(ValueError):
    """Raised when an LLM response does not contain valid JSON."""


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract the first valid JSON object or array from an LLM response.

    Handles plain JSON, JSON inside a `````json`` code block, and prose before or
    after the JSON. Strings and nested structures are handled by JSONDecoder rather
    than by a regular expression.
    """
    if not isinstance(text, str) or not text.strip():
        raise JSONExtractionError("LLM response không chứa nội dung JSON.")

    candidates = [block.group(1).strip() for block in _CODE_BLOCK_RE.finditer(text)]
    candidates.append(text.strip().lstrip("\ufeff"))

    for candidate in candidates:
        parsed = _try_parse_candidate(candidate)
        if parsed is not None:
            return parsed

    raise JSONExtractionError("Không tìm thấy JSON hợp lệ trong LLM response.")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract JSON and require the top-level value to be an object."""
    value = extract_json(text)
    if not isinstance(value, dict):
        raise JSONExtractionError("JSON response phải là object.")
    return value


def _try_parse_candidate(candidate: str) -> Any | None:
    """Try direct parsing, then scan for a JSON object or array."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None
