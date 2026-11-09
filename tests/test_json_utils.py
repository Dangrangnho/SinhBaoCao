"""Tests for extracting JSON from typical LLM responses."""

import pytest

from utils.json_utils import JSONExtractionError, extract_json, extract_json_object


def test_extract_json_from_json_code_block() -> None:
    response = "Kết quả phân tích:\n```json\n{\"feature\": \"login\"}\n```"

    assert extract_json(response) == {"feature": "login"}


def test_extract_json_with_text_before_and_after() -> None:
    response = 'Here is the result: {"items": [1, 2], "nested": {"ok": true}}. Done.'

    assert extract_json_object(response)["nested"]["ok"] is True


def test_extract_json_supports_arrays() -> None:
    assert extract_json("```json\n[1, 2, 3]\n```") == [1, 2, 3]


def test_extract_json_raises_for_invalid_response() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json("Không có JSON trong response này.")
