"""Tests for text preprocessing."""

from core.preprocessor import preprocess_content, preprocess_record, preprocess_records


def test_preprocess_content_normalizes_without_destroying_indentation() -> None:
    text = "\ufeffclass Demo:\r\n\tvalue = 1  \r\n\r\n\r\n\u00a0"

    result = preprocess_content(text)

    assert result == "class Demo:\n\tvalue = 1"


def test_preprocess_record_marks_empty_loaded_content_as_skipped() -> None:
    result = preprocess_record(
        {
            "filename": "empty.txt",
            "extension": ".txt",
            "size": 1,
            "content": "\x00\r\n",
            "status": "loaded",
            "error": None,
        }
    )

    assert result["status"] == "skipped"
    assert result["error"] == "Nội dung rỗng sau tiền xử lý."


def test_preprocess_records_does_not_mutate_input() -> None:
    original = {
        "filename": "main.py",
        "extension": ".py",
        "size": 10,
        "content": "print(1)\r\n",
        "status": "loaded",
        "error": None,
    }

    result = preprocess_records([original])

    assert original["content"] == "print(1)\r\n"
    assert result[0]["content"] == "print(1)"
