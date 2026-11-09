"""Tests for the replaceable MVP chunking strategy."""

import pytest

from core.chunker import CharacterChunker, chunk_documents, chunk_text


def test_chunk_text_preserves_filename_indexes_and_overlap() -> None:
    chunks = chunk_text(
        "abcdefghij",
        chunk_size=6,
        chunk_overlap=2,
        filename="main.py",
    )

    assert [chunk["content"] for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(chunk["filename"] == "main.py" for chunk in chunks)
    assert chunks[0]["start_char"] == 0
    assert chunks[1]["start_char"] == 4


def test_chunk_text_handles_empty_text_and_invalid_overlap() -> None:
    assert chunk_text("", chunk_size=5, chunk_overlap=1) == []

    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=5, chunk_overlap=5)


def test_chunk_documents_skips_unloaded_records() -> None:
    documents = [
        {"filename": "ok.txt", "content": "abcdef", "status": "loaded"},
        {"filename": "bad.bin", "content": "ignored", "status": "skipped"},
    ]

    chunks = chunk_documents(
        documents,
        strategy=CharacterChunker(chunk_size=4, chunk_overlap=1),
    )

    assert [chunk["content"] for chunk in chunks] == ["abcd", "def"]
