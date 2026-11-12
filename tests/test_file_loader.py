"""Tests for supported file loading behavior."""

from __future__ import annotations

import io
import zipfile

import pymupdf as fitz
from docx import Document

from core.file_loader import load_file, load_files


class FakeUploadedFile:
    """Small Streamlit UploadedFile substitute for unit tests."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF content")
    data = document.tobytes()
    document.close()
    return data


def _make_docx() -> bytes:
    document = Document()
    document.add_paragraph("DOCX content")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _make_zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_load_utf8_text_file() -> None:
    result = load_file(FakeUploadedFile("main.py", "print('Xin chào')\r\n".encode()))

    assert result["status"] == "loaded"
    assert result["extension"] == ".py"
    assert "Xin chào" in result["content"]
    assert result["error"] is None


def test_load_pdf_and_docx() -> None:
    pdf_result = load_file(FakeUploadedFile("guide.pdf", _make_pdf()))
    docx_result = load_file(FakeUploadedFile("guide.docx", _make_docx()))

    assert pdf_result["status"] == "loaded"
    assert "PDF content" in pdf_result["content"]
    assert docx_result["status"] == "loaded"
    assert "DOCX content" in docx_result["content"]


def test_skip_empty_binary_unsupported_and_too_large_files() -> None:
    results = load_files(
        [
            FakeUploadedFile("empty.txt", b""),
            FakeUploadedFile("image.png", b"\x89PNG\x00binary"),
            FakeUploadedFile("unknown.exe", b"content"),
            FakeUploadedFile("large.txt", b"12345"),
        ],
        max_file_size=4,
    )

    assert results[0]["status"] == "skipped"
    assert results[1]["status"] == "skipped"
    assert results[2]["status"] == "skipped"
    assert results[3]["status"] == "skipped"
    assert all(result["content"] == "" for result in results)


def test_invalid_pdf_returns_error_instead_of_raising() -> None:
    result = load_file(FakeUploadedFile("broken.pdf", b"not a pdf"))

    assert result["status"] == "error"
    assert result["error"]


def test_load_zip_expands_supported_files_and_keeps_archive_metadata() -> None:
    archive_data = _make_zip(
        {
            "src/main.py": b"print('hello')\n",
            "docs/README.md": b"# Demo\n",
            "assets/image.png": b"not processed",
        }
    )

    results = load_files([FakeUploadedFile("project.zip", archive_data)])

    assert results[0]["filename"] == "project.zip"
    assert results[0]["status"] == "expanded"
    loaded_names = {
        result["filename"]
        for result in results
        if result["status"] == "loaded"
    }
    assert loaded_names == {"project.zip::src/main.py", "project.zip::docs/README.md"}
    assert any(result["status"] == "skipped" for result in results)


def test_load_zip_rejects_path_traversal_and_absolute_entries() -> None:
    archive_data = _make_zip(
        {
            "../escape.py": b"print('bad')",
            "/absolute.py": b"print('bad')",
            "safe/main.py": b"print('safe')",
        }
    )

    results = load_files([FakeUploadedFile("project.zip", archive_data)])

    loaded = [result for result in results if result["status"] == "loaded"]
    skipped = [result for result in results if result["status"] == "skipped"]
    assert [result["filename"] for result in loaded] == ["project.zip::safe/main.py"]
    assert len(skipped) == 2
    assert all("không an toàn" in result["error"] for result in skipped)


def test_load_zip_enforces_extracted_size_limit() -> None:
    archive_data = _make_zip({"main.py": b"x" * 20})

    results = load_files(
        [FakeUploadedFile("project.zip", archive_data)],
        max_file_size=100,
        max_archive_extracted_size=10,
    )

    member = next(result for result in results if "main.py" in result["filename"])
    assert member["status"] == "skipped"
    assert "ZIP" in member["error"]


def test_invalid_zip_returns_error_instead_of_raising() -> None:
    results = load_files([FakeUploadedFile("broken.zip", b"not a zip")])

    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert results[0]["error"]
