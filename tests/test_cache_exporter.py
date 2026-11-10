"""Tests for cache behavior and Markdown export helpers."""

from core.analyzer import ChunkAnalyzer
from core.exporter import (
    build_report_filename,
    docx_bytes,
    export_markdown,
    markdown_bytes,
    pdf_bytes,
)
from llm.llm_client import MockLLMClient
from utils.cache import ContentCache, sha256_content


def test_content_cache_uses_sha256_and_round_trips_json(tmp_path) -> None:
    cache = ContentCache(tmp_path)
    digest = sha256_content("hello")

    assert cache.get(digest) is None
    assert cache.set(digest, {"ok": True}) is True
    assert cache.get(digest) == {"ok": True}
    assert (tmp_path / f"{digest}.json").is_file()


def test_analyzer_uses_cached_chunk_without_second_llm_call(tmp_path) -> None:
    response = {
        "purpose": "Entry point",
        "technologies": ["Python"],
        "evidence": [{"claim": "Uses Python", "source_file": "main.py"}],
    }
    client = MockLLMClient([response])
    cache = ContentCache(tmp_path)
    analyzer = ChunkAnalyzer(client, cache=cache)
    chunk = {"filename": "main.py", "chunk_index": 0, "content": "print(1)"}

    first = analyzer.analyze_chunk(chunk)
    second = analyzer.analyze_chunk(chunk)

    assert first == second
    assert len(client.calls) == 1


def test_exporter_uses_required_filename_and_utf8(tmp_path) -> None:
    report = "# Báo cáo\n\n## CHƯƠNG 1\n\n- **Mục tiêu:** `SQLite`\n\nNội dung tiếng Việt."

    assert build_report_filename("Đồ án Face Recognition") == "technical_report_do_an_face_recognition.md"
    assert build_report_filename("Đồ án Face Recognition", "docx") == "technical_report_do_an_face_recognition.docx"
    assert build_report_filename("Đồ án Face Recognition", "pdf") == "technical_report_do_an_face_recognition.pdf"
    assert markdown_bytes(report).decode("utf-8") == report
    assert docx_bytes(report).startswith(b"PK")
    assert pdf_bytes(report).startswith(b"%PDF")
    path = export_markdown(report, "Demo Project", tmp_path)
    assert path.name == "technical_report_demo_project.md"
    assert path.read_text(encoding="utf-8") == report
