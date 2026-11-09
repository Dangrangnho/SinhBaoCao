"""Pipeline smoke test for a ZIP upload."""

from __future__ import annotations

import io
import zipfile

from core.pipeline import analyze_project
from llm.llm_client import MockLLMClient
from utils.cache import ContentCache


def _zip_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("src/main.py", "print('hello')\n")
        archive.writestr("src/database.py", "import sqlite3\n")
    return output.getvalue()


def test_zip_upload_reaches_knowledge_builder_without_real_api(tmp_path) -> None:
    client = MockLLMClient(
        [
            {
                "purpose": "Application source",
                "technologies": ["Python"],
                "modules": ["Application Module"],
                "features": ["Print greeting"],
                "evidence": [
                    {
                        "claim": "The application prints a greeting.",
                        "source_file": "project.zip::src/main.py",
                        "source_location": "line 1",
                        "confidence": "high",
                    }
                ],
            },
            {
                "purpose": "Database source",
                "technologies": ["Python", "SQLite"],
                "modules": ["Database Module"],
                "features": [],
                "evidence": [
                    {
                        "claim": "The module imports sqlite3.",
                        "source_file": "project.zip::src/database.py",
                        "source_location": "line 1",
                        "confidence": "high",
                    }
                ],
            },
            {
                "project_title": "ZIP Demo",
                "subject_name": "Testing",
                "technologies": ["Python", "SQLite"],
                "modules": [
                    {"name": "Application Module", "source_files": ["project.zip::src/main.py"]},
                    {"name": "Database Module", "source_files": ["project.zip::src/database.py"]},
                ],
                "features": ["Print greeting"],
                "architecture": {
                    "description": "A small Python project.",
                    "components": ["Application Module", "Database Module"],
                },
            },
        ]
    )

    result = analyze_project(
        [{"filename": "project.zip", "content": _zip_bytes()}],
        project_title="ZIP Demo",
        subject_name="Testing",
        llm_client=client,
        cache=ContentCache(tmp_path / "cache"),
    )

    loaded = [item for item in result.file_results if item["status"] == "loaded"]
    assert {item["filename"] for item in loaded} == {
        "project.zip::src/main.py",
        "project.zip::src/database.py",
    }
    assert result.analysis_run.analyses
    assert result.knowledge_result.knowledge.project_title == "ZIP Demo"

