"""End-to-end test for the complete offline MVP flow."""

from __future__ import annotations

from pathlib import Path

from core.exporter import export_markdown
from core.pipeline import analyze_project
from core.report_generator import REPORT_HEADINGS, ReportGenerator
from core.validator import ReportValidator
from llm.llm_client import MockLLMClient
from models.project_schema import ProjectKnowledge
from utils.cache import ContentCache


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "simple_python_project"


def _report() -> str:
    return "\n\n".join(
        [
            "# BÁO CÁO ĐỒ ÁN: Simple Python Project",
            "**Môn học:** Software Engineering",
            *REPORT_HEADINGS,
            "### 1.1. Đặt vấn đề\nHệ thống mẫu minh họa lưu trữ người dùng.",
            "### 1.2. Mục tiêu đề tài\nPhân tích project Python nhỏ.",
            "### 1.3. Phạm vi nghiên cứu\nKhông đủ dữ liệu để xác định.",
        ]
    )


def test_complete_pipeline_without_real_api(tmp_path) -> None:
    uploaded_files = [
        {"filename": path.name, "content": path.read_bytes()}
        for path in sorted(EXAMPLE_DIR.glob("*"))
        if path.is_file()
    ]
    client = MockLLMClient(
        [
            {
                "purpose": "SQLite persistence module",
                "technologies": ["Python", "SQLite"],
                "modules": ["Database Module"],
                "features": ["Save users"],
                "evidence": [
                    {
                        "claim": "Save users is implemented with SQLite",
                        "source_file": "database.py",
                        "source_location": "line 20",
                        "confidence": "high",
                    }
                ],
            },
            {
                "purpose": "Application entry point",
                "technologies": ["Python"],
                "modules": ["Application Module"],
                "features": ["Initialize database"],
                "evidence": [
                    {
                        "claim": "Initialize database is called by the application",
                        "source_file": "main.py",
                        "source_location": "line 9",
                        "confidence": "high",
                    }
                ],
            },
            {
                "purpose": "Project documentation",
                "technologies": ["Python", "SQLite"],
                "evidence": [
                    {
                        "claim": "The README documents Python and SQLite",
                        "source_file": "README.md",
                        "source_location": "Technology section",
                        "confidence": "high",
                    }
                ],
            },
            {
                "project_title": "Simple Python Project",
                "subject_name": "Software Engineering",
                "technologies": ["Python", "SQLite"],
                "modules": [
                    {"name": "Application Module", "source_files": ["main.py"]},
                    {"name": "Database Module", "source_files": ["database.py"]},
                ],
                "features": ["Initialize database", "Save users"],
                "architecture": {
                    "description": "A small Python application with SQLite persistence.",
                    "components": ["Application Module", "Database Module"],
                },
            },
            _report(),
            {"valid": True, "issues": []},
        ]
    )

    pipeline_result = analyze_project(
        uploaded_files,
        project_title="Simple Python Project",
        subject_name="Software Engineering",
        llm_client=client,
        cache=ContentCache(tmp_path / "cache"),
    )
    knowledge = pipeline_result.knowledge_result.knowledge
    assert len(pipeline_result.file_results) == 3
    assert all(result["status"] == "loaded" for result in pipeline_result.file_results)
    assert pipeline_result.chunks
    assert pipeline_result.analysis_run.analyses
    assert isinstance(knowledge, ProjectKnowledge)
    assert knowledge.technologies == ["Python", "SQLite"]

    generated = ReportGenerator(client).generate(knowledge)
    validation = ReportValidator(client).validate_and_revise(
        generated.markdown,
        knowledge,
        max_revision=0,
    )
    exported = export_markdown(generated.markdown, knowledge.project_title, tmp_path / "output")

    assert all(heading in generated.markdown for heading in REPORT_HEADINGS)
    assert validation.validation_result.valid is True
    assert validation.revision_count == 0
    assert exported.name == "technical_report_simple_python_project.md"
    assert exported.read_text(encoding="utf-8") == generated.markdown
    assert len(client.calls) == 6
