"""Tests for five-chapter Markdown report generation."""

import pytest

from core.report_generator import REPORT_HEADINGS, ReportGenerator
from llm.llm_client import MockLLMClient
from models.project_schema import Evidence, ProjectKnowledge


def _knowledge() -> ProjectKnowledge:
    return ProjectKnowledge(
        project_title="Demo Project",
        subject_name="Software Engineering",
        technologies=["Python"],
        features=["CLI"],
        evidence=[
            Evidence(
                claim="The project uses Python",
                source_file="main.py",
                confidence="high",
            )
        ],
        source_files=["main.py"],
    )


def _report() -> str:
    return "\n\n".join(
        [
            "# BÁO CÁO ĐỒ ÁN: Demo Project",
            "**Môn học:** Software Engineering",
            *REPORT_HEADINGS,
            "### 1.1. Đặt vấn đề\nKhông đủ dữ liệu để xác định.",
            "### 1.2. Mục tiêu đề tài\nKhông đủ dữ liệu để xác định.",
            "### 1.3. Phạm vi nghiên cứu\nKhông đủ dữ liệu để xác định.",
        ]
    )


def test_report_generator_uses_knowledge_and_keeps_five_chapters() -> None:
    result = ReportGenerator(MockLLMClient([_report()])).generate(_knowledge())

    assert result.markdown.startswith("# BÁO CÁO ĐỒ ÁN: Demo Project")
    assert all(heading in result.markdown for heading in REPORT_HEADINGS)
    assert "main.py" in result.project_knowledge.source_files


def test_report_generator_adds_missing_structure_with_safe_placeholder() -> None:
    result = ReportGenerator(MockLLMClient(["# BÁO CÁO ĐỒ ÁN: Demo Project"])).generate(
        _knowledge()
    )

    assert all(heading in result.markdown for heading in REPORT_HEADINGS)
    assert "Không đủ dữ liệu để xác định." in result.markdown
