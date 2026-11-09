"""Tests for deterministic validation and bounded revision."""

from core.report_generator import REPORT_HEADINGS
from core.validator import ReportValidator
from llm.llm_client import MockLLMClient
from models.project_schema import Evidence, Module, ProjectKnowledge


def _knowledge() -> ProjectKnowledge:
    return ProjectKnowledge(
        project_title="Demo",
        subject_name="Software Engineering",
        technologies=["Python"],
        features=["Login"],
        modules=[Module(name="Auth Module")],
        evidence=[
            Evidence(
                claim="Login feature is implemented",
                source_file="auth.py",
                confidence="high",
            )
        ],
        source_files=["auth.py"],
    )


def _report(body: str = "") -> str:
    return "\n\n".join(
        [
            "# BÁO CÁO ĐỒ ÁN: Demo",
            "**Môn học:** Software Engineering",
            *REPORT_HEADINGS,
            body,
        ]
    )


def test_code_validator_detects_technology_feature_module_and_metric_issues() -> None:
    report = _report(
        "Công nghệ: Python, Rust\n"
        "Tính năng: Payment\n"
        "Module Payment Module\n"
        "Độ chính xác đạt 95%."
    )
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(report, _knowledge())
    reasons = " ".join(issue.reason for issue in result.issues)

    assert not result.valid
    assert "Công nghệ" in reasons
    assert "Feature" in reasons
    assert "Module" in reasons
    assert "Số liệu" in reasons


def test_code_validator_detects_missing_chapter() -> None:
    report = _report().replace(REPORT_HEADINGS[4], "")
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(report, _knowledge())

    assert not result.valid
    assert any(REPORT_HEADINGS[4] == issue.claim for issue in result.issues)


def test_llm_validator_returns_pydantic_validation_result() -> None:
    client = MockLLMClient(
        [
            {
                "valid": False,
                "issues": [
                    {
                        "claim": "accuracy 95%",
                        "reason": "Không có evidence",
                        "severity": "high",
                    }
                ],
            }
        ]
    )
    validator = ReportValidator(client)

    result = validator.validate_llm(_report(), _knowledge())

    assert result.valid is False
    assert result.issues[0].severity == "high"


def test_revision_is_bounded_and_preserves_chapter_structure() -> None:
    original = _report("Độ chính xác đạt 95%.")
    revised = _report("Kết quả định lượng: Không đủ dữ liệu để xác định.")
    client = MockLLMClient(
        [
            {
                "valid": False,
                "issues": [
                    {
                        "claim": "95%",
                        "reason": "Không có source",
                        "severity": "high",
                    }
                ],
            },
            revised,
            {"valid": True, "issues": []},
        ]
    )
    validator = ReportValidator(client)

    result = validator.validate_and_revise(original, _knowledge(), max_revision=1)

    assert result.revision_count == 1
    assert result.report == revised
    assert result.validation_result.valid is True
    assert all(heading in result.report for heading in REPORT_HEADINGS)
