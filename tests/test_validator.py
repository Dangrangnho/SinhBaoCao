"""Tests for deterministic validation and bounded revision."""

from core.report_generator import REPORT_HEADINGS
from core.validator import ReportValidator, find_features_without_evidence
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


def test_code_validator_does_not_false_flag_normal_vietnamese_prose() -> None:
    prose = (
        "Dự án dùng để kiểm thử AI Technical Report Generator và lưu trữ người dùng mẫu.\n"
        "Các thành phần chính: main.py làm entry point để khởi tạo cơ sở dữ liệu.\n"
        "Các tính năng chính đã được triển khai bao gồm lưu trữ thông qua các module Python.\n"
    )
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(_report(prose), _knowledge())

    assert result.valid is True
    assert result.issues == []

def test_empty_report_has_no_feature_issue_from_knowledge_alone() -> None:
    knowledge = _knowledge().model_copy(update={"features": ["Login", "Upload file ZIP"]})
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(_report(), knowledge)

    assert result.valid
    assert find_features_without_evidence(knowledge) == ["Upload file ZIP"]


def test_vietnamese_prose_does_not_create_false_positives() -> None:
    report = _report(
        "Hệ thống sử dụng Python để xây dựng giao diện cho người dùng.\n"
        "Người dùng có thể dùng giao diện web để tải lên tài liệu.\n"
        "Người sử dụng có thể xem báo cáo.\n"
        "Module xử lý dữ liệu đọc file đầu vào.\n"
        "Hệ thống có chức năng tải file lên."
    )
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(report, _knowledge())

    assert result.valid, [issue.claim for issue in result.issues]


def test_unknown_technology_in_label_line_is_detected() -> None:
    report = _report("Công nghệ sử dụng: Python, Rust")
    validator = ReportValidator(MockLLMClient([]))

    result = validator.validate_code(report, _knowledge())

    assert [issue.claim for issue in result.issues] == ["Rust"]


def test_fake_source_tag_does_not_bypass_metric_check() -> None:
    report = _report("Độ chính xác đạt 99% [source: fake.py]")
    validator = ReportValidator(MockLLMClient([]))

    reasons = " ".join(issue.reason for issue in validator.validate_code(report, _knowledge()).issues)

    assert "Tham chiếu source" in reasons
    assert "Số liệu" in reasons


def test_metric_with_known_source_tag_is_accepted() -> None:
    knowledge = _knowledge().model_copy(update={"source_files": ["demo.zip::src/auth.py"]})
    report = _report("Thời gian phản hồi đạt 120 ms [source: auth.py]")
    validator = ReportValidator(MockLLMClient([]))

    assert validator.validate_code(report, knowledge).valid


def test_llm_validator_failure_falls_back_to_deterministic_result() -> None:
    validator = ReportValidator(MockLLMClient([]))  # no queued response -> LLM error

    result = validator.validate(_report(), _knowledge())

    assert [issue.severity for issue in result.issues] == ["medium"]
    assert "LLM validator" in result.issues[0].reason


def test_revision_failure_keeps_current_report() -> None:
    original = _report("Độ chính xác đạt 95%.")
    client = MockLLMClient([{"valid": True, "issues": []}])  # nothing left for revise
    validator = ReportValidator(client)

    run = validator.validate_and_revise(original, _knowledge(), max_revision=1)

    assert run.report == original
    assert run.revision_count == 1
    assert any("Không revise được" in issue.reason for issue in run.validation_result.issues)
