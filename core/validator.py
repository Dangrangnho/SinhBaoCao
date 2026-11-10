"""Deterministic and LLM-assisted report validation with bounded revision."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from core.report_generator import REPORT_HEADINGS
from llm.llm_client import LLMClient, LLMClientError
from models.project_schema import Evidence, ProjectKnowledge, ValidationIssue, ValidationResult


NOT_ENOUGH_DATA = "Không đủ dữ liệu để xác định."
_VALIDATE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "validate_report.txt"
_REVISE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "revise_report.txt"
_TECH_LABEL_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?(?:công nghệ|technology|technologies|framework|thư viện|library)"
    r"(?:\s+sử dụng|\s+chính)?(?:\*\*)?\s*:\s*(.+)$",
    re.IGNORECASE,
)
_FEATURE_LABEL_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?(?:tính năng|chức năng|feature|features)"
    r"(?:\s+chính)?(?:\*\*)?\s*:\s*(.+)$",
    re.IGNORECASE,
)
_MODULE_RE = re.compile(
    r"\b(?:[Mm]odule|[Mm]ô-đun)\s+([A-Z][A-Za-z0-9_+#.\-]*(?:\s+[A-Z][A-Za-z0-9_+#.\-]*){0,2}\s+Module)\b"
)
_REVERSE_MODULE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_+#.\-]*(?:\s+[A-Z][A-Za-z0-9_+#.\-]*){0,2}\s+Module)\b"
)
_METRIC_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:%|(?:ms|s|giây|phút|giờ)\b)",
    re.IGNORECASE,
)
_METRIC_CONTEXT_RE = re.compile(
    r"(?:độ chính xác|accuracy|precision|recall|f1|thời gian phản hồi|tốc độ|"
    r"benchmark|hiệu năng đạt)",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "là",
    "gồm",
    "bao gồm",
    "của",
    "hệ thống",
    "được",
    "dùng",
    "sử dụng",
    "và",
    "các",
    "python",
    "the",
    "a",
    "an",
    "of",
    "with",
}
logger = logging.getLogger(__name__)


@dataclass
class ValidationRun:
    """Final report, validation result and number of revision attempts."""

    report: str
    validation_result: ValidationResult
    revision_count: int


class ValidatorError(RuntimeError):
    """Raised when LLM validation or revision cannot be completed."""


class ReportValidator:
    """Validate reports against Project Knowledge and revise within a hard limit."""

    def __init__(
        self,
        llm_client: Any,
        settings: Settings | None = None,
        validate_prompt_path: Path | None = None,
        revise_prompt_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.settings = settings or get_settings()
        self.validate_prompt = (validate_prompt_path or _VALIDATE_PROMPT_PATH).read_text(
            encoding="utf-8"
        )
        self.revise_prompt = (revise_prompt_path or _REVISE_PROMPT_PATH).read_text(
            encoding="utf-8"
        )

    def validate_code(
        self,
        report: str,
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
    ) -> ValidationResult:
        """Run deterministic checks that do not require another model call."""
        knowledge = _coerce_knowledge(project_knowledge)
        issues: list[ValidationIssue] = []
        issues.extend(_check_structure(report))
        issues.extend(_check_technologies(report, knowledge))
        issues.extend(_check_features(report, knowledge))
        issues.extend(_check_modules(report, knowledge))
        issues.extend(_check_unsupported_metrics(report))
        result = _result_from_issues(issues)
        logger.info("validation completed: valid=%s issues=%s", result.valid, len(result.issues))
        return result

    def validate_llm(
        self,
        report: str,
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
    ) -> ValidationResult:
        """Ask the LLM to find claims that exceed the structured evidence."""
        knowledge = _coerce_knowledge(project_knowledge)
        prompt = self.validate_prompt.replace(
            "{{PROJECT_KNOWLEDGE_JSON}}",
            json.dumps(knowledge.model_dump(mode="json"), ensure_ascii=False, indent=2),
        ).replace("{{REPORT_MARKDOWN}}", report)

        try:
            raw = self.llm_client.generate_json([{"role": "user", "content": prompt}], temperature=0.1)
            return _parse_validation_payload(raw)
        except LLMClientError as exc:
            raise ValidatorError(str(exc)) from exc
        except Exception as exc:
            raise ValidatorError(f"Không thể chạy LLM validator: {exc}") from exc

    def validate(
        self,
        report: str,
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
        *,
        include_llm: bool = True,
    ) -> ValidationResult:
        """Combine deterministic and LLM issues into one Pydantic result."""
        code_result = self.validate_code(report, project_knowledge)
        issues = list(code_result.issues)
        if include_llm:
            issues.extend(self.validate_llm(report, project_knowledge).issues)
        result = _result_from_issues(issues)
        logger.info("validation completed: valid=%s issues=%s", result.valid, len(result.issues))
        return result

    def revise_report(
        self,
        report: str,
        issues: Sequence[ValidationIssue],
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
    ) -> str:
        """Request one bounded revision from the LLM."""
        knowledge = _coerce_knowledge(project_knowledge)
        prompt = (
            self.revise_prompt
            .replace(
                "{{PROJECT_KNOWLEDGE_JSON}}",
                json.dumps(knowledge.model_dump(mode="json"), ensure_ascii=False, indent=2),
            )
            .replace("{{REPORT_MARKDOWN}}", report)
            .replace(
                "{{ISSUES_JSON}}",
                json.dumps(
                    [issue.model_dump(mode="json") for issue in issues],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        )
        try:
            revised = self.llm_client.generate_text([{"role": "user", "content": prompt}], temperature=0.2)
        except LLMClientError as exc:
            raise ValidatorError(str(exc)) from exc
        except Exception as exc:
            raise ValidatorError(f"Không thể revise report: {exc}") from exc
        if not isinstance(revised, str) or not revised.strip():
            raise ValidatorError("LLM trả về revised report rỗng.")
        return revised.strip()

    def validate_and_revise(
        self,
        report: str,
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
        *,
        max_revision: int | None = None,
    ) -> ValidationRun:
        """Validate and revise at most MAX_REVISION times; never loop indefinitely."""
        knowledge = _coerce_knowledge(project_knowledge)
        limit = max(0, max_revision if max_revision is not None else self.settings.max_revision)
        current_report = report
        result = self.validate(current_report, knowledge)
        revision_count = 0

        while _has_high_issue(result) and revision_count < limit:
            revision_count += 1
            revised = self.revise_report(current_report, result.issues, knowledge)
            if not _revision_is_allowed(current_report, revised, knowledge):
                guard_issue = ValidationIssue(
                    claim="Revision report",
                    reason=(
                        "Bản revise bị từ chối vì thay đổi cấu trúc 5 chương hoặc thêm "
                        "claim công nghệ/module/feature/số liệu không có trong Knowledge."
                    ),
                    severity="high",
                )
                result = _result_from_issues([*result.issues, guard_issue])
                break
            current_report = revised
            result = self.validate(current_report, knowledge)

        return ValidationRun(current_report, result, revision_count)


def validate_and_revise_report(
    report: str,
    project_knowledge: ProjectKnowledge | Mapping[str, Any],
    *,
    max_revision: int | None = None,
    llm_client: Any | None = None,
) -> ValidationRun:
    """Convenience entry point for the Streamlit UI."""
    try:
        client = llm_client or LLMClient()
        validator = ReportValidator(client)
        return validator.validate_and_revise(
            report,
            project_knowledge,
            max_revision=max_revision,
        )
    except LLMClientError as exc:
        raise ValidatorError(str(exc)) from exc


def _coerce_knowledge(value: ProjectKnowledge | Mapping[str, Any]) -> ProjectKnowledge:
    return value if isinstance(value, ProjectKnowledge) else ProjectKnowledge.model_validate(value)


def _check_structure(report: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for heading in REPORT_HEADINGS:
        if heading not in report:
            issues.append(
                ValidationIssue(
                    claim=heading,
                    reason="Báo cáo thiếu cấu trúc chương bắt buộc.",
                    severity="high",
                )
            )
    return issues


def _check_technologies(report: str, knowledge: ProjectKnowledge) -> list[ValidationIssue]:
    known = [technology for technology in knowledge.technologies if technology.strip()]
    known_keys = {technology.casefold() for technology in known}
    issues: list[ValidationIssue] = []
    for line in report.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for candidate in _extract_technology_candidates(line):
            if not _matches_known(candidate, known_keys):
                issues.append(
                    ValidationIssue(
                        claim=candidate,
                        reason="Công nghệ được nhắc đến không tồn tại trong Project Knowledge.",
                        severity="high",
                    )
                )
    return _deduplicate_issues(issues)


def _check_features(report: str, knowledge: ProjectKnowledge) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    all_ev = _all_evidence(knowledge)
    if knowledge.features and not all_ev:
        for feature in knowledge.features:
            issues.append(
                ValidationIssue(
                    claim=feature,
                    reason="Feature không có evidence tương ứng trong Project Knowledge.",
                    severity="high",
                )
            )

    known_features = {feature.casefold() for feature in knowledge.features}
    for line in report.splitlines():
        if line.lstrip().startswith("#"):
            continue
        candidates = _extract_label_candidates(line, _FEATURE_LABEL_RE)
        for candidate in candidates:
            if (
                candidate.casefold() not in known_features
                and candidate.casefold() != NOT_ENOUGH_DATA.casefold()
            ):
                issues.append(
                    ValidationIssue(
                        claim=candidate,
                        reason="Feature trong report không có trong Project Knowledge.",
                        severity="high",
                    )
                )
    return _deduplicate_issues(issues)


def _check_modules(report: str, knowledge: ProjectKnowledge) -> list[ValidationIssue]:
    known = {module.name.casefold() for module in knowledge.modules}
    issues: list[ValidationIssue] = []
    for line in report.splitlines():
        if line.lstrip().startswith("#"):
            continue
        candidates = [match.group(1) for match in _MODULE_RE.finditer(line)]
        candidates.extend(match.group(1) for match in _REVERSE_MODULE_RE.finditer(line))
        for candidate in candidates:
            candidate = _clean_candidate(candidate)
            if (
                candidate
                and not _matches_known(candidate, known)
                and candidate.casefold() not in _STOP_WORDS
            ):
                issues.append(
                    ValidationIssue(
                        claim=candidate,
                        reason="Module được nhắc đến không tồn tại trong Project Knowledge.",
                        severity="high",
                    )
                )
    return _deduplicate_issues(issues)


def _check_unsupported_metrics(report: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    in_code_block = False
    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or stripped.startswith("#") or "[source:" in line.casefold():
            continue
        if _METRIC_CONTEXT_RE.search(line) and _METRIC_RE.search(line):
            issues.append(
                ValidationIssue(
                    claim=stripped,
                    reason="Số liệu hoặc kết quả định lượng không có source/evidence.",
                    severity="high",
                )
            )
    return _deduplicate_issues(issues)


def _extract_technology_candidates(line: str) -> list[str]:
    match = _TECH_LABEL_RE.match(line)
    if not match:
        return []
    suffix = match.group(1).split("[source:", maxsplit=1)[0]
    return [
        candidate
        for candidate in _split_candidates(suffix)
        if len(candidate.split()) <= 4 and candidate.casefold() not in _STOP_WORDS
    ]


def _extract_label_candidates(line: str, pattern: re.Pattern[str]) -> list[str]:
    match = pattern.match(line)
    if not match:
        return []
    suffix = match.group(1).split("[source:", maxsplit=1)[0]
    return [
        candidate
        for candidate in _split_candidates(suffix)
        if len(candidate.split()) <= 5 and candidate.casefold() not in _STOP_WORDS
    ]


def _split_candidates(value: str) -> list[str]:
    value = re.sub(r"[`*_]", "", value)
    parts = re.split(r",|;|/|\bvà\b|\band\b|\bhoặc\b|\bor\b", value, flags=re.I)
    return [candidate for candidate in (_clean_candidate(part) for part in parts) if candidate]


def _clean_candidate(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .:;()[]{}\t\n")
    return value


def _matches_known(candidate: str, known_keys: set[str]) -> bool:
    key = candidate.casefold()
    if key in known_keys:
        return True
    return any(key.startswith(known + " ") or known.startswith(key + " ") for known in known_keys)


def _all_evidence(knowledge: ProjectKnowledge) -> list[Evidence]:
    evidence = list(knowledge.evidence) + list(knowledge.architecture.evidence)
    for module in knowledge.modules:
        evidence.extend(module.evidence)
    return evidence


def _revision_is_allowed(
    original: str,
    revised: str,
    knowledge: ProjectKnowledge,
) -> bool:
    """Reject revisions that change chapter structure or add new deterministic issues."""
    if _chapter_signature(original) != _chapter_signature(revised):
        return False
    if _check_structure(revised):
        return False

    original_issues = {
        (issue.claim.casefold(), issue.reason)
        for issue in _code_issues_for_guard(original, knowledge)
    }
    revised_issues = _code_issues_for_guard(revised, knowledge)
    for issue in revised_issues:
        key = (issue.claim.casefold(), issue.reason)
        if key not in original_issues and issue.severity == "high":
            return False
    return True


def _chapter_signature(report: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in report.splitlines()
        if line.strip().startswith("## CHƯƠNG")
    )


def _has_high_issue(result: ValidationResult) -> bool:
    return any(issue.severity == "high" for issue in result.issues)


def _result_from_issues(issues: Sequence[ValidationIssue]) -> ValidationResult:
    unique = _deduplicate_issues(list(issues))
    return ValidationResult(valid=not unique, issues=unique)


def _deduplicate_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    result: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.claim.casefold(), issue.reason.casefold())
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _parse_validation_payload(raw: Any) -> ValidationResult:
    if isinstance(raw, ValidationResult):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("Validation response phải là JSON object.")
    issues: list[dict[str, Any]] = []
    for item in raw.get("issues", []) or []:
        if not isinstance(item, Mapping):
            continue
        severity = str(item.get("severity") or "medium").lower()
        if severity == "critical":
            severity = "high"
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        issues.append(
            {
                "claim": str(item.get("claim") or "Unknown claim"),
                "reason": str(item.get("reason") or "Không có lý do."),
                "severity": severity,
                "source_file": item.get("source_file"),
            }
        )
    return _result_from_issues(
        [ValidationIssue.model_validate(issue) for issue in issues]
    )


def _code_issues_for_guard(report: str, knowledge: ProjectKnowledge) -> list[ValidationIssue]:
    """Static helper used by revision guard without constructing another client."""
    issues: list[ValidationIssue] = []
    issues.extend(_check_technologies(report, knowledge))
    issues.extend(_check_features(report, knowledge))
    issues.extend(_check_modules(report, knowledge))
    issues.extend(_check_unsupported_metrics(report))
    return _deduplicate_issues(issues)

