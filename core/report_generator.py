"""Generate a traceable Vietnamese Markdown report from Project Knowledge."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from llm.llm_client import LLMClient, LLMClientError
from models.project_schema import ProjectKnowledge
from utils.text_utils import NOT_ENOUGH_DATA


REPORT_HEADINGS = (
    "## CHƯƠNG 1: TỔNG QUAN VÀ MỤC TIÊU",
    "## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG",
    "## CHƯƠNG 3: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG",
    "## CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ",
    "## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN",
)
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "generate_report.txt"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedReport:
    """Generated Markdown and the knowledge snapshot used to create it."""

    markdown: str
    project_knowledge: ProjectKnowledge


class ReportGenerationError(RuntimeError):
    """Raised when report generation or structural validation fails."""


class ReportGenerator:
    """Generate Markdown from canonical Project Knowledge only."""

    def __init__(
        self,
        llm_client: Any,
        settings: Settings | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.settings = settings or get_settings()
        self.prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")

    def generate(
        self,
        project_knowledge: ProjectKnowledge | Mapping[str, Any],
        *,
        additional_notes: str = "",
    ) -> GeneratedReport:
        """Generate and structurally validate a five-chapter Markdown report."""
        knowledge = (
            project_knowledge
            if isinstance(project_knowledge, ProjectKnowledge)
            else ProjectKnowledge.model_validate(project_knowledge)
        )
        prompt = self._render_prompt(knowledge, additional_notes=additional_notes)

        try:
            markdown = self.llm_client.generate_text(
                [{"role": "user", "content": prompt}],
                temperature=self.settings.report_temperature,
            )
        except LLMClientError as exc:
            raise ReportGenerationError(str(exc)) from exc
        except Exception as exc:
            raise ReportGenerationError(f"Không thể sinh báo cáo: {exc}") from exc

        if not isinstance(markdown, str) or not markdown.strip():
            raise ReportGenerationError("LLM trả về báo cáo rỗng.")

        normalized = normalize_report_markdown(markdown, knowledge)
        logger.info("report generated: project=%s", knowledge.project_title)
        return GeneratedReport(markdown=normalized, project_knowledge=knowledge)

    def _render_prompt(
        self,
        knowledge: ProjectKnowledge,
        *,
        additional_notes: str,
    ) -> str:
        """Render only structured Project Knowledge into the report prompt."""
        knowledge_json = json.dumps(
            knowledge.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        return (
            self.prompt_template
            .replace("{{PROJECT_TITLE}}", knowledge.project_title)
            .replace("{{SUBJECT_NAME}}", knowledge.subject_name)
            .replace("{{ADDITIONAL_NOTES}}", additional_notes)
            .replace("{{PROJECT_KNOWLEDGE_JSON}}", knowledge_json)
        )


def generate_report(
    project_knowledge: ProjectKnowledge | Mapping[str, Any],
    *,
    additional_notes: str = "",
    llm_client: Any | None = None,
) -> GeneratedReport:
    """Convenience entry point that creates the configured real LLM client."""
    try:
        client = llm_client or LLMClient()
        return ReportGenerator(client).generate(
            project_knowledge,
            additional_notes=additional_notes,
        )
    except LLMClientError as exc:
        raise ReportGenerationError(str(exc)) from exc


def normalize_report_markdown(markdown: str, knowledge: ProjectKnowledge) -> str:
    """Normalize title and reject output that does not contain all five chapters."""
    text = markdown.strip()
    if not text.startswith("# BÁO CÁO ĐỒ ÁN:"):
        text = (
            f"# BÁO CÁO ĐỒ ÁN: {knowledge.project_title}\n\n"
            f"**Môn học:** {knowledge.subject_name}\n\n{text}"
        )

    missing = [heading for heading in REPORT_HEADINGS if heading not in text]
    if missing:
        missing_text = "\n\n".join(
            f"{heading}\n\n{NOT_ENOUGH_DATA}" for heading in missing
        )
        text = f"{text}\n\n{missing_text}"

    return text.strip() + "\n"
