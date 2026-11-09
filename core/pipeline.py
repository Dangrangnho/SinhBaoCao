"""Application pipeline used by the Streamlit UI."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from config import get_settings
from core.analyzer import AnalysisRun, ChunkAnalyzer
from core.chunker import chunk_documents
from core.file_loader import InMemoryFile, load_files
from core.knowledge_builder import KnowledgeBuildError, KnowledgeBuildResult, KnowledgeBuilder
from core.preprocessor import preprocess_records
from llm.llm_client import LLMClient, LLMClientError
from utils.cache import ContentCache


ProgressCallback = Callable[[str, float], None]
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """All intermediate results needed by the UI review step."""

    file_results: list[dict[str, Any]]
    preprocessed_files: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    analysis_run: AnalysisRun
    knowledge_result: KnowledgeBuildResult


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot proceed to a meaningful result."""

    def __init__(
        self,
        message: str,
        *,
        file_results: list[dict[str, Any]] | None = None,
        analysis_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.file_results = file_results or []
        self.analysis_errors = analysis_errors or []


def analyze_project(
    uploaded_files: Sequence[Mapping[str, Any]],
    *,
    project_title: str = "",
    subject_name: str = "",
    technology_stack: str = "",
    additional_notes: str = "",
    llm_client: Any | None = None,
    cache: ContentCache | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run the MVP analysis pipeline while preserving partial file errors."""
    settings = get_settings()
    if not uploaded_files:
        raise PipelineError("Chưa có file nào được upload.")

    total_size = sum(len(bytes(item.get("content", b""))) for item in uploaded_files)
    if total_size > settings.max_total_upload_size:
        raise PipelineError(
            f"Tổng dung lượng file vượt quá giới hạn {settings.max_total_upload_size} bytes."
        )

    _progress(progress_callback, "Reading files...", 0.05)
    file_inputs = [
        InMemoryFile(
            name=str(item.get("filename", "uploaded_file.txt")),
            data=bytes(item.get("content", b"")),
        )
        for item in uploaded_files
    ]
    file_results = load_files(
        file_inputs,
        max_file_size=settings.max_file_size,
        max_archive_size=settings.max_archive_size,
        max_archive_extracted_size=settings.max_archive_extracted_size,
        max_archive_members=settings.max_archive_members,
    )

    _progress(progress_callback, "Preprocessing files...", 0.18)
    preprocessed_files = preprocess_records(file_results)
    loadable_files = [record for record in preprocessed_files if record.get("status") == "loaded"]
    if not loadable_files:
        raise PipelineError(
            "Không có file hợp lệ để phân tích.",
            file_results=file_results,
        )

    _progress(progress_callback, "Creating chunks...", 0.28)
    chunks = chunk_documents(loadable_files)
    if not chunks:
        raise PipelineError("Không tạo được chunk từ các file hợp lệ.", file_results=file_results)

    try:
        client = llm_client or LLMClient(settings=settings)
    except LLMClientError as exc:
        raise PipelineError(
            str(exc),
            file_results=file_results,
        ) from exc
    analyzer = ChunkAnalyzer(client, cache=cache)
    analysis_run = analyzer.analyze_chunks(
        chunks,
        progress_callback=lambda current, total: _progress(
            progress_callback,
            f"Analyzing chunk {current}/{total}...",
            0.30 + (0.45 * current / max(total, 1)),
        ),
    )
    logger.info(
        "analysis completed: files=%s chunks=%s errors=%s",
        analysis_run.stats.files_analyzed,
        analysis_run.stats.chunks_analyzed,
        len(analysis_run.errors),
    )
    if not analysis_run.analyses:
        raise PipelineError(
            "Không có chunk nào phân tích thành công. Kiểm tra API, JSON response "
            "hoặc nội dung file.",
            file_results=file_results,
            analysis_errors=analysis_run.errors,
        )

    _progress(progress_callback, "Building project knowledge...", 0.80)
    builder = KnowledgeBuilder(client)
    try:
        knowledge_result = builder.build(
            analysis_run.analyses,
            project_title=project_title,
            subject_name=subject_name,
            additional_notes=(
                f"Technology stack supplied by user: {technology_stack}\n{additional_notes}"
            ).strip(),
        )
    except KnowledgeBuildError as exc:
        raise PipelineError(
            f"Không thể xây dựng Project Knowledge: {exc}",
            file_results=file_results,
            analysis_errors=analysis_run.errors,
        ) from exc
    _progress(progress_callback, "Analysis completed.", 1.0)

    return PipelineResult(
        file_results=file_results,
        preprocessed_files=preprocessed_files,
        chunks=chunks,
        analysis_run=analysis_run,
        knowledge_result=knowledge_result,
    )


def _progress(
    callback: ProgressCallback | None,
    message: str,
    value: float,
) -> None:
    if callback is not None:
        callback(message, max(0.0, min(value, 1.0)))
