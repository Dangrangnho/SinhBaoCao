"""Independent chunk analysis and analysis-run statistics."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from config import get_settings
from models.project_schema import ChunkAnalysis
from utils.cache import ContentCache, sha256_content
from utils.text_utils import NOT_ENOUGH_DATA, as_string_list, unique_strings


_CACHE_SCHEMA_VERSION = "chunk-analysis-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "analyze_chunk.txt"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisStats:
    """Metrics produced from validated chunk analyses."""

    files_analyzed: int
    chunks_analyzed: int
    technologies_detected: int
    modules_detected: int
    features_detected: int

    def to_dict(self) -> dict[str, int]:
        """Return metrics in the shape used by the Streamlit UI."""
        return {
            "files_analyzed": self.files_analyzed,
            "chunks_analyzed": self.chunks_analyzed,
            "technologies_detected": self.technologies_detected,
            "modules_detected": self.modules_detected,
            "features_detected": self.features_detected,
        }


@dataclass
class AnalysisRun:
    """Validated analyses plus per-chunk errors and aggregate metrics."""

    analyses: list[ChunkAnalysis]
    errors: list[dict[str, Any]]
    stats: AnalysisStats


class ChunkAnalysisError(RuntimeError):
    """Raised when one chunk cannot be analyzed or validated."""


class ChunkAnalyzer:
    """Analyze chunks independently through an injected JSON-capable LLM client."""

    def __init__(
        self,
        llm_client: Any,
        prompt_path: Path | None = None,
        cache: ContentCache | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self.settings = get_settings()
        self.cache = cache or ContentCache()
        # Cache entries are only reusable for the same model, prompt and schema.
        self._cache_namespace = sha256_content(
            "\n".join(
                (
                    _CACHE_SCHEMA_VERSION,
                    self.settings.openai_model,
                    sha256_content(self.prompt_template),
                )
            )
        )

    def analyze_chunk(self, chunk: Mapping[str, Any]) -> ChunkAnalysis:
        """Analyze exactly one chunk and validate the result with Pydantic."""
        filename = str(chunk.get("filename", "unknown"))
        chunk_index = _as_non_negative_int(chunk.get("chunk_index", 0))
        content = str(chunk.get("content", ""))
        if not content.strip():
            raise ChunkAnalysisError(f"Đoạn nội dung rỗng: {filename}#{chunk_index}.")

        content_hash = str(chunk.get("content_sha256") or sha256_content(content))
        cache_key = sha256_content(f"{self._cache_namespace}:{content_hash}")
        cached_payload = self.cache.get(cache_key)
        if cached_payload is not None:
            try:
                cached_payload = _rebind_cached_payload(
                    cached_payload,
                    filename=filename,
                    chunk_index=chunk_index,
                )
                analysis = ChunkAnalysis.model_validate(
                    _prepare_payload(
                        cached_payload,
                        filename=filename,
                        chunk_index=chunk_index,
                    )
                )
                logger.info("analysis completed from cache: %s#%s", filename, chunk_index)
                return analysis
            except Exception as exc:
                logger.warning("invalid analysis cache ignored for %s: %s", filename, exc)

        prompt = _render_prompt(
            self.prompt_template,
            filename=filename,
            chunk_index=chunk_index,
            content=content,
        )
        try:
            raw_payload = self.llm_client.generate_json(
                [{"role": "user", "content": prompt}],
                temperature=self.settings.analysis_temperature,
            )
            payload = _prepare_payload(
                raw_payload,
                filename=filename,
                chunk_index=chunk_index,
            )
            analysis = ChunkAnalysis.model_validate(payload)
            self.cache.set(cache_key, analysis.model_dump(mode="json"))
            logger.info("analysis completed: %s#%s", filename, chunk_index)
            return analysis
        except ValidationError as exc:
            raise ChunkAnalysisError(
                f"LLM trả về kết quả phân tích sai cấu trúc cho {filename}#{chunk_index}: {exc}"
            ) from exc
        except Exception as exc:
            raise ChunkAnalysisError(
                f"Không thể phân tích {filename}#{chunk_index}: {exc}"
            ) from exc

    def analyze_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        progress_callback: Any | None = None,
    ) -> AnalysisRun:
        """Analyze chunks, using bounded concurrency when client is thread-safe."""
        total = len(chunks)
        # Keep MockLLMClient (which pops from a shared queue in order) sequential.
        use_parallel = total > 1 and not hasattr(self.llm_client, "calls")

        if not use_parallel:
            analyses: list[ChunkAnalysis] = []
            errors: list[dict[str, Any]] = []
            for position, chunk in enumerate(chunks, start=1):
                try:
                    analyses.append(self.analyze_chunk(chunk))
                except ChunkAnalysisError as exc:
                    errors.append(
                        {
                            "filename": str(chunk.get("filename", "unknown")),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "error": str(exc),
                        }
                    )
                if progress_callback is not None:
                    progress_callback(position, total)
            return AnalysisRun(
                analyses=analyses,
                errors=errors,
                stats=build_analysis_stats(analyses),
            )

        ordered_analyses: dict[int, ChunkAnalysis] = {}
        ordered_errors: dict[int, dict[str, Any]] = {}
        completed_count = 0
        max_workers = min(4, total)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.analyze_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                chunk = chunks[idx]
                try:
                    ordered_analyses[idx] = future.result()
                except ChunkAnalysisError as exc:
                    ordered_errors[idx] = {
                        "filename": str(chunk.get("filename", "unknown")),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "error": str(exc),
                    }
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, total)

        analyses = [ordered_analyses[i] for i in sorted(ordered_analyses)]
        errors = [ordered_errors[i] for i in sorted(ordered_errors)]
        return AnalysisRun(
            analyses=analyses,
            errors=errors,
            stats=build_analysis_stats(analyses),
        )


def _rebind_cached_payload(
    payload: Mapping[str, Any],
    *,
    filename: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Rebind cached source metadata when identical content appears in another file."""
    rebound = dict(payload)
    rebound["filename"] = filename
    rebound["chunk_index"] = chunk_index
    evidence = rebound.get("evidence")
    if isinstance(evidence, list):
        rebound["evidence"] = [
            {**item, "source_file": filename}
            if isinstance(item, Mapping)
            else item
            for item in evidence
        ]
    return rebound


def build_analysis_stats(analyses: Sequence[ChunkAnalysis]) -> AnalysisStats:
    """Calculate unique file, technology, module and feature counts."""
    return AnalysisStats(
        files_analyzed=len({analysis.filename for analysis in analyses}),
        chunks_analyzed=len(analyses),
        technologies_detected=len(
            unique_strings(value for analysis in analyses for value in analysis.technologies)
        ),
        modules_detected=len(
            unique_strings(value for analysis in analyses for value in analysis.modules)
        ),
        features_detected=len(
            unique_strings(value for analysis in analyses for value in analysis.features)
        ),
    )


def _render_prompt(template: str, *, filename: str, chunk_index: int, content: str) -> str:
    """Replace only known placeholders, leaving source braces untouched."""
    return (
        template.replace("{{FILENAME}}", filename)
        .replace("{{CHUNK_INDEX}}", str(chunk_index))
        .replace("{{CONTENT}}", content)
    )


def _prepare_payload(
    raw_payload: Any,
    *,
    filename: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Normalize common LLM variations before Pydantic validation."""
    if isinstance(raw_payload, BaseModel):
        payload: dict[str, Any] = raw_payload.model_dump()
    elif isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    else:
        raise TypeError("Kết quả phân tích đoạn phải là một JSON object.")

    # Filename and chunk index are trusted from the pipeline, not from the model.
    payload["filename"] = filename
    payload["chunk_index"] = chunk_index

    list_fields = (
        "technologies",
        "classes",
        "functions",
        "modules",
        "inputs",
        "outputs",
        "features",
        "data_flow",
        "limitations",
    )
    for field_name in list_fields:
        payload[field_name] = as_string_list(payload.get(field_name, []))

    if not isinstance(payload.get("important_configs"), Mapping):
        payload["important_configs"] = {}
    payload["evidence"] = _normalize_evidence(payload.get("evidence", []), filename)
    if not str(payload.get("purpose", "")).strip():
        payload["purpose"] = NOT_ENOUGH_DATA
    return payload


def _normalize_evidence(value: Any, filename: str) -> list[dict[str, Any]]:
    """Normalize evidence aliases while retaining a traceable source file."""
    if not isinstance(value, list):
        value = [value] if value else []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            claim = str(item.get("claim") or item.get("description") or "").strip()
            if not claim:
                continue
            source_file = str(item.get("source_file") or item.get("source") or filename)
            source_location = item.get("source_location") or item.get("line")
            confidence = str(item.get("confidence") or "medium").lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "medium"
            normalized.append(
                {
                    "claim": claim,
                    "source_file": source_file,
                    "source_location": str(source_location) if source_location else None,
                    "confidence": confidence,
                }
            )
        elif str(item).strip():
            normalized.append(
                {
                    "claim": str(item).strip(),
                    "source_file": filename,
                    "source_location": None,
                    "confidence": "low",
                }
            )
    return normalized


def _as_non_negative_int(value: Any) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return 0
    return max(integer, 0)
