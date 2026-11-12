"""Reduce validated chunk analyses into reviewable Project Knowledge."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from config import get_settings
from core.analyzer import AnalysisStats, build_analysis_stats
from models.project_schema import (
    Architecture,
    ChunkAnalysis,
    Evidence,
    Module,
    ProjectKnowledge,
)
from utils.text_utils import NOT_ENOUGH_DATA, as_string_list, unique_strings


_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "build_project_knowledge.txt"
)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBuildResult:
    """Built knowledge, normalized inputs and aggregate analysis statistics."""

    knowledge: ProjectKnowledge
    normalized_analyses: list[ChunkAnalysis]
    stats: AnalysisStats


class KnowledgeBuildError(RuntimeError):
    """Raised when the reduce step cannot produce valid Project Knowledge."""


class KnowledgeBuilder:
    """Build Project Knowledge from analyses rather than raw source chunks."""

    def __init__(self, llm_client: Any, prompt_path: Path | None = None) -> None:
        self.llm_client = llm_client
        self.prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self.settings = get_settings()

    def build(
        self,
        analyses: Sequence[ChunkAnalysis | Mapping[str, Any]],
        *,
        project_title: str = "",
        subject_name: str = "",
        additional_notes: str = "",
    ) -> KnowledgeBuildResult:
        """Normalize analyses, reduce them with the LLM and enforce evidence bounds."""
        normalized = normalize_analyses(analyses)
        stats = build_analysis_stats(normalized)

        if not normalized:
            knowledge = ProjectKnowledge(
                project_title=project_title.strip(),
                subject_name=subject_name.strip(),
                limitations=[NOT_ENOUGH_DATA],
            )
            logger.warning("knowledge generated with insufficient analysis data")
            return KnowledgeBuildResult(knowledge, normalized, stats)

        analyses_json = json.dumps(
            [analysis.model_dump(mode="json") for analysis in normalized],
            ensure_ascii=False,
            indent=2,
        )
        prompt = _render_prompt(
            self.prompt_template,
            project_title=project_title,
            subject_name=subject_name,
            additional_notes=additional_notes,
            analyses_json=analyses_json,
        )

        try:
            raw_payload = self.llm_client.generate_json(
                [{"role": "user", "content": prompt}],
                temperature=self.settings.analysis_temperature,
            )
            candidate = ProjectKnowledge.model_validate(
                _prepare_knowledge_payload(
                    raw_payload,
                    project_title=project_title,
                    subject_name=subject_name,
                    source_files={analysis.filename for analysis in normalized},
                )
            )
        except ValidationError as exc:
            raise KnowledgeBuildError(
                f"Project Knowledge không đúng schema: {exc}"
            ) from exc
        except Exception as exc:
            raise KnowledgeBuildError(f"Không thể xây dựng Project Knowledge: {exc}") from exc

        knowledge = _apply_evidence_boundary(
            candidate,
            normalized,
            project_title=project_title,
            subject_name=subject_name,
        )
        logger.info(
            "knowledge generated: files=%s chunks=%s technologies=%s modules=%s features=%s",
            stats.files_analyzed,
            stats.chunks_analyzed,
            stats.technologies_detected,
            stats.modules_detected,
            stats.features_detected,
        )
        return KnowledgeBuildResult(knowledge, normalized, stats)


def normalize_analyses(
    analyses: Sequence[ChunkAnalysis | Mapping[str, Any]],
) -> list[ChunkAnalysis]:
    """Validate analyses and deduplicate fields before the reduce request."""
    normalized: list[ChunkAnalysis] = []
    seen_chunks: set[tuple[str, int]] = set()

    for item in analyses:
        analysis = item if isinstance(item, ChunkAnalysis) else ChunkAnalysis.model_validate(item)
        chunk_key = (analysis.filename.casefold(), analysis.chunk_index)
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        normalized.append(
            analysis.model_copy(
                update={
                    "technologies": unique_strings(analysis.technologies),
                    "modules": unique_strings(analysis.modules),
                    "classes": unique_strings(analysis.classes),
                    "functions": unique_strings(analysis.functions),
                    "inputs": unique_strings(analysis.inputs),
                    "outputs": unique_strings(analysis.outputs),
                    "features": unique_strings(analysis.features),
                    "data_flow": unique_strings(analysis.data_flow),
                    "limitations": unique_strings(analysis.limitations),
                    "evidence": _unique_evidence(analysis.evidence),
                }
            )
        )
    return normalized


def _apply_evidence_boundary(
    candidate: ProjectKnowledge,
    analyses: Sequence[ChunkAnalysis],
    *,
    project_title: str,
    subject_name: str,
) -> ProjectKnowledge:
    """Keep generated collections bounded by normalized analysis facts."""
    known_technologies = {
        value.casefold()
        for value in unique_strings(
            value for analysis in analyses for value in analysis.technologies
        )
    }
    known_modules = {
        value.casefold()
        for value in unique_strings(value for analysis in analyses for value in analysis.modules)
    }
    known_features = {
        value.casefold()
        for value in unique_strings(value for analysis in analyses for value in analysis.features)
    }

    technologies = _select_known(candidate.technologies, known_technologies)
    if not technologies:
        technologies = unique_strings(
            value for analysis in analyses for value in analysis.technologies
        )

    features = _select_known(candidate.features, known_features)
    if not features:
        features = unique_strings(value for analysis in analyses for value in analysis.features)

    modules = _merge_modules(analyses, known_modules, candidate.modules)
    evidence = _unique_evidence(
        evidence for analysis in analyses for evidence in analysis.evidence
    )
    source_files = sorted({analysis.filename for analysis in analyses})
    data_flow = unique_strings(value for analysis in analyses for value in analysis.data_flow)
    inputs = unique_strings(value for analysis in analyses for value in analysis.inputs)
    outputs = unique_strings(value for analysis in analyses for value in analysis.outputs)
    limitations = unique_strings(value for analysis in analyses for value in analysis.limitations)
    configuration = _merge_configurations(analyses)

    architecture = candidate.architecture
    architecture = Architecture(
        description=architecture.description or NOT_ENOUGH_DATA,
        components=_select_known(architecture.components, known_modules | known_technologies),
        evidence=evidence,
    )

    return ProjectKnowledge(
        project_title=(project_title.strip() or candidate.project_title),
        subject_name=(subject_name.strip() or candidate.subject_name),
        technologies=technologies,
        architecture=architecture,
        modules=modules,
        features=features,
        data_flow=data_flow,
        inputs=inputs,
        outputs=outputs,
        configuration=configuration,
        limitations=limitations or [NOT_ENOUGH_DATA],
        evidence=evidence,
        source_files=source_files,
    )


def _merge_modules(
    analyses: Sequence[ChunkAnalysis],
    known_modules: set[str],
    candidate_modules: Sequence[Module] = (),
) -> list[Module]:
    """Create modules only from names explicitly found in chunk analyses."""
    candidate_desc = {
        mod.name.strip().casefold(): mod.description.strip()
        for mod in candidate_modules
        if mod.description.strip()
    }
    merged: dict[str, Module] = {}
    for analysis in analyses:
        details_parts: list[str] = []
        if analysis.purpose and analysis.purpose != NOT_ENOUGH_DATA:
            details_parts.append(analysis.purpose)
        if analysis.classes:
            details_parts.append(f"Classes: {', '.join(analysis.classes)}")
        if analysis.functions:
            details_parts.append(f"Functions: {', '.join(analysis.functions)}")
        chunk_desc = " | ".join(details_parts)

        for module_name in analysis.modules:
            key = module_name.casefold()
            if key not in known_modules:
                continue
            existing = merged.get(key)
            related_evidence = list(analysis.evidence)
            base_desc = candidate_desc.get(key, "")
            combined_desc = (
                f"{base_desc} ({chunk_desc})"
                if base_desc and chunk_desc and chunk_desc not in base_desc
                else (base_desc or chunk_desc)
            )
            if existing is None:
                merged[key] = Module(
                    name=module_name,
                    description=combined_desc,
                    source_files=[analysis.filename],
                    evidence=related_evidence,
                )
            else:
                updated_desc = existing.description
                if chunk_desc and chunk_desc not in updated_desc:
                    updated_desc = f"{updated_desc}; {chunk_desc}".strip("; ")
                merged[key] = existing.model_copy(
                    update={
                        "description": updated_desc,
                        "source_files": unique_strings(
                            [*existing.source_files, analysis.filename]
                        ),
                        "evidence": _unique_evidence(
                            [*existing.evidence, *related_evidence]
                        ),
                    }
                )
    return list(merged.values())


def _merge_configurations(analyses: Sequence[ChunkAnalysis]) -> dict[str, Any]:
    """Merge configs without silently overwriting conflicting values.

    The first value keeps the plain key. A different value for the same key from
    another file is kept under ``"<key> [<filename>]"`` so no source is lost.
    """
    result: dict[str, Any] = {}
    file_details: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        for key, value in analysis.important_configs.items():
            if key not in result:
                result[key] = value
            elif result[key] != value:
                scoped_key = f"{key} [{analysis.filename}]"
                result.setdefault(scoped_key, value)
        entry = file_details.setdefault(
            analysis.filename,
            {
                "purpose": analysis.purpose,
                "classes": [],
                "functions": [],
                "inputs": [],
                "outputs": [],
            },
        )
        if (
            analysis.purpose
            and analysis.purpose != NOT_ENOUGH_DATA
            and entry["purpose"] == NOT_ENOUGH_DATA
        ):
            entry["purpose"] = analysis.purpose
        entry["classes"] = unique_strings([*entry["classes"], *analysis.classes])
        entry["functions"] = unique_strings([*entry["functions"], *analysis.functions])
        entry["inputs"] = unique_strings([*entry["inputs"], *analysis.inputs])
        entry["outputs"] = unique_strings([*entry["outputs"], *analysis.outputs])
    if file_details:
        result["file_details"] = file_details
    return result


def _prepare_knowledge_payload(
    raw_payload: Any,
    *,
    project_title: str,
    subject_name: str,
    source_files: set[str],
) -> dict[str, Any]:
    """Normalize common reduce-response omissions before schema validation."""
    if isinstance(raw_payload, BaseModel):
        payload = raw_payload.model_dump()
    elif isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    else:
        raise TypeError("Project Knowledge phải là một JSON object.")

    payload["project_title"] = project_title.strip() or str(payload.get("project_title", ""))
    payload["subject_name"] = subject_name.strip() or str(payload.get("subject_name", ""))
    payload["source_files"] = sorted(source_files)
    for field_name in (
        "technologies",
        "features",
        "data_flow",
        "inputs",
        "outputs",
        "limitations",
    ):
        payload[field_name] = as_string_list(payload.get(field_name, []))
    if not isinstance(payload.get("configuration"), Mapping):
        payload["configuration"] = {}
    payload["architecture"] = _prepare_architecture(payload.get("architecture", {}))
    payload["modules"] = _prepare_modules(payload.get("modules", []), source_files)
    payload["evidence"] = _prepare_evidence(payload.get("evidence", []), source_files)
    return payload


def _prepare_architecture(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"description": NOT_ENOUGH_DATA, "components": [], "evidence": []}
    return {
        "description": str(value.get("description") or NOT_ENOUGH_DATA),
        "components": as_string_list(value.get("components", [])),
        "evidence": _prepare_evidence(value.get("evidence", []), set()),
    }


def _prepare_modules(value: Any, source_files: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or not str(item.get("name", "")).strip():
            continue
        result.append(
            {
                "name": str(item["name"]).strip(),
                "description": str(item.get("description") or ""),
                "source_files": [
                    value
                    for value in as_string_list(item.get("source_files", []))
                    if value in source_files
                ],
                "evidence": _prepare_evidence(item.get("evidence", []), source_files),
            }
        )
    return result


def _prepare_evidence(value: Any, source_files: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = [value] if value else []
    result: list[dict[str, Any]] = []
    default_source = sorted(source_files)[0] if source_files else "unknown"
    for item in value:
        if isinstance(item, Mapping):
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            source_file = str(item.get("source_file") or default_source)
            if source_files and source_file not in source_files:
                source_file = default_source
            confidence = str(item.get("confidence") or "medium").lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "medium"
            result.append(
                {
                    "claim": claim,
                    "source_file": source_file,
                    "source_location": item.get("source_location") or None,
                    "confidence": confidence,
                }
            )
    return result


def _render_prompt(
    template: str,
    *,
    project_title: str,
    subject_name: str,
    additional_notes: str,
    analyses_json: str,
) -> str:
    return (
        template.replace("{{PROJECT_TITLE}}", project_title)
        .replace("{{SUBJECT_NAME}}", subject_name)
        .replace("{{ADDITIONAL_NOTES}}", additional_notes)
        .replace("{{ANALYSES_JSON}}", analyses_json)
    )


def _select_known(values: Sequence[str], known: set[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().casefold()
        if key in known and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _unique_evidence(values: Any) -> list[Evidence]:
    result: list[Evidence] = []
    seen: set[tuple[str, str, str | None]] = set()
    for value in values:
        evidence = value if isinstance(value, Evidence) else Evidence.model_validate(value)
        key = (
            evidence.claim.casefold(),
            evidence.source_file.casefold(),
            evidence.source_location.casefold() if evidence.source_location else None,
        )
        if key not in seen:
            seen.add(key)
            result.append(evidence)
    return result
