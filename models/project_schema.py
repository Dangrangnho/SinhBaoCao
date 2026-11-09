"""Canonical Pydantic schemas for the analysis pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    """Base schema with strict handling of unexpected fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Evidence(SchemaBase):
    """A traceable technical claim extracted from project material."""

    claim: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_location: str | None = None
    confidence: Literal["low", "medium", "high"] = "medium"


class ChunkAnalysis(SchemaBase):
    """Validated facts extracted from one independent source chunk."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    filename: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    purpose: str = "Không đủ dữ liệu để xác định."
    technologies: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    data_flow: list[str] = Field(default_factory=list)
    important_configs: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Module(SchemaBase):
    """A logical project module and its source-file references."""

    name: str = Field(min_length=1)
    description: str = ""
    source_files: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Architecture(SchemaBase):
    """High-level architecture derived from available evidence."""

    description: str = ""
    components: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ProjectKnowledge(SchemaBase):
    """Structured, reviewable knowledge about an analyzed project."""

    project_title: str = ""
    subject_name: str = ""
    technologies: list[str] = Field(default_factory=list)
    architecture: Architecture = Field(default_factory=Architecture)
    modules: list[Module] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    data_flow: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)


class ValidationIssue(SchemaBase):
    """One issue found while checking a generated report."""

    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"] = "medium"
    source_file: str | None = None


class ValidationResult(SchemaBase):
    """Validation outcome for a generated report."""

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
