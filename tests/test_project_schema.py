"""Schema validation tests for the analysis pipeline."""

import pytest
from pydantic import ValidationError

from models.project_schema import ChunkAnalysis, Evidence, ProjectKnowledge


def test_chunk_analysis_and_project_knowledge_schema() -> None:
    evidence = Evidence(
        claim="Ứng dụng dùng Python",
        source_file="main.py",
        source_location="line 1",
        confidence="high",
    )
    analysis = ChunkAnalysis(
        filename="main.py",
        chunk_index=0,
        technologies=["Python"],
        modules=["Main Module"],
        features=["CLI execution"],
        evidence=[evidence],
    )
    knowledge = ProjectKnowledge(
        project_title="Demo",
        technologies=["Python"],
        evidence=[evidence],
        source_files=[analysis.filename],
    )

    assert analysis.chunk_index == 0
    assert knowledge.evidence[0].source_file == "main.py"


def test_schema_rejects_invalid_evidence_confidence() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            claim="Claim",
            source_file="main.py",
            confidence="certain",
        )
