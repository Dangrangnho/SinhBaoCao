"""Tests for normalization, reduction and evidence boundaries."""

from llm.llm_client import MockLLMClient
from models.project_schema import ChunkAnalysis, Evidence
from core.knowledge_builder import KnowledgeBuilder, normalize_analyses


def _analysis() -> list[ChunkAnalysis]:
    evidence = Evidence(
        claim="main.py imports Python standard library",
        source_file="main.py",
        source_location="line 1",
        confidence="high",
    )
    return [
        ChunkAnalysis(
            filename="main.py",
            chunk_index=0,
            purpose="Entry point",
            technologies=["Python", "python"],
            modules=["Main Module"],
            features=["CLI"],
            evidence=[evidence, evidence],
        ),
        ChunkAnalysis(
            filename="main.py",
            chunk_index=1,
            technologies=["Python"],
            modules=["Main Module"],
            features=["CLI"],
            evidence=[evidence],
        ),
    ]


def test_normalize_analyses_deduplicates_values_and_evidence() -> None:
    normalized = normalize_analyses(_analysis())

    assert len(normalized) == 2
    assert normalized[0].technologies == ["Python"]
    assert normalized[0].evidence == normalized[0].evidence[:1]


def test_knowledge_builder_filters_unknown_reduce_values() -> None:
    response = {
        "project_title": "Demo",
        "technologies": ["Python", "Rust"],
        "modules": [
            {
                "name": "Main Module",
                "description": "Entry point",
                "source_files": ["main.py"],
            },
            {"name": "Unknown Module", "description": "Invented"},
        ],
        "features": ["CLI", "Payments"],
        "architecture": {
            "description": "A simple application",
            "components": ["Main Module", "Unknown Module"],
        },
    }
    builder = KnowledgeBuilder(MockLLMClient([response]))

    result = builder.build(
        _analysis(),
        project_title="Demo",
        subject_name="Software Engineering",
    )

    assert result.knowledge.technologies == ["Python"]
    assert result.knowledge.features == ["CLI"]
    assert [module.name for module in result.knowledge.modules] == ["Main Module"]
    assert result.knowledge.architecture.components == ["Main Module"]
    assert result.stats.files_analyzed == 1
    assert result.stats.chunks_analyzed == 2
    assert result.stats.technologies_detected == 1
    assert result.stats.modules_detected == 1
    assert result.stats.features_detected == 1


def test_conflicting_configuration_values_are_kept_per_file() -> None:
    from core.knowledge_builder import _merge_configurations
    from models.project_schema import ChunkAnalysis

    merged = _merge_configurations(
        [
            ChunkAnalysis(filename="a.py", chunk_index=0, important_configs={"PORT": 8000}),
            ChunkAnalysis(filename="b.py", chunk_index=0, important_configs={"PORT": 8080}),
            ChunkAnalysis(filename="c.py", chunk_index=0, important_configs={"PORT": 8000}),
        ]
    )

    merged.pop("file_details", None)
    assert merged == {"PORT": 8000, "PORT [b.py]": 8080}
