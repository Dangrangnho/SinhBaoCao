"""Tests for pipeline-level resource limits."""

import pytest

from core.pipeline import PipelineError, analyze_project
from llm.llm_client import MockLLMClient


def test_max_chunks_stops_pipeline_before_any_llm_call(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("MAX_CHUNKS", "2")
    client = MockLLMClient([])

    with pytest.raises(PipelineError, match="MAX_CHUNKS"):
        analyze_project(
            [{"filename": "main.py", "content": b"x = 1\n" * 10}],
            llm_client=client,
        )

    assert client.calls == []
