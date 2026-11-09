"""Tests for independent chunk analysis and run statistics."""

from llm.llm_client import MockLLMClient
from core.analyzer import ChunkAnalyzer


def test_analyzer_sends_one_chunk_per_request_and_validates_result(tmp_path) -> None:
    client = MockLLMClient(
        [
            {
                "purpose": "Entry point",
                "technologies": ["Python"],
                "modules": ["Main Module"],
                "features": ["CLI"],
                "evidence": [
                    {
                        "claim": "The file defines main",
                        "source": "main.py",
                        "line": "1",
                        "confidence": "high",
                    }
                ],
            }
        ]
    )
    from utils.cache import ContentCache

    analyzer = ChunkAnalyzer(client, cache=ContentCache(tmp_path))

    run = analyzer.analyze_chunks(
        [
            {
                "filename": "main.py",
                "chunk_index": 0,
                "content": "def main(): pass",
            }
        ]
    )

    assert len(run.analyses) == 1
    assert run.analyses[0].filename == "main.py"
    assert run.analyses[0].evidence[0].source_location == "1"
    assert run.stats.to_dict() == {
        "files_analyzed": 1,
        "chunks_analyzed": 1,
        "technologies_detected": 1,
        "modules_detected": 1,
        "features_detected": 1,
    }
    assert len(client.calls) == 1
    assert "def main(): pass" in client.calls[0][0]["content"]
