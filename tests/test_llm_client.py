"""Offline tests for the mock LLM client."""

from llm.llm_client import MockLLMClient


def test_mock_llm_client_returns_queued_json_without_api_key() -> None:
    client = MockLLMClient(['```json\n{"ok": true}\n```'])

    result = client.generate_json([{"role": "user", "content": "test"}])

    assert result == {"ok": True}
    assert len(client.calls) == 1
