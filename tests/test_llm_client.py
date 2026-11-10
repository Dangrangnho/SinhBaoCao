"""Offline tests for the LLM client and mock LLM client."""

from config import Settings, get_settings
from llm.llm_client import LLMClient, MockLLMClient


def test_mock_llm_client_returns_queued_json_without_api_key() -> None:
    client = MockLLMClient(['```json\n{"ok": true}\n```'])

    result = client.generate_json([{"role": "user", "content": "test"}])

    assert result == {"ok": True}
    assert len(client.calls) == 1


def test_get_settings_reads_gemini_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    settings = get_settings()

    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in settings.llm_base_url


def test_llm_client_supports_aq_vertex_express_key(monkeypatch) -> None:
    called_urls: list[str] = []

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "candidates": [
                    {"content": {"parts": [{"text": '{"status": "ok"}'}]}}
                ]
            }

    def fake_post(url: str, *, params: dict, json: dict, timeout: float) -> DummyResponse:
        called_urls.append(url)
        assert params["key"] == "AQ.test-key"
        assert json["contents"][0]["parts"][0]["text"] == "hello"
        return DummyResponse()

    monkeypatch.setattr("llm.llm_client.httpx.post", fake_post)
    settings = Settings(gemini_api_key="AQ.test-key", gemini_model="gemini-2.5-flash")
    client = LLMClient(settings=settings)

    result = client.generate_json([{"role": "user", "content": "hello"}])

    assert result == {"status": "ok"}
    assert called_urls == [
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ]


