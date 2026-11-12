"""Offline tests for the LLM client and mock LLM client."""

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from config import Settings, get_settings
from llm.llm_client import LLMClient, LLMClientError, MockLLMClient
from utils.json_utils import JSONExtractionError


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


class _FakeOpenAI:
    """Minimal stand-in for ``OpenAI`` returning queued results or raising errors."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **_: object) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = type("Message", (), {"content": outcome})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


def _status_error(cls: type, status: int) -> Exception:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=request), body=None)


def _client(outcomes: list) -> tuple[LLMClient, _FakeOpenAI]:
    fake = _FakeOpenAI(outcomes)
    client = LLMClient(settings=Settings(max_retries=2), client=fake, sleep_fn=lambda _: None)
    return client, fake


def test_authentication_error_is_not_retried() -> None:
    client, fake = _client([_status_error(AuthenticationError, 401), "unused", "unused"])

    with pytest.raises(LLMClientError):
        client.generate_text([{"role": "user", "content": "hi"}])

    assert fake.calls == 1


def test_rate_limit_error_is_retried() -> None:
    client, fake = _client([_status_error(RateLimitError, 429), "ok"])

    assert client.generate_text([{"role": "user", "content": "hi"}]) == "ok"
    assert fake.calls == 2


def test_invalid_json_response_is_requested_again() -> None:
    client, fake = _client(["not json", '{"ok": true}'])

    assert client.generate_json([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert fake.calls == 2


def test_invalid_json_is_raised_after_retries_are_exhausted() -> None:
    client, fake = _client(["no", "still no", "never"])

    with pytest.raises(JSONExtractionError):
        client.generate_json([{"role": "user", "content": "hi"}])

    assert fake.calls == 3
