"""OpenAI-compatible LLM client and a deterministic test double."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import httpx
from openai import OpenAI
from pydantic import BaseModel

from config import Settings, get_settings
from utils.json_utils import extract_json


logger = logging.getLogger(__name__)
Message = dict[str, str]
SchemaT = TypeVar("SchemaT", bound=BaseModel)

_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_VERTEX_EXPRESS_URL = (
    "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
)
_FALLBACK_GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


class LLMClientError(RuntimeError):
    """Raised when an LLM request cannot be completed."""


class LLMConfigurationError(LLMClientError):
    """Raised when the client is not configured with an API key."""


class TextGenerator(Protocol):
    """Minimal interface consumed by future analyzer/report components."""

    def generate_text(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Generate text from chat messages."""


class LLMClient:
    """Small wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._sleep = sleep_fn
        self._api_key = self.settings.gemini_api_key or self.settings.openai_api_key
        self._custom_client = client is not None
        self._preferred_rest_template: str | None = (
            _GEMINI_REST_URL if self._api_key.startswith("AQ.") else None
        )
        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise LLMConfigurationError(
                    "Không tìm thấy GEMINI_API_KEY (hoặc OPENAI_API_KEY) trong environment hoặc file .env."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.settings.llm_base_url,
                timeout=self.settings.request_timeout,
                max_retries=0,
            )

    def generate_text(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Call the configured model with bounded exponential-backoff retries."""
        if not messages:
            raise ValueError("messages không được rỗng.")

        selected_model = (
            model
            or (
                self.settings.gemini_model
                if self.settings.gemini_model != Settings.gemini_model
                else self.settings.openai_model
            )
            or self.settings.gemini_model
        )
        selected_temperature = (
            temperature
            if temperature is not None
            else self.settings.analysis_temperature
        )
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                content = self._request_completion(
                    messages=messages,
                    model=selected_model,
                    temperature=selected_temperature,
                )
                if not content:
                    raise LLMClientError("LLM trả về response rỗng.")
                return content
            except Exception as exc:  # SDK exceptions vary across compatible APIs.
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                delay = min(2**attempt, 8)
                logger.warning(
                    "LLM request failed on attempt %s/%s; retrying in %ss.",
                    attempt + 1,
                    self.settings.max_retries + 1,
                    delay,
                )
                self._sleep(delay)

        raise LLMClientError(f"LLM request thất bại: {last_error}") from last_error

    def _request_completion(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        temperature: float,
    ) -> str:
        """Dispatch to OpenAI-compatible SDK or Gemini/Vertex REST for AQ.* keys."""
        if self._custom_client or not self._preferred_rest_template:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as openai_exc:
                if self._custom_client or not self._api_key:
                    raise
                logger.info(
                    "OpenAI-compatible call failed (%s); trying Gemini/Vertex REST fallback.",
                    type(openai_exc).__name__,
                )
                return self._call_gemini_rest_with_fallback(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    prior_error=openai_exc,
                )

        return self._call_gemini_rest_with_fallback(
            messages=messages,
            model=model,
            temperature=temperature,
        )

    def _call_gemini_rest_with_fallback(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        prior_error: Exception | None = None,
    ) -> str:
        """Try Gemini REST and Vertex AI Express Mode endpoints with model fallback."""
        templates = (
            [self._preferred_rest_template]
            if self._preferred_rest_template
            else []
        )
        for candidate in (_GEMINI_REST_URL, _VERTEX_EXPRESS_URL):
            if candidate not in templates:
                templates.append(candidate)

        candidate_models = [model]
        for fallback_model in _FALLBACK_GEMINI_MODELS:
            if fallback_model not in candidate_models:
                candidate_models.append(fallback_model)

        last_exc: Exception | None = prior_error
        for current_model in candidate_models:
            for template in templates:
                try:
                    text = _call_google_generate_content(
                        url_template=template,
                        api_key=self._api_key,
                        model=current_model,
                        messages=messages,
                        temperature=temperature,
                        timeout=self.settings.request_timeout,
                    )
                    self._preferred_rest_template = template
                    return text
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code
                    # 404/503 ongenerativelanguage means model deprecated/overloaded; skip Vertex 403
                    if template == _GEMINI_REST_URL and status_code in {404, 503}:
                        break
                except Exception as exc:
                    last_exc = exc

        assert last_exc is not None
        raise last_exc

    def generate_json(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        model: str | None = None,
        schema: type[SchemaT] | None = None,
    ) -> Any:
        """Generate a response, extract JSON, and optionally validate a schema."""
        response = self.generate_text(
            messages,
            temperature=temperature,
            model=model,
        )
        payload = extract_json(response)
        return schema.model_validate(payload) if schema is not None else payload


class MockLLMClient:
    """Queue-based LLM double for tests and offline development."""

    def __init__(self, responses: Sequence[str | dict[str, Any] | list[Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def generate_text(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Return the next queued response without making a network request."""
        del temperature, model
        self.calls.append(list(messages))
        if not self._responses:
            raise LLMClientError("MockLLMClient đã hết response mẫu.")
        response = self._responses.pop(0)
        return response if isinstance(response, str) else _dump_json(response)

    def generate_json(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        model: str | None = None,
        schema: type[SchemaT] | None = None,
    ) -> Any:
        """Return the next queued JSON response, optionally schema-validated."""
        payload = extract_json(
            self.generate_text(messages, temperature=temperature, model=model)
        )
        return schema.model_validate(payload) if schema is not None else payload


def _call_google_generate_content(
    *,
    url_template: str,
    api_key: str,
    model: str,
    messages: Sequence[Message],
    temperature: float,
    timeout: float,
) -> str:
    """Call Google's native generateContent endpoint (Vertex Express or AI Studio)."""
    clean_model = model.removeprefix("models/")
    url = url_template.format(model=clean_model)
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "system":
            system_parts.append({"text": text})
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": 8192,
    }
    if clean_model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {"thinkingLevel": "MINIMAL"}

    payload: dict[str, Any] = {
        "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    response = httpx.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    if response.status_code == 400 and "thinkingConfig" in generation_config:
        generation_config.pop("thinkingConfig", None)
        response = httpx.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=timeout,
        )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMClientError(f"Gemini response không có candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [
        str(part.get("text", ""))
        for part in parts
        if "text" in part and not part.get("thought", False)
    ]
    if not text_parts:
        text_parts = [str(part.get("text", "")) for part in parts if "text" in part]
    return "".join(text_parts)


def _dump_json(value: Any) -> str:
    """Serialize mock structured responses as JSON without affecting API calls."""
    import json

    return json.dumps(value, ensure_ascii=False)
