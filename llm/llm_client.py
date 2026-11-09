"""OpenAI-compatible LLM client and a deterministic test double."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config import Settings, get_settings
from utils.json_utils import extract_json


logger = logging.getLogger(__name__)
Message = dict[str, str]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


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
        if client is not None:
            self._client = client
        else:
            if not self.settings.openai_api_key:
                raise LLMConfigurationError(
                    "Không tìm thấy OPENAI_API_KEY trong environment hoặc file .env."
                )
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
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

        selected_model = model or self.settings.openai_model
        selected_temperature = (
            temperature
            if temperature is not None
            else self.settings.analysis_temperature
        )
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=selected_model,
                    messages=list(messages),
                    temperature=selected_temperature,
                )
                content = response.choices[0].message.content
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


def _dump_json(value: Any) -> str:
    """Serialize mock structured responses as JSON without affecting API calls."""
    import json

    return json.dumps(value, ensure_ascii=False)
