"""Tests for safe logging configuration."""

import logging

import httpx

from utils.logging_config import configure_logging


def test_http_request_urls_with_api_key_are_not_logged(tmp_path) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers = []
    log_path = tmp_path / "app.log"
    try:
        configure_logging(log_path)
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        with httpx.Client(transport=transport) as client:
            client.post("https://example.invalid/v1/generate", params={"key": "SECRET-123"})
        for handler in root.handlers:
            handler.flush()
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.level = saved_handlers, saved_level

    assert "SECRET-123" not in log_path.read_text(encoding="utf-8")
