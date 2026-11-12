"""Safe application logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_path: Path | str | None = None) -> None:
    """Configure one rotating file handler without logging secrets or content."""
    root_logger = logging.getLogger()
    if any(getattr(handler, "_ai_report_handler", False) for handler in root_logger.handlers):
        return

    target = Path(log_path) if log_path else Path(__file__).resolve().parents[1] / "data" / "outputs" / "app.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            target,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
        )
        handler._ai_report_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        )
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        _silence_http_request_logs()
    except OSError:
        # Logging must never prevent the application from starting.
        return


def _silence_http_request_logs() -> None:
    """Keep HTTP client request lines (full URLs) out of the log file.

    httpx logs every request URL at INFO. The Gemini REST fallback sends the API
    key as a ``?key=`` query parameter, so those lines would write the key to disk.
    """
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
