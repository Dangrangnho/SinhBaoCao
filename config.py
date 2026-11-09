"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the report generator."""

    llm_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    chunk_size: int = 6000
    chunk_overlap: int = 500
    analysis_temperature: float = 0.1
    report_temperature: float = 0.3
    request_timeout: float = 60.0
    max_retries: int = 2
    max_file_size: int = 10 * 1024 * 1024
    max_archive_size: int = 50 * 1024 * 1024
    max_archive_extracted_size: int = 100 * 1024 * 1024
    max_archive_members: int = 1000
    max_revision: int = 1
    max_total_upload_size: int = 100 * 1024 * 1024
    max_chunks: int = 500


def get_settings() -> Settings:
    """Build settings from environment variables without exposing secrets."""
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", Settings.llm_base_url).strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", Settings.openai_model).strip(),
        chunk_size=_get_int("CHUNK_SIZE", Settings.chunk_size),
        chunk_overlap=_get_int("CHUNK_OVERLAP", Settings.chunk_overlap),
        analysis_temperature=_get_float(
            "ANALYSIS_TEMPERATURE", Settings.analysis_temperature
        ),
        report_temperature=_get_float("REPORT_TEMPERATURE", Settings.report_temperature),
        request_timeout=_get_float("REQUEST_TIMEOUT", Settings.request_timeout),
        max_retries=_get_int("MAX_RETRIES", Settings.max_retries),
        max_file_size=_get_int("MAX_FILE_SIZE", Settings.max_file_size),
        max_archive_size=_get_int("MAX_ARCHIVE_SIZE", Settings.max_archive_size),
        max_archive_extracted_size=_get_int(
            "MAX_ARCHIVE_EXTRACTED_SIZE", Settings.max_archive_extracted_size
        ),
        max_archive_members=_get_int(
            "MAX_ARCHIVE_MEMBERS", Settings.max_archive_members
        ),
        max_revision=_get_int("MAX_REVISION", Settings.max_revision),
        max_total_upload_size=_get_int(
            "MAX_TOTAL_UPLOAD_SIZE", Settings.max_total_upload_size
        ),
        max_chunks=_get_int("MAX_CHUNKS", Settings.max_chunks),
    )


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable and fall back safely."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable and fall back safely."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default
