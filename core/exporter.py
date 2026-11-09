"""Markdown export helpers for Streamlit download and local persistence."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs"


def build_report_filename(project_name: str) -> str:
    """Return a safe filename in the required technical_report_<name>.md format."""
    ascii_name = project_name.strip().replace("Đ", "D").replace("đ", "d")
    ascii_name = (
        unicodedata.normalize("NFKD", ascii_name)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    slug = slug or "project"
    return f"technical_report_{slug}.md"


def markdown_bytes(report: str) -> bytes:
    """Encode Markdown as UTF-8 for Streamlit's download button."""
    return report.encode("utf-8")


def export_markdown(
    report: str,
    project_name: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist a Markdown report locally and return its path."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / build_report_filename(project_name)
    target.write_text(report, encoding="utf-8")
    logger.info("report exported: %s", target.name)
    return target
