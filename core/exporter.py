"""Markdown, Word (.docx), and PDF (.pdf) export helpers for Streamlit download."""

from __future__ import annotations

import html
import io
import logging
import re
import unicodedata
from pathlib import Path

import pymupdf
from docx import Document
from docx.shared import Pt


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs"
_INLINE_TOKEN_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
_NUMBERED_ITEM_RE = re.compile(r"^\d+\.\s+(.+)$")


def build_report_filename(project_name: str, extension: str = "md") -> str:
    """Return a safe filename in the required technical_report_<name>.<ext> format."""
    ascii_name = project_name.strip().replace("Đ", "D").replace("đ", "d")
    ascii_name = (
        unicodedata.normalize("NFKD", ascii_name)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    slug = slug or "project"
    ext = extension.lstrip(".").lower() or "md"
    return f"technical_report_{slug}.{ext}"


def markdown_bytes(report: str) -> bytes:
    """Encode Markdown as UTF-8 for Streamlit's download button."""
    return report.encode("utf-8")


def docx_bytes(report: str) -> bytes:
    """Convert a Markdown report into an academic-styled Word (.docx) byte stream."""
    document = Document()
    normal_style = document.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(13)

    lines = report.splitlines()
    in_code_block = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        col_count = max(len(row) for row in table_rows)
        table = document.add_table(rows=len(table_rows), cols=col_count)
        table.style = "Table Grid"
        for r_idx, row in enumerate(table_rows):
            for c_idx, cell_text in enumerate(row):
                cell = table.cell(r_idx, c_idx)
                cell.text = ""
                paragraph = cell.paragraphs[0]
                _append_inline_runs(paragraph, cell_text.strip(), bold_default=(r_idx == 0))
        table_rows = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            if in_code_block:
                paragraph = document.add_paragraph()
                run = paragraph.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(10)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r"[:\-]+", cell) for cell in cells if cell):
                continue
            table_rows.append(cells)
            continue

        flush_table()

        if not stripped:
            continue

        if stripped.startswith("# "):
            heading = document.add_heading(stripped[2:].strip(), level=1)
            for run in heading.runs:
                run.font.name = "Times New Roman"
        elif stripped.startswith("## "):
            heading = document.add_heading(stripped[3:].strip(), level=2)
            for run in heading.runs:
                run.font.name = "Times New Roman"
        elif stripped.startswith("### "):
            heading = document.add_heading(stripped[4:].strip(), level=3)
            for run in heading.runs:
                run.font.name = "Times New Roman"
        elif stripped.startswith("#### "):
            heading = document.add_heading(stripped[5:].strip(), level=4)
            for run in heading.runs:
                run.font.name = "Times New Roman"
        elif stripped.startswith(("- ", "* ")):
            paragraph = document.add_paragraph(style="List Bullet")
            _append_inline_runs(paragraph, stripped[2:].strip())
        elif (num_match := _NUMBERED_ITEM_RE.match(stripped)) is not None:
            paragraph = document.add_paragraph(style="List Number")
            _append_inline_runs(paragraph, num_match.group(1).strip())
        else:
            paragraph = document.add_paragraph()
            _append_inline_runs(paragraph, stripped)

    flush_table()
    if code_lines:
        paragraph = document.add_paragraph()
        run = paragraph.add_run("\n".join(code_lines))
        run.font.name = "Consolas"
        run.font.size = Pt(10)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pdf_bytes(report: str) -> bytes:
    """Convert a Markdown report into an A4 PDF byte stream with full Vietnamese support."""
    html_content = _markdown_to_html(report)
    css = (
        "body { font-family: sans-serif; font-size: 11pt; line-height: 1.45; }"
        "h1 { font-size: 18pt; margin-top: 12pt; margin-bottom: 8pt; }"
        "h2 { font-size: 14pt; margin-top: 12pt; margin-bottom: 6pt; }"
        "h3 { font-size: 12pt; margin-top: 10pt; margin-bottom: 4pt; }"
        "p { margin-top: 4pt; margin-bottom: 6pt; }"
        "li { margin-top: 2pt; margin-bottom: 2pt; }"
        "pre { font-family: monospace; font-size: 9pt; background-color: #f5f5f5; padding: 6pt; }"
        "code { font-family: monospace; font-size: 9.5pt; }"
        "table { border-collapse: collapse; width: 100%; margin-top: 6pt; margin-bottom: 6pt; }"
        "th, td { border: 1px solid #666; padding: 4pt; font-size: 10pt; }"
    )
    story = pymupdf.Story(html=html_content, user_css=css)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (42, 42, -42, -42)

    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def export_markdown(
    report: str,
    project_name: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist a Markdown report locally and return its path."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / build_report_filename(project_name, "md")
    target.write_text(report, encoding="utf-8")
    logger.info("report exported: %s", target.name)
    return target


def _append_inline_runs(paragraph: Any, text: str, *, bold_default: bool = False) -> None:
    """Render **bold** and `code` spans into a python-docx paragraph."""
    for token in _INLINE_TOKEN_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.name = "Times New Roman"
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(11)
        else:
            run = paragraph.add_run(token)
            run.bold = bold_default
            run.font.name = "Times New Roman"


def _markdown_to_html(report: str) -> str:
    """Convert report Markdown into clean HTML suitable for PyMuPDF Story."""
    parts: list[str] = ["<html><body>"]
    in_code_block = False
    in_ul = False
    in_ol = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        close_lists()
        parts.append("<table>")
        for r_idx, row in enumerate(table_rows):
            tag = "th" if r_idx == 0 else "td"
            cells_html = "".join(f"<{tag}>{_inline_html(c)}</{tag}>" for c in row)
            parts.append(f"<tr>{cells_html}</tr>")
        parts.append("</table>")
        table_rows = []

    for line in report.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_table()
            close_lists()
            if in_code_block:
                escaped_code = html.escape("\n".join(code_lines))
                parts.append(f"<pre>{escaped_code}</pre>")
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r"[:\-]+", cell) for cell in cells if cell):
                continue
            table_rows.append(cells)
            continue

        flush_table()

        if not stripped:
            close_lists()
            continue

        if stripped.startswith("# "):
            close_lists()
            parts.append(f"<h1>{_inline_html(stripped[2:].strip())}</h1>")
        elif stripped.startswith("## "):
            close_lists()
            parts.append(f"<h2>{_inline_html(stripped[3:].strip())}</h2>")
        elif stripped.startswith("### "):
            close_lists()
            parts.append(f"<h3>{_inline_html(stripped[4:].strip())}</h3>")
        elif stripped.startswith("#### "):
            close_lists()
            parts.append(f"<h4>{_inline_html(stripped[5:].strip())}</h4>")
        elif stripped.startswith(("- ", "* ")):
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{_inline_html(stripped[2:].strip())}</li>")
        elif (num_match := _NUMBERED_ITEM_RE.match(stripped)) is not None:
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append("<ol>")
                in_ol = True
            parts.append(f"<li>{_inline_html(num_match.group(1).strip())}</li>")
        else:
            close_lists()
            parts.append(f"<p>{_inline_html(stripped)}</p>")

    flush_table()
    close_lists()
    if code_lines:
        parts.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _inline_html(text: str) -> str:
    """Escape HTML and format **bold** and `code` tokens."""
    out: list[str] = []
    for token in _INLINE_TOKEN_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            out.append(f"<b>{html.escape(token[2:-2])}</b>")
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            out.append(f"<code>{html.escape(token[1:-1])}</code>")
        else:
            out.append(html.escape(token))
    return "".join(out)

