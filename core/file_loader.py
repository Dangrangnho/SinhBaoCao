"""Load supported project files into a common text-oriented representation."""

from __future__ import annotations

import io
import logging
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, BinaryIO

import pymupdf as fitz
from docx import Document

from config import get_settings
from utils.cache import sha256_content


logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".java",
        ".kt",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".properties",
        ".gradle",
        ".kts",
        ".sql",
        ".html",
        ".css",
        ".pdf",
        ".docx",
    }
)

TEXT_EXTENSIONS = SUPPORTED_EXTENSIONS - {".pdf", ".docx"}
ARCHIVE_EXTENSIONS = frozenset({".zip"})
_IGNORED_ARCHIVE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".github",
        ".idea",
        ".vscode",
        ".gradle",
        ".agents",
        ".agent",
        ".claude",
        ".cursor",
        ".gemini",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        "build",
        "dist",
    }
)
_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class InMemoryFile:
    """File-like adapter for uploaded bytes persisted in Streamlit state."""

    name: str
    data: bytes

    def getvalue(self) -> bytes:
        """Return the stored upload bytes."""
        return self.data


def load_file(file_input: Any, max_file_size: int | None = None) -> dict[str, Any]:
    """Load one path, bytes object, or Streamlit-style uploaded file.

    The function always returns a metadata dictionary. A malformed or unsupported
    file becomes an ``error`` or ``skipped`` result rather than raising to callers.
    """
    filename = _get_filename(file_input)
    extension = Path(filename).suffix.lower()
    raw_data: bytes = b""

    try:
        raw_data = _read_bytes(file_input)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _result(
            filename=filename,
            extension=extension,
            size=0,
            status="error",
            error=f"Không thể đọc file: {exc}",
        )

    return _load_raw_file(filename, raw_data, max_file_size=max_file_size)


def _load_raw_file(
    filename: str,
    raw_data: bytes,
    *,
    max_file_size: int | None = None,
) -> dict[str, Any]:
    """Load already-read bytes using the same rules for normal and ZIP files."""
    extension = Path(filename).suffix.lower()
    size = len(raw_data)
    size_limit = max_file_size if max_file_size is not None else get_settings().max_file_size

    if extension not in SUPPORTED_EXTENSIONS:
        return _result(
            filename=filename,
            extension=extension,
            size=size,
            status="skipped",
            error=f"Định dạng không được hỗ trợ: {extension or '[không có extension]'}.",
        )

    if size == 0:
        return _result(
            filename=filename,
            extension=extension,
            size=0,
            status="skipped",
            error="File rỗng.",
        )

    if size > size_limit:
        return _result(
            filename=filename,
            extension=extension,
            size=size,
            status="skipped",
            error=f"Tệp vượt quá giới hạn {size_limit} byte.",
        )

    try:
        if extension in TEXT_EXTENSIONS:
            if _looks_binary(raw_data):
                return _result(
                    filename=filename,
                    extension=extension,
                    size=size,
                    status="skipped",
                    error="File có dấu hiệu là binary.",
                )
            content = _decode_text(raw_data, filename)
        elif extension == ".pdf":
            content = _extract_pdf_text(raw_data)
        else:
            content = _extract_docx_text(raw_data)
    except Exception as exc:
        return _result(
            filename=filename,
            extension=extension,
            size=size,
            status="error",
            error=f"Không thể phân tích file: {exc}",
        )

    if not content.strip():
        return _result(
            filename=filename,
            extension=extension,
            size=size,
            status="skipped",
            error="Không tìm thấy nội dung văn bản.",
        )

    return _result(
        filename=filename,
        extension=extension,
        size=size,
        content=content,
        status="loaded",
    )


def load_files(
    files: list[Any] | tuple[Any, ...],
    max_file_size: int | None = None,
    max_archive_size: int | None = None,
    max_archive_extracted_size: int | None = None,
    max_archive_members: int | None = None,
) -> list[dict[str, Any]]:
    """Load a batch, expanding ZIP archives without letting one failure stop it.

    ZIP entries are returned with names such as ``project.zip::src/main.py`` so
    evidence can still point back to the uploaded archive. Unsafe paths and
    oversized entries are skipped before their contents are passed downstream.
    """
    settings = get_settings()
    results: list[dict[str, Any]] = []
    for file_input in files:
        filename = _get_filename(file_input)
        if Path(filename).suffix.lower() in ARCHIVE_EXTENSIONS:
            results.extend(
                _load_zip_file(
                    file_input,
                    max_file_size=max_file_size,
                    max_archive_size=(
                        max_archive_size
                        if max_archive_size is not None
                        else settings.max_archive_size
                    ),
                    max_archive_extracted_size=(
                        max_archive_extracted_size
                        if max_archive_extracted_size is not None
                        else settings.max_archive_extracted_size
                    ),
                    max_archive_members=(
                        max_archive_members
                        if max_archive_members is not None
                        else settings.max_archive_members
                    ),
                )
            )
        else:
            results.append(load_file(file_input, max_file_size=max_file_size))
    return results


def _load_zip_file(
    file_input: Any,
    *,
    max_file_size: int | None,
    max_archive_size: int,
    max_archive_extracted_size: int,
    max_archive_members: int,
) -> list[dict[str, Any]]:
    """Safely expand a ZIP archive into ordinary loader records."""
    archive_filename = _get_filename(file_input)
    archive_extension = Path(archive_filename).suffix.lower()
    try:
        raw_data = _read_bytes(file_input)
    except Exception as exc:  # pragma: no cover - defensive boundary
        return [
            _result(
                filename=archive_filename,
                extension=archive_extension,
                size=0,
                status="error",
                error=f"Không thể đọc tệp ZIP: {exc}",
            )
        ]

    archive_size = len(raw_data)
    if archive_size == 0:
        return [
            _result(
                filename=archive_filename,
                extension=archive_extension,
                size=0,
                status="skipped",
                error="Tệp ZIP rỗng.",
            )
        ]
    if archive_size > max_archive_size:
        return [
            _result(
                filename=archive_filename,
                extension=archive_extension,
                size=archive_size,
                status="skipped",
                error=f"Tệp ZIP vượt quá giới hạn {max_archive_size} byte.",
            )
        ]

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_data))
    except (zipfile.BadZipFile, OSError) as exc:
        return [
            _result(
                filename=archive_filename,
                extension=archive_extension,
                size=archive_size,
                status="error",
                error=f"Tệp ZIP không hợp lệ: {exc}",
            )
        ]

    with archive:
        members = archive.infolist()
        archive_record = _result(
            filename=archive_filename,
            extension=archive_extension,
            size=archive_size,
            status="expanded",
            error=None,
        )
        if len(members) > max_archive_members:
            archive_record["error"] = (
                f"Tệp ZIP chứa {len(members)} mục, vượt quá giới hạn "
                f"{max_archive_members} mục."
            )
            archive_record["status"] = "skipped"
            logger.info("file skipped: %s (%s)", archive_filename, archive_record["error"])
            return [archive_record]

        results = [archive_record]
        seen_names: set[str] = set()
        extracted_size = 0
        loaded_count = 0

        for member in members:
            if member.is_dir() or member.filename.endswith(("/", "\\")):
                continue

            safe_name = _safe_archive_member_name(member.filename)
            display_name = f"{archive_filename}::{safe_name or member.filename}"
            if safe_name is None:
                results.append(
                    _result(
                        filename=display_name,
                        extension=Path(member.filename).suffix.lower(),
                        size=member.file_size,
                        status="skipped",
                        error="Mục trong ZIP có đường dẫn không an toàn.",
                    )
                )
                continue
            if safe_name in seen_names:
                results.append(
                    _result(
                        filename=display_name,
                        extension=Path(safe_name).suffix.lower(),
                        size=member.file_size,
                        status="skipped",
                        error="Tên mục trong ZIP bị trùng.",
                    )
                )
                continue
            seen_names.add(safe_name)

            path_parts = PurePosixPath(safe_name).parts[:-1]
            if any(part in _IGNORED_ARCHIVE_DIRS for part in path_parts):
                continue

            extension = Path(safe_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="skipped",
                        error=f"Định dạng mục trong ZIP không được hỗ trợ: {extension or '[không có]'}.",
                    )
                )
                continue
            if member.flag_bits & 0x1:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="skipped",
                        error="Không hỗ trợ mục ZIP được mã hóa.",
                    )
                )
                continue
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="skipped",
                        error="Không hỗ trợ liên kết tượng trưng (symlink) trong ZIP.",
                    )
                )
                continue
            if member.file_size == 0:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=0,
                        status="skipped",
                        error="Mục trong ZIP rỗng.",
                    )
                )
                continue

            size_limit = max_file_size if max_file_size is not None else get_settings().max_file_size
            if member.file_size > size_limit:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="skipped",
                        error=f"Mục trong ZIP vượt quá giới hạn {size_limit} byte.",
                    )
                )
                continue
            remaining = max_archive_extracted_size - extracted_size
            if remaining <= 0:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="skipped",
                        error="Tổng dung lượng ZIP sau giải nén vượt quá giới hạn.",
                    )
                )
                continue

            try:
                with archive.open(member, "r") as entry:
                    raw_entry = entry.read(min(size_limit, remaining) + 1)
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=member.file_size,
                        status="error",
                        error=f"Không thể đọc mục trong ZIP: {exc}",
                    )
                )
                continue

            if len(raw_entry) > size_limit or len(raw_entry) > remaining:
                extracted_size = max_archive_extracted_size
                results.append(
                    _result(
                        filename=display_name,
                        extension=extension,
                        size=len(raw_entry),
                        status="skipped",
                        error="Mục trong ZIP vượt quá giới hạn giải nén an toàn.",
                    )
                )
                continue

            extracted_size += len(raw_entry)
            result = _load_raw_file(
                display_name,
                raw_entry,
                max_file_size=max_file_size,
            )
            results.append(result)
            if result["status"] == "loaded":
                loaded_count += 1

        archive_record["error"] = f"Đã giải nén an toàn {loaded_count} file được hỗ trợ."
        return results


def _safe_archive_member_name(member_name: str) -> str | None:
    """Normalize a ZIP member path and reject traversal/absolute paths."""
    normalized = member_name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or _WINDOWS_DRIVE_RE.match(normalized):
        return None
    parts = PurePosixPath(normalized).parts
    if any(part in {"..", ""} for part in parts):
        return None
    safe_parts = [part for part in parts if part != "."]
    return "/".join(safe_parts) or None


def _get_filename(file_input: Any) -> str:
    """Get a safe display name without allowing path components in metadata."""
    if isinstance(file_input, (str, Path)):
        candidate = str(file_input)
    else:
        candidate = str(getattr(file_input, "name", "uploaded_file.txt"))

    candidate = candidate.replace("\\", "/").rstrip("/")
    return candidate.rsplit("/", maxsplit=1)[-1] or "uploaded_file.txt"


def _read_bytes(file_input: Any) -> bytes:
    """Read bytes from supported file input types."""
    if isinstance(file_input, (str, Path)):
        return Path(file_input).read_bytes()
    if isinstance(file_input, (bytes, bytearray, memoryview)):
        return bytes(file_input)

    getvalue = getattr(file_input, "getvalue", None)
    if callable(getvalue):
        value = getvalue()
    else:
        read = getattr(file_input, "read", None)
        if not callable(read):
            raise TypeError("Đầu vào không phải là path, bytes hoặc file object.")
        value = _read_and_restore_position(read, file_input)

    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("File object không trả về bytes.")


def _read_and_restore_position(read: Any, file_input: Any) -> Any:
    """Read a file-like object and restore its cursor where possible."""
    tell = getattr(file_input, "tell", None)
    seek = getattr(file_input, "seek", None)
    position = tell() if callable(tell) else None
    value = read()
    if position is not None and callable(seek):
        seek(position)
    return value


def _looks_binary(data: bytes) -> bool:
    """Detect common binary content while allowing normal UTF-8 text."""
    if b"\x00" in data:
        return True
    sample = data[:4096]
    if not sample:
        return False
    control_count = sum(byte < 32 and byte not in {9, 10, 13} for byte in sample)
    return control_count / len(sample) > 0.02


def _decode_text(data: bytes, filename: str = "unknown") -> str:
    """Decode text as UTF-8, with safe Windows/code-page fallbacks."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("text encoding fallback used for file: %s", filename)
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError:
            logger.warning("text encoding replacement used for file: %s", filename)
            return data.decode("utf-8", errors="replace")


def _extract_pdf_text(data: bytes) -> str:
    """Extract text from every page of a PDF using PyMuPDF."""
    with fitz.open(stream=data, filetype="pdf") as document:
        pages = [page.get_text("text") for page in document]
    return "\n\n".join(page for page in pages if page)


def _extract_docx_text(data: bytes) -> str:
    """Extract paragraphs and table cells from a DOCX document."""
    document = Document(io.BytesIO(data))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(part for part in parts if part)


def _result(
    *,
    filename: str,
    extension: str,
    size: int,
    content: str = "",
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Create the stable metadata shape used by later pipeline stages."""
    result = {
        "filename": filename,
        "extension": extension,
        "size": size,
        "content": content,
        "content_sha256": sha256_content(content) if content else None,
        "status": status,
        "error": error,
    }
    if status == "loaded":
        logger.info("file loaded: %s (%s bytes)", filename, size)
    elif status == "expanded":
        logger.info("ZIP expanded: %s (%s bytes)", filename, size)
    elif status == "skipped":
        logger.info("file skipped: %s (%s)", filename, error or "no reason")
    else:
        logger.error("file error: %s (%s)", filename, error or "unknown error")
    return result
