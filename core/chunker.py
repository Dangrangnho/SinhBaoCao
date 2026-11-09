"""Chunking strategies for project documents."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from config import get_settings
from utils.cache import sha256_content


logger = logging.getLogger(__name__)


class ChunkingStrategy(Protocol):
    """Interface implemented by character, AST or Tree-sitter chunkers."""

    def chunk(self, filename: str, content: str) -> list[dict[str, Any]]:
        """Split one document into metadata-preserving chunks."""


@dataclass(frozen=True)
class CharacterChunker:
    """Simple overlapping character chunker used by the MVP."""

    chunk_size: int = 6000
    chunk_overlap: int = 500

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size phải lớn hơn 0.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap không được âm.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")

    def chunk(self, filename: str, content: str) -> list[dict[str, Any]]:
        """Split content while preserving source filename and character offsets."""
        if not content:
            return []

        chunks: list[dict[str, Any]] = []
        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk = {
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "content": content[start:end],
                    "content_sha256": sha256_content(content[start:end]),
                    "start_char": start,
                    "end_char": end,
                }
            chunks.append(chunk)
            logger.info(
                "chunk created: %s#%s (%s chars)",
                filename,
                chunk_index,
                end - start,
            )
            chunk_index += 1
            if end >= len(content):
                break
            start += step

        return chunks


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    filename: str = "unknown",
) -> list[dict[str, Any]]:
    """Chunk one text value using the configured MVP strategy."""
    settings = get_settings()
    strategy = CharacterChunker(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        ),
    )
    return strategy.chunk(filename=filename, content=text)


def chunk_documents(
    documents: Iterable[Mapping[str, Any]],
    strategy: ChunkingStrategy | None = None,
) -> list[dict[str, Any]]:
    """Chunk loaded/preprocessed records and skip non-loadable records."""
    selected_strategy = strategy or CharacterChunker(
        chunk_size=get_settings().chunk_size,
        chunk_overlap=get_settings().chunk_overlap,
    )
    chunks: list[dict[str, Any]] = []
    for document in documents:
        if document.get("status") != "loaded":
            continue
        chunks.extend(
            selected_strategy.chunk(
                filename=str(document.get("filename", "unknown")),
                content=str(document.get("content", "")),
            )
        )
    return chunks
