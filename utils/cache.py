"""Best-effort SHA-256 JSON cache for file and chunk analysis."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def sha256_content(content: bytes | str) -> str:
    """Return a stable SHA-256 digest for bytes or UTF-8 text."""
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class ContentCache:
    """Filesystem JSON cache that never makes the main pipeline fail."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("cache directory unavailable: %s", exc)

    def get(self, content_hash: str) -> dict[str, Any] | None:
        """Read one cached JSON object, returning None on a cache miss/error."""
        cache_path = self.cache_dir / f"{content_hash}.json"
        try:
            if not cache_path.is_file():
                return None
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            logger.warning("cache read failed for %s: %s", content_hash, exc)
            return None

    def set(self, content_hash: str, payload: dict[str, Any]) -> bool:
        """Write one JSON object; return False if the cache is unavailable."""
        cache_path = self.cache_dir / f"{content_hash}.json"
        try:
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("cache write failed for %s: %s", content_hash, exc)
            return False
