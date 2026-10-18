"""Shared, dependency-free helpers used by adapters and the download manager.

This module deliberately lives OUTSIDE the ``adapters`` package: PRD §4.3 iron
rule #1 forbids adapters from importing each other, but file-name sanitizing,
hashing, and collision-safe naming still need a single implementation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Characters kept in sanitized file names: ASCII alphanumerics, CJK chars,
# dot, underscore and dash. Everything else collapses to a single underscore.
_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")
_QUERY_TAIL = re.compile(r"[?#].*$")
_EXTENSION = re.compile(r"\.[0-9A-Za-z]{1,10}")


def sanitize_filename(title: str, max_len: int = 80) -> str:
    """Make a file name safe: keep alphanumerics/CJK/._-, spaces to "_", truncate.

    Args:
        title: Raw title, potentially containing illegal characters.
        max_len: Maximum length of the sanitized name (before extension).

    Returns:
        A file-name-safe string; ``"untitled"`` when nothing usable remains.
    """
    name = (title or "").replace(" ", "_")
    name = _SAFE_CHARS.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name[:max_len] or "untitled"


def url_extension(url: str) -> str:
    """Derive a safe lowercase file extension from a URL path.

    Args:
        url: Source URL; query string and fragment are ignored.

    Returns:
        Extension such as ``".pdf"``, or ``""`` when none can be inferred.
    """
    if not url:
        return ""
    path = _QUERY_TAIL.sub("", url)
    suffix = Path(path).suffix
    return suffix.lower() if _EXTENSION.fullmatch(suffix) else ""


def ensure_extension(stem: str, ext: str) -> str:
    """Append ``ext`` to ``stem`` unless it already ends with it (case-insensitive).

    Prevents doubled suffixes such as ``"main.py.py"`` when a title already
    carries the extension.

    Args:
        stem: Sanitized file name without (or possibly with) the extension.
        ext: Extension including the leading dot, e.g. ``".py"``.

    Returns:
        The name carrying exactly one extension.
    """
    if not ext or stem.lower().endswith(ext.lower()):
        return stem
    return f"{stem}{ext}"


def build_filename(seq: int, title: str, ext: str = "", max_len: int = 80) -> str:
    """Build a ``{序号}_{安全化标题}{ext}`` file name (PRD §2.5 F5.2).

    Args:
        seq: 1-based index of the result inside the current task.
        title: Raw result title.
        ext: Extension including the leading dot, e.g. ``".pdf"``.
        max_len: Maximum length of the sanitized title part.

    Returns:
        The composed, file-name-safe name with exactly one extension.
    """
    stem = sanitize_filename(title, max_len=max_len)
    return f"{seq:02d}_{ensure_extension(stem, ext)}"


def sha256_of(data: bytes) -> str:
    """Return the hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_path(directory: Path, filename: str) -> Path:
    """Return ``directory/filename``, appending ``_2``, ``_3``... on conflict.

    Args:
        directory: Target directory (assumed to exist).
        filename: Desired file name.

    Returns:
        A path that does not exist yet, preserving the original extension.
    """
    candidate = Path(directory) / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    index = 2
    while True:
        candidate = Path(directory) / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
