"""X-4 search history — append-only log of past tasks (PRD §2.7 F7.4).

Layout: one JSON object per line, one file per day (``history/2026-10-20.jsonl``).
Append-only keeps writes cheap and crash-safe; reading tolerates the messy parts
of real usage (missing directory, a half-written trailing line, unknown fields
from a newer version) so ``retrieve history`` never fails because of the log.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

DEFAULT_HISTORY_DIR = Path("./history")
DEFAULT_LIMIT = 10


class HistoryEntry(BaseModel):
    """One recorded ``retrieve`` task."""

    task_id: str
    query: str
    created_at: datetime
    intent: str = ""
    parser: str = ""
    sources: list[str] = Field(default_factory=list)
    result_count: int = 0
    downloaded: int = 0
    failed: int = 0
    manifest_path: str | None = None


def _resolve_dir(history_dir: Path | None) -> Path:
    """Return the history directory, falling back to the module default at call time.

    Resolving lazily (instead of binding the default in the signature) keeps the
    directory overridable in tests by monkeypatching :data:`DEFAULT_HISTORY_DIR`.
    """
    return Path(history_dir if history_dir is not None else DEFAULT_HISTORY_DIR)


def _day_file(history_dir: Path, moment: datetime) -> Path:
    """Return the JSONL file holding ``moment``'s entries."""
    return Path(history_dir) / f"{moment.astimezone(UTC).date().isoformat()}.jsonl"


def save_history(record: HistoryEntry, history_dir: Path | None = None) -> Path:
    """Append one entry to the history log (creating the directory if needed).

    Args:
        record: The task record to persist.
        history_dir: Override for ``./history``.

    Returns:
        Path of the file the entry was appended to.

    Raises:
        OSError: When the directory or file cannot be written; callers treat
            history as best-effort and must not fail the run because of it.
    """
    directory = _resolve_dir(history_dir)
    directory.mkdir(parents=True, exist_ok=True)
    moment = record.created_at if record.created_at.tzinfo else record.created_at.replace(tzinfo=UTC)
    path = _day_file(directory, moment)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    logger.debug("history entry appended to {}", path)
    return path


def load_history(history_dir: Path | None = None, limit: int = DEFAULT_LIMIT) -> list[HistoryEntry]:
    """Return the most recent entries, newest first.

    Args:
        history_dir: Override for ``./history``.
        limit: Maximum number of entries to return (``<= 0`` returns nothing).

    Returns:
        Parsed entries; malformed lines are skipped with a warning, and a missing
        directory simply yields an empty list.
    """
    directory = _resolve_dir(history_dir)
    if limit <= 0 or not directory.exists():
        return []

    entries: list[HistoryEntry] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("cannot read history file {}: {}", path, exc)
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(HistoryEntry.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning("skipping malformed history line in {}: {}", path, exc)

    entries.sort(key=lambda entry: entry.created_at, reverse=True)
    return entries[:limit]


def record_task(
    *,
    task_id: str,
    query: str,
    intent: str = "",
    parser: str = "",
    sources: list[str] | None = None,
    result_count: int = 0,
    downloaded: int = 0,
    failed: int = 0,
    manifest_path: Path | str | None = None,
    history_dir: Path | None = None,
    created_at: datetime | None = None,
) -> Path | None:
    """Build and persist one history entry, swallowing write failures.

    The CLI calls this after a run finishes; history is a convenience, so a
    read-only or full disk logs a warning instead of failing the search.

    Args:
        task_id: Task identifier (also used to name the download directory).
        query: Original user query.
        intent: Parsed intent name.
        parser: Which intent parser produced the plan (``llm`` / ``rules``).
        sources: Data sources that answered.
        result_count: Number of results after the M4 pipeline.
        downloaded: Successfully downloaded files.
        failed: Failed downloads.
        manifest_path: Where the task manifest was written, if any.
        history_dir: Override for ``./history``.
        created_at: Timestamp override (tests).

    Returns:
        The file the entry was appended to, or ``None`` when writing failed.
    """
    entry = HistoryEntry(
        task_id=task_id,
        query=query,
        created_at=created_at or datetime.now(UTC),
        intent=intent,
        parser=parser,
        sources=list(sources or []),
        result_count=result_count,
        downloaded=downloaded,
        failed=failed,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )
    try:
        return save_history(entry, history_dir)
    except OSError as exc:
        logger.warning("could not write the history entry: {}", exc)
        return None
