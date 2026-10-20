"""Tests for the X-4 history module. Pure filesystem, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from retriever.history import (
    HistoryEntry,
    load_history,
    record_task,
    save_history,
)


def _entry(**overrides: object) -> HistoryEntry:
    """Build a history entry with sensible defaults."""
    data: dict = {
        "task_id": "abc123",
        "query": "transformer survey",
        "created_at": datetime(2026, 10, 20, 9, 30, tzinfo=UTC),
        "intent": "paper",
        "parser": "llm",
        "sources": ["arxiv"],
        "result_count": 3,
        "downloaded": 2,
        "failed": 0,
    }
    data.update(overrides)
    return HistoryEntry(**data)  # type: ignore[arg-type]


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    """An entry written through save_history comes back unchanged."""
    path = save_history(_entry(), tmp_path)

    assert path.exists()
    assert path.name == "2026-10-20.jsonl"
    loaded = load_history(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "abc123"
    assert loaded[0].sources == ["arxiv"]
    assert loaded[0].parser == "llm"


def test_entries_are_ordered_newest_first(tmp_path: Path) -> None:
    """Reading merges day files and sorts by timestamp, newest first."""
    save_history(_entry(task_id="old", created_at=datetime(2026, 10, 18, tzinfo=UTC)), tmp_path)
    save_history(_entry(task_id="new", created_at=datetime(2026, 10, 20, tzinfo=UTC)), tmp_path)

    assert [entry.task_id for entry in load_history(tmp_path)] == ["new", "old"]


def test_limit_caps_the_result(tmp_path: Path) -> None:
    """``limit`` keeps only the most recent entries."""
    for index in range(4):
        save_history(
            _entry(task_id=f"t{index}",
                   created_at=datetime(2026, 10, 20, tzinfo=UTC) + timedelta(minutes=index)),
            tmp_path,
        )

    assert [entry.task_id for entry in load_history(tmp_path, limit=2)] == ["t3", "t2"]


def test_non_positive_limit_returns_nothing(tmp_path: Path) -> None:
    """A zero/negative limit is treated as "show nothing", not "show all"."""
    save_history(_entry(), tmp_path)
    assert load_history(tmp_path, limit=0) == []


def test_missing_directory_is_empty_not_an_error(tmp_path: Path) -> None:
    """Before the first run there is simply no history."""
    assert load_history(tmp_path / "absent") == []


def test_malformed_line_is_skipped(tmp_path: Path) -> None:
    """A corrupt line (half a write, manual edit) does not hide the other rows."""
    save_history(_entry(task_id="good"), tmp_path)
    day_file = tmp_path / "2026-10-20.jsonl"
    with day_file.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    loaded = load_history(tmp_path)
    assert [entry.task_id for entry in loaded] == ["good"]


def test_naive_timestamp_is_treated_as_utc(tmp_path: Path) -> None:
    """A naive datetime (hand-written record) still lands in a day file."""
    # 故意构造 naive datetime（模拟手写/老版本记录），因此这里必须不带 tzinfo
    entry = _entry(created_at=datetime(2026, 10, 20, 12, 0))  # noqa: DTZ001
    path = save_history(entry, tmp_path)

    assert path.name == "2026-10-20.jsonl"
    assert load_history(tmp_path)[0].task_id == "abc123"


def test_record_task_writes_and_reports_the_file(tmp_path: Path) -> None:
    """``record_task`` is the convenience wrapper the CLI uses."""
    path = record_task(
        task_id="t1", query="rag", intent="repo", parser="rules", sources=["github"],
        result_count=2, downloaded=1, failed=1, manifest_path=tmp_path / "m.json",
        history_dir=tmp_path,
    )

    assert path is not None
    entry = load_history(tmp_path)[0]
    assert entry.query == "rag"
    assert entry.manifest_path is not None
    assert entry.manifest_path.endswith("m.json")


def test_record_task_swallows_write_failures(tmp_path: Path) -> None:
    """History is best-effort: a broken path returns None instead of raising."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")

    assert record_task(task_id="t", query="q", history_dir=blocker) is None
