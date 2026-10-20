"""Tests for the presenter rendering helpers. Pure string assertions — no network."""

import io
from datetime import UTC, datetime
from pathlib import Path

from retriever.history import HistoryEntry
from retriever.intent import ParsedIntent
from retriever.models import DownloadResult, Filters, SearchResult
from retriever.presenter import (
    render_download_batch,
    render_history,
    render_intent,
    render_results,
    render_summary,
)


def _result(**overrides: object) -> SearchResult:
    data: dict = {
        "source": "arxiv",
        "title": "A Paper",
        "url": "https://arxiv.org/abs/2401.00001",
    }
    data.update(overrides)
    return SearchResult(**data)  # type: ignore[arg-type]


def test_render_results_numbered_with_source_labels() -> None:
    out = io.StringIO()
    results = [
        _result(
            source="arxiv",
            title="Attention Is All You Need",
            authors=["Alice", "Bob"],
            published_at=datetime(2017, 6, 12, tzinfo=UTC),
        ),
        _result(source="github", title="org/repo", url="https://github.com/org/repo"),
    ]

    render_results(results, out=out)

    text = out.getvalue()
    assert "[1] [arXiv] Attention Is All You Need" in text
    assert "[2] [GitHub] org/repo" in text
    assert "https://arxiv.org/abs/2401.00001" in text
    assert "Authors: Alice, Bob" in text
    assert "Published: 2017-06-12" in text


def test_render_results_empty_shows_friendly_message() -> None:
    out = io.StringIO()
    render_results([], out=out)
    assert "No results found" in out.getvalue()


def test_render_results_shows_the_pipeline_summary() -> None:
    """A result carrying ``extra["summary"]`` renders it under the URL."""
    out = io.StringIO()
    render_results([_result(extra={"summary": "一篇讲注意力的论文"})], out=out)
    assert "Summary: 一篇讲注意力的论文" in out.getvalue()


def test_render_results_without_summary_renders_no_summary_line() -> None:
    """Results that never went through the summarizer show no empty Summary line."""
    out = io.StringIO()
    render_results([_result()], out=out)
    assert "Summary:" not in out.getvalue()


def test_render_download_batch_line() -> None:
    """A concurrent batch is announced once, with its destination."""
    out = io.StringIO()
    render_download_batch(3, Path("downloads/abc"), out=out)
    assert "Downloading 3 file(s) in parallel" in out.getvalue()
    assert "downloads" in out.getvalue()


def test_render_summary_counts_success_and_failure() -> None:
    out = io.StringIO()
    ok = _result(source="arxiv", title="Good paper")
    bad = _result(source="github", title="Broken repo")
    outcomes = [
        (ok, DownloadResult(success=True, local_path=Path("/tmp/good.pdf"), size_bytes=10, sha256="x")),
        (bad, DownloadResult(success=False, error="boom")),
    ]

    render_summary(outcomes, failure_log=Path("downloads/abc/download_failures.log"), out=out)

    text = out.getvalue()
    assert "1 succeeded / 1 failed" in text
    assert "Good paper" in text
    assert "Broken repo" in text
    assert "boom" in text
    assert "download_failures.log" in text


def test_render_summary_omits_the_failure_log_when_all_succeeded() -> None:
    """No failures means no failure-log line, even if a path was supplied."""
    out = io.StringIO()
    outcomes = [(_result(), DownloadResult(success=True, local_path=Path("/tmp/a.pdf"),
                                           size_bytes=1, sha256="x"))]
    render_summary(outcomes, failure_log=Path("downloads/abc/download_failures.log"), out=out)

    assert "failure details" not in out.getvalue()


def _history_entry(**overrides: object) -> HistoryEntry:
    data: dict = {
        "task_id": "abc123",
        "query": "transformer survey",
        "created_at": datetime(2026, 10, 20, 9, 30, tzinfo=UTC),
        "intent": "paper",
        "parser": "llm",
        "result_count": 3,
        "downloaded": 2,
        "failed": 1,
        "manifest_path": "downloads/abc123/manifest.json",
    }
    data.update(overrides)
    return HistoryEntry(**data)  # type: ignore[arg-type]


def test_render_history_lists_tasks_newest_first() -> None:
    """Each past task shows its timestamp, intent, query and counters."""
    out = io.StringIO()
    render_history([_history_entry()], out=out)

    text = out.getvalue()
    assert "Recent 1 task(s)" in text
    assert "2026-10-20 09:30" in text
    assert "abc123" in text
    assert "[paper]" in text
    assert "transformer survey" in text
    assert "3 result(s), 2 downloaded, 1 failed" in text
    assert "parser=llm" in text
    assert "manifest: downloads/abc123/manifest.json" in text


def test_render_history_empty_shows_a_hint() -> None:
    """No history yet explains what to run instead of printing nothing."""
    out = io.StringIO()
    render_history([], out=out)

    assert "No history yet" in out.getvalue()


def _parsed(**overrides: object) -> ParsedIntent:
    data: dict = {
        "intent": "paper",
        "keywords": ["transformer"],
        "raw_query": "q",
        "filters": Filters(),
    }
    data.update(overrides)
    return ParsedIntent(**data)  # type: ignore[arg-type]


def test_render_intent_labels_the_llm_parser() -> None:
    """A model-parsed plan shows the model name so the run is verifiable."""
    out = io.StringIO()
    render_intent(_parsed(), "llm", model="deepseek-chat", out=out)

    text = out.getvalue()
    assert text.startswith("Plan: ")
    assert "intent=paper" in text
    assert "keywords=transformer" in text
    assert "parser=llm(deepseek-chat)" in text


def test_render_intent_labels_the_rule_parser_and_slots() -> None:
    """Filter slots are shown next to the parser that produced them."""
    out = io.StringIO()
    parsed = _parsed(
        intent="repo",
        keywords=["RAG"],
        filters=Filters(year_from=2023, language="python", min_stars=500, authors=["Wang"]),
    )
    render_intent(parsed, "rules", out=out)

    text = out.getvalue()
    assert "intent=repo" in text
    assert "authors=Wang" in text
    assert "years=2023-?" in text
    assert "language=python" in text
    assert "min_stars=500" in text
    assert "parser=rules" in text
