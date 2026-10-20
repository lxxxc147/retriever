"""Tests for the presenter rendering helpers. Pure string assertions — no network."""

import io
from datetime import UTC, datetime
from pathlib import Path

from retriever.intent import ParsedIntent
from retriever.models import DownloadResult, Filters, SearchResult
from retriever.presenter import (
    render_download_progress,
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


def test_render_download_progress_line() -> None:
    out = io.StringIO()
    render_download_progress(1, 3, _result(title="Paper A"), out=out)
    assert "Downloading [1/3] Paper A ..." in out.getvalue()


def test_render_summary_counts_success_and_failure() -> None:
    out = io.StringIO()
    ok = _result(source="arxiv", title="Good paper")
    bad = _result(source="github", title="Broken repo")
    outcomes = [
        (ok, DownloadResult(success=True, local_path=Path("/tmp/good.pdf"), size_bytes=10, sha256="x")),
        (bad, DownloadResult(success=False, error="boom")),
    ]

    render_summary(outcomes, out=out)

    text = out.getvalue()
    assert "1 succeeded / 1 failed" in text
    assert "Good paper" in text
    assert "Broken repo" in text
    assert "boom" in text


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
