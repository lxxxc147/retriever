"""Tests for the presenter rendering helpers. Pure string assertions — no network."""

import io
from datetime import UTC, datetime
from pathlib import Path

from retriever.models import DownloadResult, SearchResult
from retriever.presenter import render_download_progress, render_results, render_summary


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
