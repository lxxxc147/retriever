"""M7 Presenter — human-readable CLI rendering (PRD §2.7).

All output goes through plain ``print(..., file=out)`` so tests can capture
it with an ``io.StringIO`` and the CLI keeps its UTF-8 stdout guard.
"""

from __future__ import annotations

import sys
from typing import TextIO

from .models import DownloadResult, SearchResult

_SOURCE_LABELS = {"arxiv": "arXiv", "github": "GitHub"}


def _source_label(source: str) -> str:
    """Human-friendly tag for a result source."""
    return _SOURCE_LABELS.get(source, source)


def render_results(results: list[SearchResult], out: TextIO | None = None) -> None:
    """Render numbered search results, each tagged with its source.

    Args:
        results: Aggregated results (already grouped and ordered upstream).
        out: Output stream; ``None`` resolves to the current ``sys.stdout``
            at call time (so test capture fixtures keep working).
    """
    stream = sys.stdout if out is None else out
    if not results:
        print("No results found — try broader keywords or fewer filters.", file=stream)
        return

    print(f"Found {len(results)} result(s):", file=stream)
    print(file=stream)
    for index, result in enumerate(results, start=1):
        print(f"[{index}] [{_source_label(result.source)}] {result.title}", file=stream)
        print(f"    URL: {result.url}", file=stream)
        summary: list[str] = []
        if result.authors:
            summary.append("Authors: " + ", ".join(result.authors[:5]))
        if result.published_at is not None:
            summary.append("Published: " + result.published_at.date().isoformat())
        if summary:
            print("    " + " | ".join(summary), file=stream)


def render_download_progress(
    current: int,
    total: int,
    result: SearchResult,
    out: TextIO | None = None,
) -> None:
    """Print one download progress line, e.g. ``Downloading [1/3] <title> ...``."""
    stream = sys.stdout if out is None else out
    print(f"Downloading [{current}/{total}] {result.title} ...", file=stream)


def render_summary(
    outcomes: list[tuple[SearchResult, DownloadResult]],
    out: TextIO | None = None,
) -> None:
    """Render the end-of-task download summary with per-item failure reasons.

    Args:
        outcomes: Pairs of the searched result and its download outcome.
        out: Output stream; ``None`` resolves to the current ``sys.stdout``
            at call time (so test capture fixtures keep working).
    """
    stream = sys.stdout if out is None else out
    succeeded = [(result, download) for result, download in outcomes if download.success]
    failed = [(result, download) for result, download in outcomes if not download.success]

    print(file=stream)
    print(f"Download complete: {len(succeeded)} succeeded / {len(failed)} failed.", file=stream)
    for result, download in succeeded:
        print(f"  ✓ {result.title} -> {download.local_path}", file=stream)
    for result, download in failed:
        print(f"  ✗ {result.title}: {download.error or 'unknown error'}", file=stream)
