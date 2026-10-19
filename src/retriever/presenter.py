"""M7 Presenter — human-readable CLI rendering (PRD §2.7).

All output goes through plain ``print(..., file=out)`` so tests can capture
it with an ``io.StringIO`` and the CLI keeps its UTF-8 stdout guard.
"""

from __future__ import annotations

import sys
from typing import TextIO

from .intent import ParsedIntent
from .models import DownloadResult, SearchResult

_SOURCE_LABELS = {"arxiv": "arXiv", "github": "GitHub"}


def _source_label(source: str) -> str:
    """Human-friendly tag for a result source."""
    return _SOURCE_LABELS.get(source, source)


def render_intent(
    parsed: ParsedIntent,
    source: str,
    model: str | None = None,
    out: TextIO | None = None,
) -> None:
    """Render the parsed plan so the user sees what the agent understood.

    Printing the plan is what makes an LLM-parsed run verifiable at a glance:
    the same query shows ``parser=rules`` or ``parser=llm(<model>)`` depending on
    which parser produced it (PRD §2.7 F7.1 "关键元数据").

    Args:
        parsed: The parsed intent (intent, keywords, filter slots).
        source: Which parser produced it — ``"llm"`` or ``"rules"``.
        model: Model name, used in the label when the LLM parsed the query.
        out: Output stream; ``None`` resolves to ``sys.stdout`` at call time.
    """
    stream = sys.stdout if out is None else out
    parser_label = f"llm({model})" if source == "llm" and model else source

    parts = [f"intent={parsed.intent}", "keywords=" + (", ".join(parsed.keywords) or "-")]
    slots = _filter_slots(parsed)
    if slots:
        parts.append(" | ".join(slots))
    parts.append(f"parser={parser_label}")
    print("Plan: " + " | ".join(parts), file=stream)


def _filter_slots(parsed: ParsedIntent) -> list[str]:
    """Format the non-empty filter slots of ``parsed`` for the plan line."""
    filters = parsed.filters
    slots: list[str] = []
    if filters.authors:
        slots.append("authors=" + ",".join(filters.authors))
    if filters.year_from is not None or filters.year_to is not None:
        start = filters.year_from if filters.year_from is not None else "?"
        end = filters.year_to if filters.year_to is not None else "?"
        slots.append(f"years={start}-{end}")
    if filters.language:
        slots.append(f"language={filters.language}")
    if filters.min_stars is not None:
        slots.append(f"min_stars={filters.min_stars}")
    return slots


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
