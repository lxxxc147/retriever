"""M7 Presenter — human-readable CLI rendering (PRD §2.7).

All output goes through plain ``print(..., file=out)`` so tests can capture
it with an ``io.StringIO`` and the CLI keeps its UTF-8 stdout guard.
"""

from __future__ import annotations

import sys
from datetime import UTC
from pathlib import Path
from typing import TextIO

from .history import HistoryEntry
from .intent import ParsedIntent
from .models import DownloadResult, SearchResult

_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "github": "GitHub",
    "scholar": "Scholar",
    "web": "Web",
    "pdf": "PDF",
}


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
        summary_line = str(result.extra.get("summary") or "").strip()
        if summary_line:
            print(f"    Summary: {summary_line}", file=stream)
        summary: list[str] = []
        if result.authors:
            summary.append("Authors: " + ", ".join(result.authors[:5]))
        if result.published_at is not None:
            summary.append("Published: " + result.published_at.date().isoformat())
        if summary:
            print("    " + " | ".join(summary), file=stream)


def render_download_batch(
    count: int,
    dest_dir: Path,
    out: TextIO | None = None,
) -> None:
    """Announce a concurrent download batch.

    Downloads run in parallel through :class:`~retriever.downloader.DownloadManager`,
    so per-file progress lines would interleave; one batch line plus the closing
    summary keeps the output readable (F5.7 progress bars are Phase 2).

    Args:
        count: How many files were handed to the manager.
        dest_dir: The per-task directory the files land in.
        out: Output stream; ``None`` resolves to ``sys.stdout`` at call time.
    """
    stream = sys.stdout if out is None else out
    print(f"Downloading {count} file(s) in parallel -> {dest_dir} ...", file=stream)


def render_summary(
    outcomes: list[tuple[SearchResult, DownloadResult]],
    failure_log: Path | None = None,
    out: TextIO | None = None,
) -> None:
    """Render the end-of-task download summary with per-item failure reasons.

    Args:
        outcomes: Pairs of the searched result and its download outcome.
        failure_log: Path of ``download_failures.log`` when at least one download
            failed and the manager wrote one (PRD F5.6).
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
    if failed and failure_log is not None:
        print(f"  failure details: {failure_log}", file=stream)


def render_interactive_welcome(out: TextIO | None = None) -> None:
    """Open an interactive session (F8.2) and point at the help command.

    Args:
        out: Output stream; ``None`` resolves to the current ``sys.stdout``.
    """
    stream = sys.stdout if out is None else out
    print("Retriever 交互模式 — 输入检索内容开始，输入 help 查看命令，q 退出。", file=stream)


def render_interactive_help(out: TextIO | None = None) -> None:
    """List the interactive commands (F8.2).

    Args:
        out: Output stream; ``None`` resolves to the current ``sys.stdout``.
    """
    stream = sys.stdout if out is None else out
    print(
        """可用命令：
  <检索内容>     发起检索（会话第一轮）
  <自然语言>     基于当前结果继续 refine（F7.3），如“只要 2024 年以后的”
  <序号>         查看第 N 条结果详情（如 2）
  d              下载当前全部结果
  d N            下载第 N 条结果（如 d 2）
  help / ?       显示本帮助
  q / quit       退出交互模式""",
        file=stream,
    )


def render_refine_prompt(out: TextIO | None = None) -> None:
    """Print the one-line follow-up hint shown after every result page (F7.2).

    Args:
        out: Output stream; ``None`` resolves to the current ``sys.stdout``.
    """
    stream = sys.stdout if out is None else out
    print(
        "— 继续输入可 refine 结果；d 下载全部、d N 下载第 N 条、N 看详情、help 帮助、q 退出",
        file=stream,
    )


def render_detail(result: SearchResult, index: int, out: TextIO | None = None) -> None:
    """Render one result in full (interactive detail view).

    Args:
        result: The result to expand.
        index: Its 1-based position in the current list.
        out: Output stream; ``None`` resolves to the current ``sys.stdout``.
    """
    stream = sys.stdout if out is None else out
    print(f"[{index}] [{_source_label(result.source)}] {result.title}", file=stream)
    print(f"    URL: {result.url}", file=stream)
    if result.download_url:
        print(f"    Download: {result.download_url}", file=stream)
    if result.authors:
        print("    Authors: " + ", ".join(result.authors), file=stream)
    if result.published_at is not None:
        print("    Published: " + result.published_at.date().isoformat(), file=stream)
    if result.abstract:
        print(file=stream)
        print("    " + result.abstract.replace("\n", " "), file=stream)


def render_history(entries: list[HistoryEntry], out: TextIO | None = None) -> None:
    """Render past tasks, newest first (PRD §2.7 F7.4).

    Args:
        entries: History entries already ordered newest first.
        out: Output stream; ``None`` resolves to ``sys.stdout`` at call time.
    """
    stream = sys.stdout if out is None else out
    if not entries:
        print('No history yet — run `retrieve "你的查询"` first.', file=stream)
        return

    print(f"Recent {len(entries)} task(s):", file=stream)
    for entry in entries:
        stamp = entry.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
        print(f"  {stamp}  {entry.task_id}  [{entry.intent or '?'}]  {entry.query}", file=stream)
        details = f"{entry.result_count} result(s), {entry.downloaded} downloaded"
        if entry.failed:
            details += f", {entry.failed} failed"
        if entry.parser:
            details += f", parser={entry.parser}"
        print(f"      {details}", file=stream)
        if entry.manifest_path:
            print(f"      manifest: {entry.manifest_path}", file=stream)
