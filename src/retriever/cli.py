"""CLI entry point — ``retrieve "query"`` (argparse).

Flow: parse intent → route via :class:`Router` → M4 pipeline → render results →
optional download through :class:`~retriever.downloader.DownloadManager` → record
history (PRD §2.8).

Details that matter:

- the intent stage is pluggable (``--parser auto|rule|llm``) and always falls back
  to the rule parser, so a model outage never breaks a search;
- downloads go through the DownloadManager (M5), which brings concurrency,
  sha256 dedupe, retries, ``download_failures.log`` and the manifest — the CLI no
  longer hand-rolls any of that;
- ``--source`` accepts whatever the registry (``config/sources.yaml``) registers,
  and ``retrieve history`` replays the last tasks from ``./history`` (X-4).

The console-script entry point is ``app`` (pyproject ``[project.scripts]`` points
at ``retriever.cli:app``); :func:`main` accepts an explicit argv list so tests can
drive the full flow without subprocesses or network access.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .downloader import FAILURE_LOG, DownloadManager
from .history import DEFAULT_HISTORY_DIR, load_history, record_task
from .intent_llm import parse_intent_auto
from .llm import LLMClient
from .models import SearchQuery, SearchResult
from .pipeline import run_pipeline
from .presenter import (
    render_download_batch,
    render_history,
    render_intent,
    render_results,
    render_summary,
)
from .router import Router
from .summarizer import LLMSummarizer

# ``retrieve history`` — a bare positional switches to the history listing.
HISTORY_COMMAND = "history"

# Windows consoles default to GBK; force UTF-8 so CJK/emoji output works.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _run_sync(value: Any) -> Any:
    """Return ``value``, awaiting it first when it is a coroutine.

    Adapter and manager methods are ``async``; running them with
    :func:`asyncio.run` keeps the CLI flow plain and sequential.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the ``retrieve`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="retrieve",
        description="Retriever — intelligent search & download agent.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help=(
            'Natural-language search query (a source URL also works). '
            f'Use "{HISTORY_COMMAND}" to list past tasks.'
        ),
    )
    parser.add_argument(
        "--source",
        metavar="NAME",
        default="all",
        help=(
            "Restrict the search to one registered data source, e.g. arxiv, github "
            "(see config/sources.yaml). Default: all."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results per source (default: 5).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the results into --out/<task_id>/ and write manifest.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./downloads"),
        help="Download root directory (default: ./downloads).",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help=f"History directory (default: {DEFAULT_HISTORY_DIR}).",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help=(
            "Parse downloaded PDFs into structured JSON under --parsed-dir "
            "(M6, F6.1/F6.4). Needs the optional pymupdf dependency."
        ),
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("./parsed"),
        help="Root directory for parsed documents (default: ./parsed).",
    )
    parser.add_argument(
        "--parser",
        choices=["auto", "rule", "llm"],
        default="auto",
        help=(
            "Intent parser: 'auto' uses the LLM when an API key is configured, "
            "'rule' forces the deterministic regex/dictionary parser, "
            "'llm' forces the model (falls back to the rules on failure). "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Add a one-sentence Chinese summary per result (F4.3). Uses the LLM "
            "when an API key is configured and falls back to extractive summaries."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one retrieve task end to end.

    Args:
        argv: Argument list (``None`` → ``sys.argv``); tests pass explicit lists.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    history_dir = args.history_dir

    if args.query is None or args.query == HISTORY_COMMAND:
        return _show_history(history_dir, args.limit)

    if args.parser == "llm" and not LLMClient().configured:
        parser.error("--parser llm requires an LLM API key; set LLM_API_KEY in .env")

    outcome = parse_intent_auto(args.query, mode=args.parser)
    if outcome.fallback:
        print(f"warning: {outcome.note} — using the rule parser", file=sys.stderr)
    parsed = outcome.parsed
    parsed.limit = max(1, args.limit)
    render_intent(parsed, outcome.source, model=outcome.note if outcome.source == "llm" else None)

    router = Router()
    if args.source == "all":
        adapters = router.route(parsed)  # intent-driven adapter selection
    else:
        adapters = [a for a in router.adapters if a.name == args.source]
        if not adapters:
            available = ", ".join(sorted(a.name for a in router.adapters)) or "none"
            parser.error(f"unknown source {args.source!r}; available: {available}, all")

    query = SearchQuery(
        intent=parsed.intent,
        raw_query=parsed.raw_query,
        keywords=parsed.keywords,
        filters=parsed.filters,
        limit=parsed.limit,
    )
    results = router.search_all(query, adapters=adapters)

    summarizer = None
    if args.summary:
        summarizer = LLMSummarizer()
        if not summarizer.configured:
            print(
                "note: no LLM API key configured — using extractive summaries",
                file=sys.stderr,
            )
    # M4: cross-source dedupe + weighted ranking (+ optional summaries).
    results = run_pipeline(results, summarizer=summarizer)
    render_results(results)

    task_id = uuid.uuid4().hex[:12]
    manifest_path: Path | None = None
    downloaded = failed = 0

    if args.download and results:
        dest_dir = args.out / task_id
        manager = DownloadManager(adapters={adapter.name: adapter for adapter in adapters})
        render_download_batch(len(results), dest_dir)
        outcomes = _download_all(manager, results, dest_dir, task_id, args.query)
        failure_log = dest_dir / FAILURE_LOG
        render_summary(
            outcomes,
            failure_log=failure_log if failure_log.exists() else None,
        )
        downloaded = sum(1 for _, download in outcomes if download.success)
        failed = len(outcomes) - downloaded
        written = dest_dir / "manifest.json"
        if written.exists():
            manifest_path = written
            print(f"Manifest written to {written}")

        if args.parse and downloaded:
            ok, bad = _parse_downloads(outcomes, args.parsed_dir, task_id)
            if ok or bad:
                print(f"Parsed {ok} PDF file(s) -> {args.parsed_dir} ({bad} failed).")
    elif args.download:
        print("Nothing to download (no results).")

    record_task(
        task_id=task_id,
        query=args.query,
        intent=parsed.intent,
        parser=outcome.source,
        sources=[adapter.name for adapter in adapters],
        result_count=len(results),
        downloaded=downloaded,
        failed=failed,
        manifest_path=manifest_path,
        history_dir=history_dir,
    )
    return 0


def _download_all(
    manager: DownloadManager,
    results: list[SearchResult],
    dest_dir: Path,
    task_id: str,
    query: str,
) -> list[tuple[SearchResult, Any]]:
    """Run the manager and pair every result with its outcome.

    Args:
        manager: Configured download manager.
        results: Results to download, in display order.
        dest_dir: Per-task directory.
        task_id: Task id recorded in the manifest.
        query: Original query recorded in the manifest.

    Returns:
        ``(result, DownloadResult)`` pairs in the same order as ``results``.
    """
    try:
        outcomes = _run_sync(
            manager.download_all(results, dest_dir, task_id=task_id, query=query)
        )
    finally:
        _run_sync(manager.aclose())  # the manager owns its HTTP client here
    return list(zip(results, outcomes, strict=True))


def _show_history(history_dir: Path | None, limit: int) -> int:
    """Print the most recent tasks and return the exit code."""
    entries = load_history(history_dir, max(1, limit))
    render_history(entries)
    return 0


def _resolve_pdf_parser() -> Any:
    """Return :func:`retriever.parser.pdf.parse_pdf`, or ``None`` when unavailable.

    ``pymupdf`` ships native binaries, so importing it can fail on a machine where
    the wheel cannot load its DLL. The parser is therefore an *optional* runtime
    extra: ``--parse`` degrades with a note instead of crashing the run.
    """
    try:
        from .parser.pdf import parse_pdf
    except Exception as exc:  # noqa: BLE001 — 可选依赖缺失不该炸掉检索
        print(f"note: PDF parsing unavailable ({exc}); --parse skipped", file=sys.stderr)
        return None
    return parse_pdf


def _parse_downloads(
    outcomes: list[tuple[SearchResult, Any]],
    parsed_root: Path,
    task_id: str,
) -> tuple[int, int]:
    """Parse every downloaded PDF into structured JSON (M6, F6.4).

    Output layout mirrors ``parsed/<date>/<task_id>/<file>.json`` so the artefacts
    stay traceable to the task that produced them.

    Args:
        outcomes: ``(result, DownloadResult)`` pairs from the download stage.
        parsed_root: Root ``parsed/`` directory.
        task_id: Task id, used as the per-run sub-directory.

    Returns:
        ``(parsed_count, failed_count)``.
    """
    parse_pdf = _resolve_pdf_parser()
    if parse_pdf is None:
        return 0, 0

    target_dir = Path(parsed_root) / datetime.now(UTC).date().isoformat() / task_id
    parsed_count = failed_count = 0
    for _result, download in outcomes:
        path = download.local_path
        if not download.success or path is None or Path(path).suffix.lower() != ".pdf":
            continue
        try:
            document = parse_pdf(Path(path))
        except Exception as exc:  # noqa: BLE001 — 单个坏文件不影响其他文件
            failed_count += 1
            print(f"warning: cannot parse {path}: {exc}", file=sys.stderr)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{Path(path).stem}.json"
        out_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        parsed_count += 1
    return parsed_count, failed_count


# pyproject [project.scripts] points the ``retrieve`` command at this symbol.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
