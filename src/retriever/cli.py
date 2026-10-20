"""CLI entry point — ``retrieve "query"`` (argparse).

Phase 1 main-line wiring: parse intent → route via :class:`Router` → render
results → optional ``--download`` of the results into ``--out/<task_id>/`` with
a ``manifest.json`` summary (PRD §2.8).

The intent stage is pluggable (``--parser``): ``auto`` uses the LLM when an API
key is configured and the deterministic rule parser otherwise, ``rule`` forces
the rules, ``llm`` forces the model. Either way a parse failure falls back to the
rules, so the CLI never dies because a model endpoint is unavailable.

The console-script entry point is ``app`` (pyproject ``[project.scripts]``
points at ``retriever.cli:app``); :func:`main` accepts an explicit argv list
so tests can drive the full flow without subprocesses or network access.
Downloads go straight through ``adapter.download`` — the DownloadManager
(task M5) is a separate workstream and is intentionally not used here.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .intent_llm import parse_intent_auto
from .llm import LLMClient
from .models import DownloadResult, Manifest, ManifestFile, SearchQuery, SearchResult
from .pipeline import run_pipeline
from .presenter import render_download_progress, render_intent, render_results, render_summary
from .router import Router
from .summarizer import LLMSummarizer

# Windows consoles default to GBK; force UTF-8 so CJK/emoji output works.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _run_sync(value: Any) -> Any:
    """Return ``value``, awaiting it first when it is a coroutine.

    Adapter ``download`` methods are ``async``; running them with
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
    parser.add_argument("query", help="Natural-language search query (a source URL also works).")
    parser.add_argument(
        "--source",
        choices=["arxiv", "github", "all"],
        default="all",
        help="Restrict the search to one data source (default: all).",
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
            parser.error(f"no adapter registered for --source {args.source!r}")

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

    if not args.download:
        return 0
    if not results:
        print("Nothing to download (no results).")
        return 0

    task_id = uuid.uuid4().hex[:12]
    dest_dir = args.out / task_id
    outcomes: list[tuple[SearchResult, DownloadResult]] = []
    total = len(results)
    for index, result in enumerate(results, start=1):
        adapter = next((a for a in adapters if a.name == result.source), None)
        if adapter is None:
            outcomes.append(
                (
                    result,
                    DownloadResult(success=False, error=f"no adapter for source '{result.source}'"),
                )
            )
            continue
        render_download_progress(index, total, result)
        outcomes.append((result, _run_sync(adapter.download(result, dest_dir))))

    render_summary(outcomes)

    manifest = Manifest(
        task_id=task_id,
        query=args.query,
        created_at=datetime.now(UTC),
        files=[
            ManifestFile(
                title=result.title,
                source=result.source,
                url=result.url,
                local_path=download.local_path,
                size_bytes=download.size_bytes,
                sha256=download.sha256 or "",
            )
            for result, download in outcomes
            if download.success and download.local_path is not None
        ],
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return 0


# pyproject [project.scripts] points the ``retrieve`` command at this symbol.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
