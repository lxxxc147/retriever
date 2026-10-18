"""CLI entry point — ``retrieve "query"`` (argparse).

Phase 1 main-line wiring: parse intent (rules) → route via :class:`Router`
→ render results → optional ``--download`` of the results into
``--out/<task_id>/`` with a ``manifest.json`` summary (PRD §2.8).

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
from .intent import parse_intent
from .models import DownloadResult, Manifest, ManifestFile, SearchQuery, SearchResult
from .presenter import render_download_progress, render_results, render_summary
from .router import Router

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

    parsed = parse_intent(args.query)
    parsed.limit = max(1, args.limit)

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
