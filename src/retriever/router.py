"""M2 Router (task X-3, PRD §2.2).

Maps a parsed intent onto the registered data-source adapters and fans
search calls out to them:

- :meth:`Router.route` — intent → adapter mapping. The mapping comes from
  ``config/sources.yaml`` (each source may declare ``intents``); sources that do
  not declare them fall back to the built-in table below, so behaviour is
  unchanged for arXiv/GitHub and a new source needs no code change (PRD F2.3).
- :meth:`Router.search_all` — parallel search via a thread pool; a failing
  adapter only produces a stderr warning, never aborts the other sources.

Adapters come from the registry (``retriever.sources``); when the registry is
missing or unusable the router registers the built-in arXiv + GitHub pair, so
the CLI keeps working.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from .adapters.arxiv import ArxivAdapter
from .adapters.base import SearchAdapter
from .adapters.github import GithubAdapter
from .intent import ParsedIntent
from .models import Intent, SearchQuery, SearchResult
from .sources import SourceRegistryError, load_adapters

# Fallback intent -> adapter names, used for sources that do not declare
# `intents` in config/sources.yaml.
_INTENT_SOURCES: dict[Intent, tuple[str, ...]] = {
    "paper": ("arxiv",),
    "repo": ("github",),
    "code_file": ("github",),
    "dataset": ("arxiv", "github"),
    "web": ("arxiv", "github"),
}

# Canonical group order for aggregated results (papers first, then code, then web).
_CANONICAL_SOURCE_ORDER = ("arxiv", "scholar", "github", "pdf", "web")

# Clean arXiv identifier: new style with optional version, or old-style category ID.
_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}(?:v\d+)?|[a-z.-]+/\d{7})$", re.IGNORECASE)


def _run_sync(value: Any) -> Any:
    """Return ``value``, awaiting it first when it is a coroutine.

    Adapters expose ``async`` methods; synchronous test stubs do not. Running
    each coroutine with :func:`asyncio.run` inside a thread-pool worker keeps
    :meth:`Router.search_all` a plain blocking call for CLI consumers.
    """
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


class Router:
    """Select adapters by intent and aggregate their results."""

    def __init__(
        self,
        adapters: list[SearchAdapter] | None = None,
        sources_path: Path | None = None,
    ) -> None:
        """Register adapters from the source registry, or from an explicit list.

        Args:
            adapters: Explicit adapters (mainly tests). ``None`` loads
                ``config/sources.yaml`` and falls back to the built-in
                arXiv + GitHub pair when the registry is missing or unusable.
            sources_path: Override for the registry file (tests, alternate
                profiles such as a "papers only" setup).
        """
        if adapters is not None:
            self._adapters: list[SearchAdapter] = list(adapters)
            self._declared_intents: dict[str, list[Intent]] = {}
            return

        try:
            loaded, declared = load_adapters(sources_path)
        except SourceRegistryError as exc:
            logger.warning("source registry unusable ({}), using built-in defaults", exc)
            loaded, declared = [], {}
        self._adapters = loaded or [ArxivAdapter(), GithubAdapter()]
        self._declared_intents = declared

    @property
    def adapters(self) -> list[SearchAdapter]:
        """Registered adapters (defensive copy)."""
        return list(self._adapters)

    def route(self, parsed: ParsedIntent) -> list[SearchAdapter]:
        """Map a parsed intent onto the registered adapters.

        Args:
            parsed: Output of :func:`retriever.intent.parse_intent`.

        Returns:
            Adapters registered for ``parsed.intent``, in registration order.
        """
        return self._select(parsed.intent)

    def search_all(
        self,
        query: SearchQuery,
        adapters: list[SearchAdapter] | None = None,
    ) -> list[SearchResult]:
        """Search the selected adapters in parallel and aggregate the results.

        A single failing adapter is reported to stderr and contributes no
        results; the remaining sources still answer. When ``query`` carries a
        bare arXiv ID as keyword (parsed from an arXiv URL), adapters exposing
        ``get_by_id`` resolve it directly instead of running a keyword search.

        Args:
            query: Structured search task.
            adapters: Explicit adapter subset to query; ``None`` routes by
                ``query.intent`` via :meth:`route`.

        Returns:
            Aggregated results grouped by source — arXiv first, GitHub after,
            unknown sources last in registration order.
        """
        selected = self._select(query.intent) if adapters is None else list(adapters)
        if not selected:
            return []

        grouped: dict[str, list[SearchResult]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
            futures = {
                pool.submit(self._safe_search, adapter, query): adapter for adapter in selected
            }
            for future in as_completed(futures):
                adapter = futures[future]
                grouped[adapter.name] = future.result()

        ordered: list[SearchResult] = []
        for name in _CANONICAL_SOURCE_ORDER:
            ordered.extend(grouped.get(name, []))
        for adapter in selected:
            if adapter.name not in _CANONICAL_SOURCE_ORDER:
                ordered.extend(grouped.get(adapter.name, []))
        return ordered

    # ------------------------------------------------------------------ helpers
    def _select(self, intent: Intent) -> list[SearchAdapter]:
        """Adapters answering ``intent`` — registry table first, built-ins after.

        A registry entry with ``intents`` wins over the built-in table, so
        ``config/sources.yaml`` alone can reroute an intent to a new source.
        """
        wanted: list[str] = [
            name for name, intents in self._declared_intents.items() if intent in intents
        ]
        if not wanted:
            wanted = list(_INTENT_SOURCES.get(intent, _INTENT_SOURCES["web"]))
        return [adapter for adapter in self._adapters if adapter.name in wanted]

    def _safe_search(self, adapter: SearchAdapter, query: SearchQuery) -> list[SearchResult]:
        """Search one adapter, converting any failure into a stderr warning."""
        try:
            direct = self._direct_lookup(adapter, query)
            if direct is not None:
                return direct
            results = _run_sync(adapter.search(query))
        except Exception as exc:  # noqa: BLE001 — one bad source must never kill the others
            print(f"warning: adapter '{adapter.name}' failed: {exc}", file=sys.stderr)
            return []
        return list(results or [])

    @staticmethod
    def _direct_lookup(adapter: SearchAdapter, query: SearchQuery) -> list[SearchResult] | None:
        """Resolve a bare arXiv ID keyword via ``adapter.get_by_id`` when available.

        Returns:
            ``None`` when the fast path does not apply (intent is not
            ``paper``, the adapter has no ``get_by_id``, or no keyword is a
            clean arXiv ID) — the caller then falls back to keyword search.
            An empty list means the fast path applied but the ID was not found.
        """
        if query.intent != "paper":
            return None
        get_by_id = getattr(adapter, "get_by_id", None)
        if not callable(get_by_id):
            return None
        for keyword in query.keywords:
            if _ARXIV_ID_RE.match(keyword):
                result = _run_sync(get_by_id(keyword))
                return [result] if result is not None else []
        return None
