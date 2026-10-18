"""M2 Router (task X-3, PRD §2.2).

Maps a parsed intent onto the registered data-source adapters and fans
search calls out to them:

- :meth:`Router.route` — intent → adapter mapping (paper→arXiv,
  repo/code_file→GitHub, dataset/web→both).
- :meth:`Router.search_all` — parallel search via a thread pool; a failing
  adapter only produces a stderr warning, never aborts the other sources.

Note: ``adapters/__init__.py`` does not re-export ``ArxivAdapter`` yet, so
it is imported from its own module here.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .adapters.arxiv import ArxivAdapter
from .adapters.base import SearchAdapter
from .adapters.github import GithubAdapter
from .intent import ParsedIntent
from .models import Intent, SearchQuery, SearchResult

# intent -> adapter names consulted, used for routing.
_INTENT_SOURCES: dict[Intent, tuple[str, ...]] = {
    "paper": ("arxiv",),
    "repo": ("github",),
    "code_file": ("github",),
    "dataset": ("arxiv", "github"),
    "web": ("arxiv", "github"),
}

# Canonical group order for aggregated results (arXiv first, GitHub after).
_CANONICAL_SOURCE_ORDER = ("arxiv", "github")

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

    def __init__(self, adapters: list[SearchAdapter] | None = None) -> None:
        """Register adapters; defaults to ArxivAdapter + GithubAdapter.

        Args:
            adapters: Adapters to manage. ``None`` registers the two
                production adapters (arXiv + GitHub).
        """
        self._adapters: list[SearchAdapter] = (
            list(adapters) if adapters is not None else [ArxivAdapter(), GithubAdapter()]
        )

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
        """Adapters whose name is registered for ``intent``."""
        wanted = _INTENT_SOURCES.get(intent, _INTENT_SOURCES["web"])
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
