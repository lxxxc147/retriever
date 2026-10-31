"""Interactive session state — multi-turn refine (PRD §2.7 F7.3, goal G4).

One :class:`SearchSession` holds the current result list and the accumulated
:class:`~retriever.models.SearchQuery`. Each follow-up utterance is parsed the
same way as the opening query; the resulting *addition* is first applied as a
local filter over the current results. When nothing survives (e.g. "只看
2024 年以后的" over a list whose results all predate 2024), the addition is
merged into the accumulated query and the caller re-runs the router — a
refine must never silently return an empty page.

Local filtering is deliberately metadata-driven and conservative: a result
that lacks the metadata a filter asks for (no year, no stars, no language) is
dropped rather than kept, because "cannot verify" is closer to "no" than to
"yes" when the user is narrowing down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger

from .intent import ParsedIntent
from .models import Filters, SearchQuery, SearchResult

RefineAction = Literal["local", "requery"]


@dataclass(frozen=True)
class RefineOutcome:
    """What a refine turn decided to do.

    Attributes:
        action: ``"local"`` — ``results`` holds the filtered list and is
            authoritative; ``"requery"`` — local filtering emptied the list and
            ``query`` is the merged query to send through the router again.
        results: Filtered results (only for ``action == "local"``).
        query: Merged accumulated query (only for ``action == "requery"``).
    """

    action: RefineAction
    results: list[SearchResult] | None = None
    query: SearchQuery | None = None


class SearchSession:
    """State of one interactive conversation (F8.2).

    Attributes:
        results: Current result list (shrinks as local refines succeed).
        query: Accumulated query; every accepted refinement merges its slots
            into this query, so a later re-query keeps every constraint the
            user has asked for so far.
        turns: Number of refine turns processed so far.
    """

    def __init__(self, results: list[SearchResult], query: SearchQuery) -> None:
        self.results = list(results)
        self.query = query
        self.turns = 0

    def refine(self, addition: ParsedIntent) -> RefineOutcome:
        """Apply one follow-up utterance to the current state.

        Args:
            addition: The parsed follow-up (intent, keywords, filter slots).

        Returns:
            A :class:`RefineOutcome`; the session is updated in place so that
            ``self.results`` / ``self.query`` always describe the current page.
        """
        self.turns += 1
        merged = merge_intent(self.query, addition)
        if addition.intent == "web" and not addition.keywords:
            # A pure-filter utterance ("只要 2024 年的") carries no topic signal:
            # the rule parser falls back to its default "web" intent, which must
            # not reroute the session away from the adapters that produced the
            # current results. Utterances with real intent signals (keywords,
            # source URLs) still override via ``merge_intent`` as usual.
            merged = merged.model_copy(update={"intent": self.query.intent})
        self.query = merged
        filtered = apply_refine(self.results, addition)
        if filtered:
            self.results = filtered
            logger.debug("refine turn {}: local filter kept {} result(s)", self.turns, len(filtered))
            return RefineOutcome(action="local", results=filtered)
        logger.debug("refine turn {}: local filter empty — merged query requeued", self.turns)
        return RefineOutcome(action="requery", query=self.query)


def merge_filters(base: Filters, addition: Filters) -> Filters:
    """Merge two filter objects; every slot the addition sets overrides base.

    Args:
        base: The accumulated filters.
        addition: Filters parsed from the follow-up utterance.

    Returns:
        A new :class:`Filters` where each ``addition`` slot that is not
        ``None`` wins, and unset slots inherit from ``base``.
    """
    return Filters(
        authors=addition.authors if addition.authors is not None else base.authors,
        year_from=addition.year_from if addition.year_from is not None else base.year_from,
        year_to=addition.year_to if addition.year_to is not None else base.year_to,
        language=addition.language if addition.language is not None else base.language,
        min_stars=addition.min_stars if addition.min_stars is not None else base.min_stars,
        categories=addition.categories if addition.categories is not None else base.categories,
    )


def merge_intent(query: SearchQuery, addition: ParsedIntent) -> SearchQuery:
    """Merge a parsed follow-up into the accumulated query.

    Filter slots set by the addition override; its keywords are appended
    (order-preserving dedupe); its intent becomes the current intent.

    Args:
        query: The accumulated query.
        addition: The parsed follow-up utterance.

    Returns:
        A copy of ``query`` with the addition merged in.
    """
    keywords = list(query.keywords)
    for keyword in addition.keywords:
        if keyword not in keywords:
            keywords.append(keyword)
    return query.model_copy(
        update={
            "intent": addition.intent,
            "raw_query": addition.raw_query,
            "keywords": keywords,
            "filters": merge_filters(query.filters, addition.filters),
        }
    )


# ----------------------------------------------------------------- local filter
def apply_refine(results: list[SearchResult], addition: ParsedIntent) -> list[SearchResult]:
    """Filter ``results`` locally with whatever the addition specifies.

    Every constraint the addition carries must pass, and results missing the
    metadata a constraint needs are dropped (see the module docstring).

    Args:
        results: Current results.
        addition: Parsed follow-up utterance.

    Returns:
        The surviving results in their current order (possibly empty — the
        caller then decides whether to re-query the sources).
    """
    kept: list[SearchResult] = []
    for result in results:
        if not _passes_filters(result, addition.filters):
            continue
        if not _passes_keywords(result, addition.keywords):
            continue
        kept.append(result)
    return kept


def _result_year(result: SearchResult) -> int | None:
    """Publication year from ``published_at``, falling back to ``extra.year``."""
    if result.published_at is not None:
        return result.published_at.year
    raw = result.extra.get("year")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _passes_filters(result: SearchResult, filters: Filters) -> bool:
    """Check every filter slot of ``filters`` against ``result``'s metadata."""
    year = _result_year(result)
    if filters.year_from is not None and (year is None or year < filters.year_from):
        return False
    if filters.year_to is not None and (year is None or year > filters.year_to):
        return False

    if filters.min_stars is not None:
        try:
            stars = float(result.extra.get("stars") or 0)
        except (TypeError, ValueError):
            return False
        if stars < filters.min_stars:
            return False

    if filters.language is not None:
        language = str(result.extra.get("language") or "").strip().lower()
        if language != filters.language.strip().lower():
            return False

    if filters.authors:
        authors = [name.casefold() for name in result.authors]
        wanted = [name.casefold() for name in filters.authors]
        if not any(needle in hay for needle in wanted for hay in authors):
            return False

    if filters.categories:
        cats = [str(cat).casefold() for cat in result.extra.get("categories") or []]
        wanted = [str(cat).casefold() for cat in filters.categories]
        if not any(needle in cats for needle in wanted):
            return False

    return True


def _passes_keywords(result: SearchResult, keywords: list[str]) -> bool:
    """Require every keyword to appear (case-insensitively) in title/abstract."""
    haystack = f"{result.title}\n{result.abstract}".casefold()
    return all(keyword.casefold() in haystack for keyword in keywords)
