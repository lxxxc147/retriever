"""M4 Result Pipeline — dedupe, rank, summarize (task X-2, PRD §2.4).

Sits between the router and the presenter and does three jobs:

1. **Deduplicate** (F4.1) — one canonical key per result: arXiv ID, GitHub
   ``owner/repo``, DOI, else the normalized URL. The same paper returned by both
   arXiv and Scholar therefore shows up once; duplicates are merged so no
   metadata is lost.
2. **Rank** (F4.2) — weighted score whose weights come from
   ``config/settings.yaml`` (``ranking.weights``): source trust, recency
   (exponential decay) and popularity (citations / stars, log-normalized).
   Sorting is stable, so equal scores keep the adapters' arrival order.
3. **Summarize** (F4.3) — one one-sentence summary per result. The summarizer is
   *injected* (see :mod:`retriever.summarizer`), so the pipeline stays free of
   LLM code and degrades to deterministic truncation when none is supplied.

Scoring is a plain weighted sum instead of a search-engine relevance model
(PRD §1.3 non-goal: "不做搜索引擎级别的相关性排序算法").
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

from .config import resolve_ranking_weights
from .models import SearchResult

# Trust per source (0..1). Unknown sources get DEFAULT_TRUST.
SOURCE_TRUST: dict[str, float] = {
    "arxiv": 1.0,
    "scholar": 0.95,
    "github": 0.9,
    "pdf": 0.7,
    "web": 0.6,
}
DEFAULT_TRUST = 0.5

# Recency: score halves every RECENCY_HALF_LIFE_YEARS.
RECENCY_HALF_LIFE_YEARS = 5.0
NEUTRAL_RECENCY = 0.5  # used when a result has no published_at

# Popularity: log1p(value) / log1p(POPULARITY_REFERENCE), capped at 1.0,
# so ~1000 citations or stars already saturates the dimension.
POPULARITY_REFERENCE = 1000.0

SUMMARY_MAX_CHARS = 60

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/([^/?#]+)/([^/?#]+)", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_URL_PARTS_RE = re.compile(r"^[a-z]+://([^/?#]+)([^?#]*)", re.IGNORECASE)


@dataclass(frozen=True)
class RankingWeights:
    """Weights of the three ranking dimensions (PRD §10.2)."""

    source: float = 0.4
    recency: float = 0.3
    popularity: float = 0.3

    @classmethod
    def from_mapping(cls, values: dict | None) -> RankingWeights:
        """Build weights from a ``{dimension: weight}`` mapping.

        Unknown keys are ignored and unparsable/negative values fall back to the
        defaults, so a bad config never breaks a search run (PRD NF4).

        Args:
            values: Mapping such as ``{"source": 0.5, "recency": 0.3}``.

        Returns:
            The resulting weights.
        """
        defaults = cls()
        resolved: dict[str, float] = {}
        for name in ("source", "recency", "popularity"):
            default = getattr(defaults, name)
            raw = (values or {}).get(name, default)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
            resolved[name] = value if value >= 0 else default
        return cls(**resolved)

    def as_dict(self) -> dict[str, float]:
        """Return the weights as a plain mapping (for logging and tests)."""
        return {"source": self.source, "recency": self.recency, "popularity": self.popularity}


# --------------------------------------------------------------------- dedupe
def canonical_key(result: SearchResult) -> str:
    """Return the identity used to deduplicate ``result`` across sources.

    Precedence: arXiv ID → DOI → GitHub ``owner/repo`` → normalized URL →
    normalized title. The arXiv version suffix (``v2``) is dropped so different
    versions of one paper collapse into a single entry.

    Args:
        result: A search result from any adapter.

    Returns:
        A namespaced key such as ``"arxiv:2401.00001"`` or ``"github:org/repo"``.
    """
    for url in (result.url, result.download_url or ""):
        match = _ARXIV_RE.search(url)
        if match:
            arxiv_id = match.group(1).strip().rstrip("/")
            arxiv_id = re.sub(r"\.pdf$", "", arxiv_id, flags=re.IGNORECASE)
            arxiv_id = _ARXIV_VERSION_RE.sub("", arxiv_id)
            return f"arxiv:{arxiv_id.lower()}"

    doi = str(result.extra.get("doi") or "").strip().lower()
    if doi:
        doi = doi.removeprefix("https://doi.org/").removeprefix("doi:")
        return f"doi:{doi}"

    for url in (result.url, result.download_url or ""):
        match = _GITHUB_RE.search(url)
        if match:
            owner, repo = match.group(1), match.group(2).removesuffix(".git")
            return f"github:{owner.lower()}/{repo.lower()}"

    normalized = _normalize_url(result.url)
    if normalized:
        return f"url:{normalized}"

    title = " ".join(result.title.lower().split())
    return f"title:{title}"


def _normalize_url(url: str) -> str:
    """Lowercase host, drop scheme/``www.``/query/fragment/trailing slash."""
    match = _URL_PARTS_RE.match(url.strip())
    if not match:
        return ""
    host = match.group(1).lower().removeprefix("www.")
    path = match.group(2).rstrip("/").lower()
    return f"{host}{path}"


def dedupe(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Drop cross-source duplicates, merging their metadata into the first hit.

    Args:
        results: Results in arrival order.

    Returns:
        Unique results in first-seen order; for each duplicate group the first
        result is kept and missing metadata (abstract, authors, date, ``extra``
        keys) is filled in from the later copies.
    """
    unique: dict[str, SearchResult] = {}
    order: list[str] = []
    for result in results:
        key = canonical_key(result)
        if key not in unique:
            unique[key] = result
            order.append(key)
            continue
        unique[key] = _merge(unique[key], result)
    return [unique[key] for key in order]


def _merge(primary: SearchResult, duplicate: SearchResult) -> SearchResult:
    """Return ``primary`` enriched with whatever ``duplicate`` knows."""
    extra = dict(duplicate.extra)
    extra.update({k: v for k, v in primary.extra.items() if v not in (None, "", [])})
    return primary.model_copy(
        update={
            "abstract": primary.abstract or duplicate.abstract,
            "authors": primary.authors or duplicate.authors,
            "published_at": primary.published_at or duplicate.published_at,
            "download_url": primary.download_url or duplicate.download_url,
            "extra": extra,
        }
    )


# --------------------------------------------------------------------- ranking
def score_result(
    result: SearchResult,
    weights: RankingWeights | None = None,
    now: datetime | None = None,
) -> float:
    """Score one result as a weighted sum of source trust, recency, popularity.

    Args:
        result: The result to score.
        weights: Ranking weights; ``None`` reads ``ranking.weights`` from config.
        now: Reference time for the recency decay (defaults to the current UTC
            time; injected in tests for determinism).

    Returns:
        A score in ``[0, 1]`` — higher is better.
    """
    resolved = weights or RankingWeights.from_mapping(resolve_ranking_weights())
    moment = now or datetime.now(UTC)
    return (
        resolved.source * _source_score(result.source)
        + resolved.recency * _recency_score(result.published_at, moment)
        + resolved.popularity * _popularity_score(result.extra)
    )


def _source_score(source: str) -> float:
    """Trust of a data source (unknown sources fall back to the default)."""
    return SOURCE_TRUST.get(str(source).lower(), DEFAULT_TRUST)


def _recency_score(published_at: datetime | None, now: datetime) -> float:
    """Exponential decay: 1.0 today, 0.5 after the half-life, floor at 0."""
    if published_at is None:
        return NEUTRAL_RECENCY
    published = published_at if published_at.tzinfo else published_at.replace(tzinfo=UTC)
    age_years = max(0.0, (now - published).total_seconds() / (365.25 * 24 * 3600))
    return 0.5 ** (age_years / RECENCY_HALF_LIFE_YEARS)


def _popularity_score(extra: dict) -> float:
    """Log-normalized citations/stars, capped at 1.0."""
    value = extra.get("citations")
    if value in (None, 0, ""):
        value = extra.get("stars", 0)
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0:
        return 0.0
    return min(1.0, math.log1p(number) / math.log1p(POPULARITY_REFERENCE))


def rank(
    results: Sequence[SearchResult],
    weights: RankingWeights | None = None,
    now: datetime | None = None,
) -> list[SearchResult]:
    """Sort results by descending score, keeping arrival order on ties.

    Args:
        results: Results to sort (typically already deduplicated).
        weights: Ranking weights; ``None`` reads ``ranking.weights`` from config.
        now: Reference time for the recency decay.

    Returns:
        A new list, best result first.
    """
    resolved = weights or RankingWeights.from_mapping(resolve_ranking_weights())
    moment = now or datetime.now(UTC)
    decorated = [
        (-score_result(result, resolved, moment), index, result)
        for index, result in enumerate(results)
    ]
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


# ------------------------------------------------------------------- summaries
def fallback_summary(result: SearchResult, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Build a deterministic one-line summary without any model call.

    Uses the first sentence of the abstract when available, else the title, and
    truncates to ``max_chars``.

    Args:
        result: The result to summarize.
        max_chars: Maximum length before the ellipsis.

    Returns:
        A single-line summary (empty only when the result carries neither an
        abstract nor a title).
    """
    text = ""
    for candidate in (result.abstract, result.title):
        text = " ".join((candidate or "").split())
        if text:
            break
    if not text:
        return ""
    for delimiter in ("。", ". ", "；", "; "):
        head, sep, _ = text.partition(delimiter)
        if sep and head:
            text = head + (delimiter if delimiter == "。" else delimiter.strip())
            break
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def summarize(
    results: Sequence[SearchResult],
    summarizer: Callable[[list[SearchResult]], list[str]] | None = None,
) -> list[SearchResult]:
    """Attach a one-sentence summary to every result (``extra["summary"]``).

    Args:
        results: Results to summarize.
        summarizer: Callable returning one summary per result, in order. ``None``
            (or a callable that fails / returns the wrong length) falls back to
            :func:`fallback_summary`, so a dead model never blocks a run.

    Returns:
        Copies of the results carrying ``extra["summary"]``.
    """
    items = list(results)
    if not items:
        return []
    summaries: list[str] = []
    if summarizer is not None:
        try:
            produced = list(summarizer(items))
        except Exception as exc:  # noqa: BLE001 — the extractive fallback is the safety net
            logger.warning("summarizer failed, using extractive summaries: {}", exc)
            produced = []
        if len(produced) == len(items):
            summaries = [str(text).strip() for text in produced]
    if not summaries:
        summaries = [fallback_summary(item) for item in items]
    return [
        item.model_copy(update={"extra": {**item.extra, "summary": summary}})
        for item, summary in zip(items, summaries, strict=True)
    ]


# -------------------------------------------------------------------- pipeline
def run_pipeline(
    results: Iterable[SearchResult],
    *,
    weights: RankingWeights | None = None,
    limit: int | None = None,
    summarizer: Callable[[list[SearchResult]], list[str]] | None = None,
    now: datetime | None = None,
) -> list[SearchResult]:
    """Run the full M4 pipeline: dedupe → rank → (cap) → summarize.

    Args:
        results: Raw results aggregated from the adapters.
        weights: Ranking weights; ``None`` reads ``ranking.weights`` from config.
        limit: Maximum number of results to keep after ranking; ``None`` keeps all.
        summarizer: Optional summary producer (see :mod:`retriever.summarizer`).
        now: Reference time for the recency decay (tests inject it).

    Returns:
        Deduplicated, ranked (and optionally summarized) results.
    """
    unique = dedupe(results)
    ranked = rank(unique, weights=weights, now=now)
    if limit is not None and limit >= 0:
        ranked = ranked[:limit]
    if summarizer is None:
        return ranked
    return summarize(ranked, summarizer)
