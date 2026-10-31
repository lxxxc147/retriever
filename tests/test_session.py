"""Tests for the interactive session state — multi-turn refine (F7.3/G4).

Everything here is pure logic over in-memory results; no network, no CLI.
"""

from datetime import UTC, datetime

from retriever.intent import ParsedIntent
from retriever.models import Filters, SearchQuery, SearchResult
from retriever.session import (
    SearchSession,
    apply_refine,
    merge_filters,
    merge_intent,
)


def _result(
    title: str = "paper",
    *,
    abstract: str = "",
    published: datetime | None = None,
    authors: list[str] | None = None,
    extra: dict | None = None,
) -> SearchResult:
    return SearchResult(
        source="arxiv",
        title=title,
        abstract=abstract,
        url="https://arxiv.org/abs/2401.00001",
        published_at=published,
        authors=authors or [],
        extra=extra or {},
    )


def _addition(**overrides: object) -> ParsedIntent:
    values: dict = {
        "intent": "paper",
        "keywords": [],
        "filters": Filters(),
        "raw_query": "refine",
    }
    values.update(overrides)
    return ParsedIntent(**values)  # type: ignore[arg-type]


def _query(**overrides: object) -> SearchQuery:
    values: dict = {
        "intent": "paper",
        "raw_query": "base query",
        "keywords": ["base"],
        "filters": Filters(),
        "limit": 5,
    }
    values.update(overrides)
    return SearchQuery(**values)  # type: ignore[arg-type]


# ------------------------------------------------------------------ merge rules
def test_merge_filters_addition_overrides_only_set_slots() -> None:
    """Set slots win; unset slots inherit the accumulated value."""
    base = Filters(year_from=2020, language="python")
    addition = Filters(year_from=2024, min_stars=100)
    merged = merge_filters(base, addition)
    assert merged.year_from == 2024  # overridden
    assert merged.language == "python"  # inherited
    assert merged.min_stars == 100  # added


def test_merge_intent_dedupes_keywords_and_takes_new_intent() -> None:
    """Keywords append without duplicates; the addition's intent becomes current."""
    query = _query(keywords=["llm", "agent"])
    addition = _addition(intent="repo", keywords=["agent", "rag"])
    merged = merge_intent(query, addition)
    assert merged.intent == "repo"
    assert merged.keywords == ["llm", "agent", "rag"]
    assert merged.raw_query == "refine"


# --------------------------------------------------------------- local filtering
def test_apply_refine_year_range_keeps_in_range_and_drops_unknown() -> None:
    """Year filters compare against published_at (extra.year fallback); results
    without any year are dropped (cannot verify ≈ no when narrowing down)."""
    results = [
        _result("old", published=datetime(2021, 3, 1, tzinfo=UTC)),
        _result("new", published=datetime(2024, 7, 1, tzinfo=UTC)),
        _result("unknown"),
    ]
    addition = _addition(filters=Filters(year_from=2024))
    kept = apply_refine(results, addition)
    assert [r.title for r in kept] == ["new"]


def test_apply_refine_year_falls_back_to_extra_year() -> None:
    """GitHub-style results carry ``extra["year"]`` instead of ``published_at``."""
    results = [
        _result("py repo", extra={"year": 2023}),
        _result("rs repo", extra={"year": 2024}),
    ]
    addition = _addition(filters=Filters(year_to=2023))
    assert [r.title for r in apply_refine(results, addition)] == ["py repo"]


def test_apply_refine_min_stars_uses_extra_stars() -> None:
    """Star thresholds read ``extra["stars"]``; results without stars are dropped."""
    results = [
        _result("popular", extra={"stars": 500}),
        _result("small", extra={"stars": 10}),
        _result("no-meta"),
    ]
    addition = _addition(filters=Filters(min_stars=100))
    assert [r.title for r in apply_refine(results, addition)] == ["popular"]


def test_apply_refine_language_is_case_insensitive() -> None:
    results = [
        _result("python tool", extra={"language": "Python"}),
        _result("rust tool", extra={"language": "rust"}),
        _result("no language"),
    ]
    addition = _addition(filters=Filters(language="PYTHON"))
    assert [r.title for r in apply_refine(results, addition)] == ["python tool"]


def test_apply_refine_authors_match_substring_case_insensitive() -> None:
    results = [
        _result("vaswani paper", authors=["Ashish Vaswani"]),
        _result("other", authors=["Yann LeCun"]),
        _result("no authors"),
    ]
    addition = _addition(filters=Filters(authors=["vaswani"]))
    assert [r.title for r in apply_refine(results, addition)] == ["vaswani paper"]


def test_apply_refine_categories_intersect() -> None:
    results = [
        _result("nlp", extra={"categories": ["cs.CL", "cs.LG"]}),
        _result("vision", extra={"categories": ["cs.CV"]}),
    ]
    addition = _addition(filters=Filters(categories=["cs.cl"]))
    assert [r.title for r in apply_refine(results, addition)] == ["nlp"]


def test_apply_refine_keywords_must_all_appear_in_title_or_abstract() -> None:
    results = [
        _result("Attention is all you need", abstract="transformer architecture"),
        _result("RNN survey", abstract="recurrent networks"),
    ]
    addition = _addition(keywords=["attention"])
    assert [r.title for r in apply_refine(results, addition)] == ["Attention is all you need"]


def test_apply_refine_without_constraints_keeps_everything() -> None:
    results = [_result("a"), _result("b")]
    assert apply_refine(results, _addition()) == results


# ------------------------------------------------------------------ the session
def test_session_refine_local_shrinks_results_and_accumulates() -> None:
    """A successful local refine updates results and merges slots into the query."""
    results = [
        _result("old", published=datetime(2021, 1, 1, tzinfo=UTC)),
        _result("new", published=datetime(2024, 1, 1, tzinfo=UTC)),
    ]
    session = SearchSession(results, _query())
    outcome = session.refine(_addition(filters=Filters(year_from=2024)))

    assert outcome.action == "local"
    assert outcome.results is not None
    assert [r.title for r in outcome.results] == ["new"]
    assert [r.title for r in session.results] == ["new"]  # state updated in place
    assert session.query.filters.year_from == 2024  # slot accumulated
    assert session.turns == 1


def test_session_pure_filter_utterance_keeps_the_base_intent() -> None:
    """"只要 2024 年的" has no topic signal — the rule parser's default "web"
    intent must not reroute the session; its filters still apply."""
    session = SearchSession([_result("a")], _query(intent="paper"))
    session.refine(_addition(intent="web", keywords=[], filters=Filters(year_from=2024)))
    assert session.query.intent == "paper"
    assert session.query.filters.year_from == 2024


def test_session_explicit_intent_change_still_overrides() -> None:
    """A follow-up with a real intent signal (keywords) does reroute the session."""
    session = SearchSession([_result("a")], _query(intent="paper"))
    session.refine(_addition(intent="repo", keywords=["github"]))
    assert session.query.intent == "repo"
    assert "github" in session.query.keywords


def test_session_refine_empty_falls_back_to_requery_with_merged_query() -> None:
    """When local filtering empties the list the caller gets the merged query."""
    results = [_result("old", published=datetime(2021, 1, 1, tzinfo=UTC))]
    session = SearchSession(results, _query(filters=Filters(min_stars=50)))
    outcome = session.refine(
        _addition(filters=Filters(year_from=2024), keywords=["attention"])
    )

    assert outcome.action == "requery"
    assert outcome.results is None
    assert outcome.query is not None
    # both the old and the new constraint must survive into the re-query
    assert outcome.query.filters.min_stars == 50
    assert outcome.query.filters.year_from == 2024
    assert "attention" in outcome.query.keywords
    # the session carries the same accumulated query for the next turn
    assert session.query.filters.year_from == 2024
    assert session.turns == 1
