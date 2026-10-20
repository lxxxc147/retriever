"""Tests for the M4 result pipeline (dedupe / rank / summarize). No network."""

from datetime import UTC, datetime

import pytest

from retriever.models import SearchResult
from retriever.pipeline import (
    NEUTRAL_RECENCY,
    POPULARITY_REFERENCE,
    RankingWeights,
    canonical_key,
    dedupe,
    fallback_summary,
    rank,
    run_pipeline,
    score_result,
    summarize,
)

NOW = datetime(2026, 10, 19, tzinfo=UTC)


def _result(**overrides: object) -> SearchResult:
    """Build a result with sensible defaults for pipeline tests."""
    data: dict = {
        "source": "arxiv",
        "title": "A Paper",
        "abstract": "We study things.",
        "url": "https://arxiv.org/abs/2401.00001",
    }
    data.update(overrides)
    return SearchResult(**data)  # type: ignore[arg-type]


# ------------------------------------------------------------- canonical key
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2401.00001", "arxiv:2401.00001"),
        ("https://arxiv.org/pdf/2401.00001v3.pdf", "arxiv:2401.00001"),
        ("https://github.com/PyTorch/PyTorch", "github:pytorch/pytorch"),
        ("https://github.com/pytorch/pytorch.git", "github:pytorch/pytorch"),
        ("https://Example.org/Path/?q=1#top", "url:example.org/path"),
        ("https://example.org/Path", "url:example.org/path"),
    ],
)
def test_canonical_key_is_source_aware(url: str, expected: str) -> None:
    """arXiv IDs, GitHub repos and plain URLs normalize to stable identities."""
    assert canonical_key(_result(url=url)) == expected


def test_canonical_key_prefers_doi_when_url_is_opaque() -> None:
    """A DOI in ``extra`` identifies the paper even behind a redirect URL."""
    result = _result(url="https://doi.org/10.1000/XYZ", extra={"doi": "10.1000/xyz"})
    assert canonical_key(result) == "doi:10.1000/xyz"


def test_canonical_key_falls_back_to_title() -> None:
    """A result without any usable URL is deduplicated by its normalized title."""
    assert canonical_key(_result(url="", title="  A  Paper ")) == "title:a paper"


# -------------------------------------------------------------------- dedupe
def test_dedupe_merges_the_same_paper_across_sources() -> None:
    """The same arXiv paper found twice collapses, filling missing metadata."""
    first = _result(source="arxiv", abstract="", authors=[])
    second = _result(
        source="scholar",
        url="https://arxiv.org/abs/2401.00001v2",
        abstract="A survey of things.",
        authors=["Ada"],
        extra={"citations": 42},
    )

    unique = dedupe([first, second])

    assert len(unique) == 1
    assert unique[0].source == "arxiv"  # first hit wins
    assert unique[0].abstract == "A survey of things."  # enriched from the duplicate
    assert unique[0].authors == ["Ada"]
    assert unique[0].extra["citations"] == 42


def test_dedupe_keeps_distinct_results_in_arrival_order() -> None:
    """Different papers are all kept, in the order the adapters returned them."""
    results = [
        _result(url="https://arxiv.org/abs/2401.00001"),
        _result(source="github", url="https://github.com/org/repo", title="org/repo"),
        _result(url="https://arxiv.org/abs/2401.00002"),
    ]
    assert [item.url for item in dedupe(results)] == [item.url for item in results]


def test_dedupe_on_empty_input() -> None:
    """An empty result set stays empty."""
    assert dedupe([]) == []


# -------------------------------------------------------------------- ranking
def test_score_is_a_weighted_sum_within_unit_interval() -> None:
    """With the default weights (sum 1.0) the score stays inside [0, 1]."""
    score = score_result(
        _result(published_at=NOW, extra={"citations": 500}),
        RankingWeights(),
        NOW,
    )
    assert 0.0 <= score <= 1.0


def test_newer_result_outranks_older_one() -> None:
    """Recency decays with the half-life, so a fresh paper wins."""
    fresh = _result(url="https://arxiv.org/abs/2401.00001", published_at=NOW)
    stale = _result(
        url="https://arxiv.org/abs/2401.00002",
        published_at=datetime(2016, 10, 19, tzinfo=UTC),
    )
    assert rank([stale, fresh], now=NOW)[0] is fresh


def test_more_citations_outrank_fewer_when_other_dimensions_tie() -> None:
    """Popularity decides between results that are otherwise identical."""
    popular = _result(url="https://arxiv.org/abs/2401.00001", published_at=NOW,
                      extra={"citations": 800})
    quiet = _result(url="https://arxiv.org/abs/2401.00002", published_at=NOW,
                    extra={"citations": 2})
    assert rank([quiet, popular], now=NOW)[0] is popular


def test_stars_are_used_when_citations_are_absent() -> None:
    """GitHub results expose ``stars`` instead of ``citations``."""
    rich = _result(source="github", url="https://github.com/a/rich",
                   published_at=NOW, extra={"stars": 900})
    poor = _result(source="github", url="https://github.com/a/poor",
                   published_at=NOW, extra={"stars": 3})
    assert rank([poor, rich], now=NOW)[0] is rich


def test_source_trust_orders_equal_results() -> None:
    """With identical metadata, the more trusted source sorts first."""
    arxiv = _result(source="arxiv", url="https://arxiv.org/abs/2401.00001",
                    published_at=NOW)
    web = _result(source="web", url="https://example.org/a", published_at=NOW)
    assert rank([web, arxiv], now=NOW)[0] is arxiv


def test_weights_are_configurable() -> None:
    """Switching the weights to popularity-only flips the order."""
    old_famous = _result(
        url="https://arxiv.org/abs/2401.00001",
        published_at=datetime(2005, 1, 1, tzinfo=UTC),
        extra={"citations": 5000},
    )
    fresh_obscure = _result(
        url="https://arxiv.org/abs/2401.00002",
        published_at=NOW,
        extra={"citations": 0},
    )

    recency_only = RankingWeights(source=0.0, recency=1.0, popularity=0.0)
    popularity_only = RankingWeights(source=0.0, recency=0.0, popularity=1.0)

    assert rank([old_famous, fresh_obscure], weights=recency_only, now=NOW)[0] is fresh_obscure
    assert rank([old_famous, fresh_obscure], weights=popularity_only, now=NOW)[0] is old_famous


def test_equal_scores_keep_arrival_order() -> None:
    """Sorting is stable, so a tie never shuffles the adapters' order."""
    first = _result(url="https://arxiv.org/abs/2401.00001", published_at=NOW)
    second = _result(url="https://arxiv.org/abs/2401.00002", published_at=NOW)
    assert rank([first, second], now=NOW) == [first, second]


def test_missing_published_at_gets_the_neutral_recency() -> None:
    """A result without a date is neither rewarded nor punished."""
    assert (
        score_result(_result(source="", extra={}), RankingWeights(1.0, 0.0, 0.0), NOW)
        == 0.5
    )
    assert NEUTRAL_RECENCY == 0.5


def test_popularity_is_log_scaled_and_capped() -> None:
    """The popularity dimension saturates at the reference value and never exceeds 1."""
    weights = RankingWeights(source=0.0, recency=0.0, popularity=1.0)
    reference = score_result(
        _result(extra={"citations": int(POPULARITY_REFERENCE)}), weights, NOW
    )
    huge = score_result(_result(extra={"citations": 10**9}), weights, NOW)
    small = score_result(_result(extra={"citations": 1}), weights, NOW)

    assert reference == pytest.approx(1.0)
    assert huge == pytest.approx(1.0)
    assert 0.0 < small < reference


def test_unparsable_popularity_is_ignored() -> None:
    """A malformed ``stars`` value scores 0 instead of raising."""
    weights = RankingWeights(source=0.0, recency=0.0, popularity=1.0)
    assert score_result(_result(extra={"stars": "many"}), weights, NOW) == 0.0


def test_weights_from_mapping_ignores_bad_input() -> None:
    """Unknown keys, junk values and negatives fall back to the defaults."""
    weights = RankingWeights.from_mapping(
        {"source": 0.6, "recency": "x", "popularity": -1, "bogus": 5}
    )
    assert weights.source == 0.6
    assert weights.recency == RankingWeights().recency
    assert weights.popularity == RankingWeights().popularity


# ------------------------------------------------------------------ summaries
def test_fallback_summary_uses_the_first_sentence_and_truncates() -> None:
    """Extractive summaries stop at the first sentence and add an ellipsis."""
    assert fallback_summary(_result(abstract="一句话。第二句不应该出现。")) == "一句话。"
    long = fallback_summary(_result(abstract="x" * 200))
    assert len(long) == 60
    assert long.endswith("…")


def test_fallback_summary_falls_back_to_the_title() -> None:
    """Without an abstract the title is summarized."""
    assert fallback_summary(_result(abstract="", title="Attention Is All You Need")) == (
        "Attention Is All You Need"
    )


def test_summarize_uses_the_injected_summarizer() -> None:
    """Provided summaries land in ``extra["summary"]`` without touching the input."""
    results = [
        _result(url="https://arxiv.org/abs/2401.00001"),
        _result(url="https://arxiv.org/abs/2401.00002", title="B Paper"),
    ]
    summarized = summarize(results, lambda items: [f"s{index}" for index in range(len(items))])

    assert [item.extra["summary"] for item in summarized] == ["s0", "s1"]
    assert "summary" not in results[0].extra  # inputs untouched
    assert summarized[0].title == results[0].title


def test_summarize_falls_back_when_the_summarizer_raises() -> None:
    """A broken summarizer degrades to extractive summaries."""

    def boom(items: list[SearchResult]) -> list[str]:
        raise RuntimeError("model down")

    summarized = summarize([_result(abstract="一句话。")], boom)
    assert summarized[0].extra["summary"] == "一句话。"


def test_summarize_falls_back_on_a_length_mismatch() -> None:
    """A summarizer returning the wrong count is ignored entirely."""
    summarized = summarize([_result(), _result(url="https://arxiv.org/abs/2401.00002")],
                           lambda items: ["only one"])
    assert len(summarized) == 2
    assert all(item.extra["summary"] for item in summarized)


def test_summarize_on_empty_input() -> None:
    """Summarizing nothing returns nothing."""
    assert summarize([]) == []


# ------------------------------------------------------------------ pipeline
def test_run_pipeline_dedupes_ranks_and_caps() -> None:
    """The combined pipeline removes duplicates, ranks, then applies the limit."""
    results = [
        _result(url="https://arxiv.org/abs/2401.00001",
                published_at=datetime(2015, 1, 1, tzinfo=UTC)),
        _result(source="scholar", url="https://arxiv.org/abs/2401.00001v2",
                published_at=datetime(2015, 1, 1, tzinfo=UTC)),
        _result(url="https://arxiv.org/abs/2401.00002", published_at=NOW),
        _result(url="https://arxiv.org/abs/2401.00003", published_at=NOW),
    ]

    ranked = run_pipeline(results, limit=2, now=NOW)

    assert len(ranked) == 2
    assert {item.url for item in ranked} == {
        "https://arxiv.org/abs/2401.00002",
        "https://arxiv.org/abs/2401.00003",
    }


def test_run_pipeline_without_summarizer_adds_no_summary() -> None:
    """Summaries are opt-in: the pipeline stays free of model calls by default."""
    ranked = run_pipeline([_result()], now=NOW)
    assert "summary" not in ranked[0].extra


def test_run_pipeline_summarizes_when_asked() -> None:
    """A summarizer is applied to the ranked survivors only."""
    ranked = run_pipeline(
        [_result(), _result(url="https://arxiv.org/abs/2401.00002")],
        summarizer=lambda items: ["概括"] * len(items),
        now=NOW,
    )
    assert [item.extra["summary"] for item in ranked] == ["概括", "概括"]


def test_run_pipeline_on_empty_input() -> None:
    """An empty result set never raises."""
    assert run_pipeline([], now=NOW) == []
