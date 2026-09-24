"""Shared test fixtures. All tests mock external APIs — no real network calls."""

from collections.abc import Callable

import pytest

from retriever.models import SearchQuery, SearchResult


@pytest.fixture
def sample_query() -> Callable[..., SearchQuery]:
    """Factory building a SearchQuery with sensible defaults."""

    def _make(**overrides: object) -> SearchQuery:
        data: dict = {
            "intent": "paper",
            "raw_query": "LLM agent survey 2024",
            "keywords": ["LLM", "agent", "survey"],
        }
        data.update(overrides)
        return SearchQuery(**data)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def sample_result() -> Callable[..., SearchResult]:
    """Factory building a SearchResult with sensible defaults."""

    def _make(**overrides: object) -> SearchResult:
        data: dict = {
            "source": "arxiv",
            "title": "A Survey on LLM-based Agents",
            "abstract": "A survey of LLM agents.",
            "url": "https://arxiv.org/abs/2401.00001",
        }
        data.update(overrides)
        return SearchResult(**data)  # type: ignore[arg-type]

    return _make
