"""Shared test fixtures. All tests mock external APIs — no real network calls."""

from collections.abc import Callable

import pytest

from retriever.models import SearchQuery, SearchResult


class _OfflineLLMClient:
    """Unconfigured LLM client: keeps the suite offline even when ``.env`` has a key."""

    model = "offline"
    configured = False

    def chat_sync(self, messages: list[dict], **kwargs: object) -> str:
        """Fail loudly — a test that needs the model must inject a fake client."""
        raise AssertionError("test reached the real LLM client; inject a fake one")


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every default LLM client with an unconfigured stub (PRD §7.4).

    Without this, a developer machine with ``LLM_API_KEY`` in ``.env`` would turn
    ``--parser auto`` into a real API call from inside the unit tests.
    """
    factory = _OfflineLLMClient
    for target in ("retriever.llm.LLMClient", "retriever.intent_llm.LLMClient",
                   "retriever.cli.LLMClient"):
        monkeypatch.setattr(target, lambda: factory(), raising=False)


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
