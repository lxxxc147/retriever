"""Tests for the Hugging Face datasets adapter (Phase 4).

The Hub API is mocked with respx — no real network access in any test here.
"""

from pathlib import Path

import pytest
import respx

from retriever.adapters.base import AdapterError
from retriever.adapters.huggingface import API_URL, HuggingfaceAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult

_PAYLOAD = [
    {
        "id": "nyu-mll/glue",
        "lastModified": "2024-03-01T10:00:00.000Z",
        "downloads": 250000,
        "likes": 1200,
        "tags": ["task_categories:text-classification", "language:en", "arxiv:1804.07461"],
    },
    {
        "id": "clips/mfaq",
        "lastModified": "2023-07-15T08:30:00.000Z",
        "downloads": 800,
        "likes": 12,
        "tags": ["task_categories:question-answering", "language:zh"],
    },
    {
        "not-a-dict-entries-are-filtered": True,
    },
]


@pytest.fixture
def adapter() -> HuggingfaceAdapter:
    return HuggingfaceAdapter(HttpClient())


def _dataset_query(**overrides: object) -> SearchQuery:
    values: dict = {
        "raw_query": "question answering dataset",
        "keywords": ["question", "answering"],
        "limit": 5,
        "intent": "dataset",
    }
    values.update(overrides)
    return SearchQuery(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
@respx.mock
async def test_search_maps_hub_entries_onto_the_contract(adapter: HuggingfaceAdapter) -> None:
    """Hub API entries become unified results with popularity metadata."""
    respx.get(API_URL).respond(status_code=200, json=_PAYLOAD)

    results = await adapter.search(_dataset_query())

    assert len(results) == 2  # the non-dict entry was skipped
    first = results[0]
    assert first.source == "huggingface"
    assert first.title == "nyu-mll/glue"
    assert first.url == "https://huggingface.co/datasets/nyu-mll/glue"
    assert first.download_url and "nyu-mll/glue" in first.download_url
    assert "downloads 250000" in first.abstract
    assert first.published_at is not None and first.published_at.year == 2024
    assert first.extra["stars"] == 1200  # M4 popularity dimension
    assert first.extra["language"] == "en"


@pytest.mark.asyncio
@respx.mock
async def test_search_language_filter_is_applied_client_side(adapter: HuggingfaceAdapter) -> None:
    """``filters.language`` keeps only datasets tagged language:<lang>."""
    from retriever.models import Filters

    respx.get(API_URL).respond(status_code=200, json=_PAYLOAD)

    results = await adapter.search(_dataset_query(filters=Filters(language="zh")))

    assert [result.title for result in results] == ["clips/mfaq"]
    assert results[0].extra["language"] == "zh"


@pytest.mark.asyncio
@respx.mock
async def test_search_respects_the_limit(adapter: HuggingfaceAdapter) -> None:
    """At most ``query.limit`` results are returned."""
    respx.get(API_URL).respond(status_code=200, json=_PAYLOAD)

    assert len(await adapter.search(_dataset_query(limit=1))) == 1


@pytest.mark.asyncio
@respx.mock
async def test_search_classifies_failures(adapter: HuggingfaceAdapter) -> None:
    """Transport failures raise AdapterError (kind=network) for router degradation."""
    respx.get(API_URL).respond(status_code=500)

    with pytest.raises(AdapterError) as excinfo:
        await adapter.search(_dataset_query())

    assert excinfo.value.source == "huggingface"
    assert excinfo.value.kind == "network"


@pytest.mark.asyncio
@respx.mock
async def test_download_saves_the_dataset_card(adapter: HuggingfaceAdapter, tmp_path: Path) -> None:
    """The card README is stored as Markdown with size and sha256 recorded."""
    respx.get("https://huggingface.co/datasets/nyu-mll/glue/raw/main/README.md").respond(
        status_code=200,
        text="# GLUE benchmark\n\nGeneral Language Understanding Evaluation...",
    )
    result = SearchResult(
        source="huggingface",
        title="nyu-mll/glue",
        url="https://huggingface.co/datasets/nyu-mll/glue",
        download_url="https://huggingface.co/datasets/nyu-mll/glue/raw/main/README.md",
    )

    outcome = await adapter.download(result, tmp_path)

    assert outcome.success
    assert outcome.local_path is not None and outcome.local_path.suffix == ".md"
    assert "GLUE benchmark" in outcome.local_path.read_text(encoding="utf-8")
    assert outcome.size_bytes > 0 and outcome.sha256


@pytest.mark.asyncio
@respx.mock
async def test_download_reports_failure_instead_of_raising(
    adapter: HuggingfaceAdapter, tmp_path: Path
) -> None:
    """A missing card yields success=False with the reason, never an exception."""
    respx.get("https://huggingface.co/datasets/x/y/raw/main/README.md").respond(status_code=404)
    result = SearchResult(
        source="huggingface",
        title="x/y",
        url="https://huggingface.co/datasets/x/y",
        download_url="https://huggingface.co/datasets/x/y/raw/main/README.md",
    )

    outcome = await adapter.download(result, tmp_path)

    assert not outcome.success
    assert outcome.error
