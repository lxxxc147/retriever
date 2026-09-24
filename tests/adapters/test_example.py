"""Tests for ExampleAdapter. 测试不打真实 API，全部 mock。"""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from retriever.adapters.example import ExampleAdapter, sanitize_filename
from retriever.http import HttpClient

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "example_search.json"
PDF_CONTENT = b"%PDF-1.4 fake paper bytes"


@pytest.fixture
def adapter(respx_mock: respx.MockRouter) -> ExampleAdapter:
    """ExampleAdapter backed by a fast (never-waiting) HttpClient over respx."""
    return ExampleAdapter(http=HttpClient(rate_limit_qps=1000.0))


def _load_fixture() -> dict:
    with FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


@respx.mock
async def test_search_parses_fixture(adapter: ExampleAdapter, sample_query) -> None:
    """① search parses the fixture JSON into 3 SearchResults with correct fields."""
    respx.get("https://example.org/api/search").mock(
        return_value=httpx.Response(200, json=_load_fixture())
    )
    results = await adapter.search(sample_query(keywords=["LLM", "agent"]))

    assert len(results) == 3
    assert results[0].source == "example"
    assert "大模型智能体综述" in results[0].title
    assert results[0].url == "https://example.org/papers/2401.00001"
    assert results[0].download_url == "https://example.org/files/2401.00001.pdf"
    assert "survey" in results[0].abstract
    assert results[1].download_url is None  # fixture item without download_url


@respx.mock
async def test_download_writes_file_with_hash(
    adapter: ExampleAdapter, sample_result, tmp_path: Path
) -> None:
    """② download writes bytes, computes sha256/size, and sanitizes the file name."""
    respx.get("https://example.org/files/2401.00001.pdf").mock(
        return_value=httpx.Response(200, content=PDF_CONTENT)
    )
    result = sample_result(
        source="example",
        title="What is RAG? 检索增强生成 explained in detail",
        url="https://example.org/papers/2401.00001",
        download_url="https://example.org/files/2401.00001.pdf",
    )

    download = await adapter.download(result, tmp_path)

    assert download.success
    assert download.size_bytes == len(PDF_CONTENT)
    assert download.sha256 == hashlib.sha256(PDF_CONTENT).hexdigest()
    assert download.local_path is not None
    assert download.local_path.read_bytes() == PDF_CONTENT
    name = download.local_path.name
    assert "?" not in name and " " not in name
    assert "_" in name
    assert name.endswith(".pdf")


def test_sanitize_filename_truncates_long_titles() -> None:
    """File names are capped at 80 characters for the stem."""
    sanitized = sanitize_filename("Long " * 40)
    assert len(sanitized) == 80
    assert sanitize_filename("") == "untitled"


@respx.mock
async def test_download_creates_dest_dir(
    adapter: ExampleAdapter, sample_result, tmp_path: Path
) -> None:
    """③ A missing destination directory is created automatically."""
    respx.get("https://example.org/files/1706.03762.pdf").mock(
        return_value=httpx.Response(200, content=PDF_CONTENT)
    )
    result = sample_result(
        source="example",
        title="Attention Is All You Need",
        download_url="https://example.org/files/1706.03762.pdf",
    )
    dest = tmp_path / "nested" / "deep" / "dir"

    download = await adapter.download(result, dest)

    assert download.success
    assert download.local_path is not None
    assert download.local_path.parent == dest
    assert dest.is_dir()


@respx.mock
async def test_download_404_returns_failure(
    adapter: ExampleAdapter, sample_result, tmp_path: Path
) -> None:
    """④ A 404 from the target URL yields success=False instead of raising."""
    respx.get("https://example.org/files/gone.pdf").mock(
        return_value=httpx.Response(404)
    )
    result = sample_result(
        source="example",
        title="Gone paper",
        download_url="https://example.org/files/gone.pdf",
    )

    download = await adapter.download(result, tmp_path)

    assert not download.success
    assert download.error
    assert download.local_path is None
