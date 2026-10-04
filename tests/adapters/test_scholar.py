from __future__ import annotations

import asyncio
import os

import pytest
import respx

from retriever.adapters.scholar import ScholarAdapter, ScholarRateLimitError
from retriever.http import HttpClient
from retriever.models import Filters, SearchQuery, SearchResult


@pytest.fixture
def http_client():
    return HttpClient()


@pytest.fixture
def scholar_adapter(http_client):
    return ScholarAdapter(http_client, api_key="test_api_key")


@pytest.mark.asyncio
@respx.mock
async def test_search_normal(scholar_adapter):
    """测试正常搜索场景，验证字段与扩展信息正确封装"""
    mock_response = {
        "organic_results": [
            {
                "title": "Deep Learning for Natural Language Processing",
                "link": "https://example.com/paper1",
                "snippet": "This paper explores deep learning methods in NLP...",
                "publication_info": {
                    "summary": "Journal of AI, 2023 - Springer"
                },
                "inline_links": {
                    "cited_by": {
                        "total": 156
                    }
                }
            }
        ]
    }

    route = respx.get("https://serpapi.com/search").respond(json=mock_response, status_code=200)

    query = SearchQuery(
        intent="paper",
        raw_query="deep learning nlp",
        keywords=["deep learning", "nlp"],
    )
    results = await scholar_adapter.search(query)

    assert route.called
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "scholar"
    assert results[0].title == "Deep Learning for Natural Language Processing"
    assert results[0].url == "https://example.com/paper1"
    # 验证扩展字段
    assert results[0].extra["citations"] == 156
    assert "Journal of AI" in results[0].extra["venue"]
    assert results[0].extra["year"] == 2023


@pytest.mark.asyncio
@respx.mock
async def test_search_with_filters(scholar_adapter):
    """测试带年份范围、作者筛选的参数映射"""
    mock_response = {"organic_results": []}
    route = respx.get("https://serpapi.com/search").respond(json=mock_response, status_code=200)

    query = SearchQuery(
        intent="paper",
        raw_query="transformer",
        keywords=["transformer"],
        filters=Filters(
            year_from=2020,
            year_to=2024,
            authors=["vaswani"]
        )
    )
    await scholar_adapter.search(query)

    assert route.called
    params = route.calls[0].request.url.params
    assert params["as_ylo"] == "2020"
    assert params["as_yhi"] == "2024"
    assert "author:vaswani" in params["q"]


@pytest.mark.asyncio
@respx.mock
async def test_search_rate_limit(scholar_adapter):
    """测试 429 限流场景，验证抛出可识别异常"""
    respx.get("https://serpapi.com/search").respond(status_code=429)

    query = SearchQuery(
        intent="paper",
        raw_query="test",
        keywords=["test"],
    )
    with pytest.raises(ScholarRateLimitError):
        await scholar_adapter.search(query)


@pytest.mark.asyncio
@respx.mock
async def test_download_success(scholar_adapter, tmp_path):
    """测试下载成功场景，验证文件正确落盘"""
    pdf_content = b"%PDF-1.4 test content"
    respx.get("https://example.com/paper1").respond(content=pdf_content, status_code=200)

    result = SearchResult(
        title="Test Paper: NLP Survey",
        url="https://example.com/paper1",
        source="scholar"
    )

    file_path = await scholar_adapter.download(result, str(tmp_path))

    assert os.path.exists(file_path)
    assert file_path.endswith(".pdf")

    # 异步线程读取文件，避免阻塞事件循环
    def _read_pdf(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    content = await asyncio.to_thread(_read_pdf, file_path)
    assert content == pdf_content


@pytest.mark.asyncio
@respx.mock
async def test_download_fallback(scholar_adapter, tmp_path):
    """测试下载失败降级，验证保存 HTML 快照"""
    respx.get("https://example.com/broken").respond(status_code=404, text="Not Found")

    result = SearchResult(
        title="Broken Paper",
        url="https://example.com/broken",
        source="scholar"
    )

    file_path = await scholar_adapter.download(result, str(tmp_path))

    assert os.path.exists(file_path)
    assert file_path.endswith(".html")

    # 异步线程读取文件，避免阻塞事件循环
    def _read_html(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    content = await asyncio.to_thread(_read_html, file_path)
    assert "Not Found" in content
    assert "https://example.com/broken" in content


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_result(scholar_adapter):
    """测试无搜索结果场景，验证返回空列表不报错"""
    mock_response = {"organic_results": []}
    respx.get("https://serpapi.com/search").respond(json=mock_response, status_code=200)

    query = SearchQuery(
        intent="paper",
        raw_query="nonexistent_keyword_xyz",
        keywords=["nonexistent_keyword_xyz"],
    )
    results = await scholar_adapter.search(query)

    assert isinstance(results, list)
    assert len(results) == 0
