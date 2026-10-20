from __future__ import annotations

import pytest
import respx

from retriever.adapters.scholar import ScholarAdapter, ScholarBlockedError, ScholarRateLimitError
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


@pytest.fixture
def http_client():
    return HttpClient()


@pytest.fixture
def scholar_adapter(http_client):
    return ScholarAdapter(http_client)


@pytest.mark.asyncio
@respx.mock
async def test_search_success_extract_fields(scholar_adapter):
    """测试搜索成功，正确提取引用数、期刊、年份"""
    mock_html = """
    <div class="gs_r gs_or gs_scl">
        <h3 class="gs_rt"><a href="https://example.com/paper1">深度学习研究进展</a></h3>
        <div class="gs_rs">本文综述了深度学习的最新发展与应用。</div>
        <div class="gs_a">张三, 李四 - 计算机学报, 2023 - 被引用次数：128</div>
    </div>
    """
    respx.get("https://scholar.google.com/scholar").respond(
        status_code=200,
        text=mock_html,
    )

    query = SearchQuery(
        raw_query="deep learning",
        keywords=["deep learning"],
        limit=10,
        intent="paper",
    )
    results = await scholar_adapter.search(query)

    assert len(results) == 1
    assert results[0].source == "scholar"
    assert results[0].title == "深度学习研究进展"
    assert results[0].extra["citations"] == 128
    assert results[0].extra["venue"] == "计算机学报"
    assert results[0].extra["year"] == "2023"


@pytest.mark.asyncio
@respx.mock
async def test_search_rate_limit_raise_exception(scholar_adapter):
    """测试触发限流时抛出可识别异常"""
    respx.get("https://scholar.google.com/scholar").respond(status_code=429)

    query = SearchQuery(
        raw_query="test",
        keywords=["test"],
        limit=10,
        intent="paper",
    )
    with pytest.raises(ScholarRateLimitError):
        await scholar_adapter.search(query)


@pytest.mark.asyncio
@respx.mock
async def test_search_blocked_raise_exception(scholar_adapter):
    """测试封禁时抛出可识别异常"""
    respx.get("https://scholar.google.com/scholar").respond(status_code=403)

    query = SearchQuery(
        raw_query="test",
        keywords=["test"],
        limit=10,
        intent="paper",
    )
    with pytest.raises(ScholarBlockedError):
        await scholar_adapter.search(query)


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_results(scholar_adapter):
    """测试无结果场景"""
    respx.get("https://scholar.google.com/scholar").respond(
        status_code=200,
        text="<html><body></body></html>",
    )

    query = SearchQuery(
        raw_query="nonexist",
        keywords=["nonexist"],
        limit=10,
        intent="paper",
    )
    results = await scholar_adapter.search(query)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_download_returns_failure_result(scholar_adapter, tmp_path):
    """Scholar 只做检索：download 返回 success=False，而不是抛 NotImplementedError。"""
    result = SearchResult(source="scholar", title="t", url="https://example.com/x")

    outcome = await scholar_adapter.download(result, tmp_path)

    assert outcome.success is False
    assert outcome.error


@pytest.mark.asyncio
@respx.mock
async def test_search_maps_year_to_published_at(scholar_adapter):
    """页面年份要写进 published_at，M4 的时效性打分才拿得到。"""
    html = (
        '<div class="gs_r gs_or gs_scl">'
        '<h3 class="gs_rt"><a href="https://example.com/p">论文标题</a></h3>'
        '<div class="gs_rs">摘要</div>'
        '<div class="gs_a">张三 - 计算机学报, 2021 - example.com</div>'
        "</div>"
    )
    respx.get("https://scholar.google.com/scholar").respond(text=html, status_code=200)

    query = SearchQuery(intent="paper", raw_query="x", keywords=["x"])
    results = await scholar_adapter.search(query)

    assert len(results) == 1
    assert results[0].published_at is not None
    assert results[0].published_at.year == 2021