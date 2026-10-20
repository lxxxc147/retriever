from __future__ import annotations

import pytest
import respx

from retriever.adapters.web import WebAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


@pytest.fixture
def http_client():
    return HttpClient()


@pytest.fixture
def web_adapter(http_client):
    return WebAdapter(http_client)


@pytest.mark.asyncio
@respx.mock
async def test_search_success(web_adapter):
    """测试搜索成功，返回标准化结果"""
    mock_html = """
    <html>
    <body>
        <div class="result">
            <a class="result__a" href="https://example.com/doc1">测试文档1</a>
            <div class="result__snippet">这是测试摘要内容</div>
        </div>
        <div class="result">
            <a class="result__a" href="https://example.com/doc2">测试文档2</a>
            <div class="result__snippet">第二篇摘要</div>
        </div>
    </body>
    </html>
    """
    respx.get("https://html.duckduckgo.com/html/").respond(
        status_code=200,
        text=mock_html,
    )

    query = SearchQuery(
        raw_query="test",
        keywords=["test"],
        limit=10,
        intent="web",
    )
    results = await web_adapter.search(query)

    assert len(results) == 2
    assert results[0].source == "web"
    assert results[0].title == "测试文档1"
    assert results[0].url == "https://example.com/doc1"
    assert "测试摘要" in results[0].abstract


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_results(web_adapter):
    """测试搜索无结果场景"""
    respx.get("https://html.duckduckgo.com/html/").respond(
        status_code=200,
        text="<html><body></body></html>",
    )

    query = SearchQuery(
        raw_query="nonexist",
        keywords=["nonexist"],
        limit=10,
        intent="web",
    )
    results = await web_adapter.search(query)

    assert len(results) == 0


@pytest.mark.asyncio
@respx.mock
async def test_download_extract_markdown_success(tmp_path, web_adapter):
    """测试下载成功提取正文为 Markdown，自动过滤导航、广告等页面冗余元素"""
    mock_html = """
    <html>
    <head><title>测试页面</title></head>
    <body>
        <nav>导航栏</nav>
        <main>
            <h1>测试标题</h1>
            <p>这是用于测试正文提取的核心内容，包含足够的字符数以满足最小长度阈值，验证提取工具可以正确过滤掉页面中的顶部菜单和底部推广等冗余元素，只保留有效正文。</p>
        </main>
        <footer>页脚广告</footer>
    </body>
    </html>
    """
    respx.get("https://example.com/test").respond(
        status_code=200,
        text=mock_html,
    )

    result = SearchResult(
        source="web",
        title="测试页面",
        url="https://example.com/test",
        abstract="",
    )
    download_result = await web_adapter.download(result, str(tmp_path))

    assert download_result.success is True
    assert download_result.local_path.exists()
    content = download_result.local_path.read_text(encoding="utf-8")
    assert "测试标题" in content
    assert "正文提取的核心内容" in content
    # 导航、广告等页面冗余元素被过滤
    assert "导航栏" not in content
    assert "页脚广告" not in content


@pytest.mark.asyncio
@respx.mock
async def test_download_fallback_to_html(tmp_path, web_adapter):
    """测试正文提取失败，降级保存原始 HTML"""
    # 构造无法提取有效正文的空页面
    mock_html = "<html><body><div></div></body></html>"
    respx.get("https://example.com/empty").respond(
        status_code=200,
        text=mock_html,
    )

    result = SearchResult(
        source="web",
        title="空页面",
        url="https://example.com/empty",
        abstract="",
    )
    download_result = await web_adapter.download(result, str(tmp_path))

    assert download_result.success is True
    assert download_result.local_path.suffix == ".html"
    content = download_result.local_path.read_text(encoding="utf-8")
    assert "<html>" in content


@pytest.mark.asyncio
@respx.mock
async def test_download_network_error(tmp_path, web_adapter):
    """测试下载网络异常，返回失败结果"""
    respx.get("https://example.com/error").respond(status_code=500)

    result = SearchResult(
        source="web",
        title="错误页面",
        url="https://example.com/error",
        abstract="",
    )
    download_result = await web_adapter.download(result, str(tmp_path))

    assert download_result.success is False
    assert download_result.error is not None


@pytest.mark.asyncio
@respx.mock
async def test_search_resolves_duckduckgo_redirect(web_adapter):
    """DDG 的 /l/?uddg= 跳转包装要还原成真实地址，否则 M4 去重键会失效。"""
    mock_html = """
    <html><body>
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Freal">真实结果</a>
            <div class="result__snippet">摘要</div>
        </div>
    </body></html>
    """
    respx.get("https://html.duckduckgo.com/html/").respond(status_code=200, text=mock_html)

    query = SearchQuery(raw_query="test", keywords=["test"], intent="web")
    results = await web_adapter.search(query)

    assert len(results) == 1
    assert results[0].url == "https://example.com/real"


@pytest.mark.asyncio
@respx.mock
async def test_relative_dd_links_are_skipped(web_adapter):
    """仍是相对链接（无法还原）的结果直接丢弃，不会污染 url 字段。"""
    mock_html = """
    <html><body>
        <div class="result">
            <a class="result__a" href="/l/?kh=-1">无目标</a>
        </div>
    </body></html>
    """
    respx.get("https://html.duckduckgo.com/html/").respond(status_code=200, text=mock_html)

    query = SearchQuery(raw_query="test", keywords=["test"], intent="web")
    assert await web_adapter.search(query) == []
