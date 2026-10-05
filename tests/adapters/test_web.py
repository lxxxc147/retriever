from __future__ import annotations

import asyncio
import os

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
async def test_search_normal(web_adapter):
    """测试正常搜索场景，验证 HTML 结果解析与字段封装"""
    mock_search_html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/doc1">Python 异步编程最佳实践</a>
          <div class="result__snippet">本文详细介绍 Python 异步编程的核心概念与实战技巧...</div>
        </div>
        <div class="result">
          <a class="result__a" href="https://example.com/doc2">FastAPI 入门指南</a>
          <div class="result__snippet">FastAPI 是一个现代、高性能的 Web 框架...</div>
        </div>
      </body>
    </html>
    """

    route = respx.get("https://html.duckduckgo.com/html/").respond(
        text=mock_search_html, status_code=200
    )

    query = SearchQuery(
        intent="web",
        raw_query="python async best practice",
        keywords=["python", "async"],
    )
    results = await web_adapter.search(query)

    assert route.called
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "web"
    assert results[0].title == "Python 异步编程最佳实践"
    assert results[0].url == "https://example.com/doc1"
    assert "异步编程" in results[0].abstract


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_result(web_adapter):
    """测试无搜索结果场景，验证返回空列表不报错"""
    mock_empty_html = "<html><body><div class='no-results'>无匹配结果</div></body></html>"
    respx.get("https://html.duckduckgo.com/html/").respond(text=mock_empty_html, status_code=200)

    query = SearchQuery(
        intent="web",
        raw_query="nonexistent_keyword_xyz_123",
        keywords=["nonexistent_keyword_xyz_123"],
    )
    results = await web_adapter.search(query)

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
@respx.mock
async def test_download_markdown_success(web_adapter, tmp_path):
    """测试正文提取成功场景，验证保存为 Markdown 且核心正文完整"""
    mock_page_html = """
    <html>
      <head><title>测试文档</title></head>
      <body>
        <nav>顶部导航栏 广告内容</nav>
        <article>
          <h1>Python 异步编程最佳实践</h1>
          <p>异步编程可以显著提升 IO 密集型任务的性能。</p>
          <p>核心概念包括事件循环、协程与任务调度。</p>
          <p>本文将从基础概念出发，逐步深入讲解异步编程的高级用法与常见坑点，
             配合大量实战案例帮助读者快速掌握异步开发能力。</p>
        </article>
        <footer>页脚版权信息</footer>
      </body>
    </html>
    """

    respx.get("https://example.com/doc1").respond(text=mock_page_html, status_code=200)

    result = SearchResult(
        title="Python 异步编程最佳实践",
        url="https://example.com/doc1",
        source="web",
    )

    file_path = await web_adapter.download(result, str(tmp_path))

    assert os.path.exists(file_path)
    assert file_path.endswith(".md")

    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    content = await asyncio.to_thread(_read_file, file_path)
    assert "异步编程可以显著提升 IO 密集型任务的性能" in content
    assert "事件循环、协程与任务调度" in content
    assert "高级用法与常见坑点" in content


@pytest.mark.asyncio
@respx.mock
async def test_download_fallback_html(web_adapter, tmp_path):
    """测试正文提取失败降级，验证保存原始 HTML"""
    mock_bad_page = """
    <html>
      <body>
        <script>console.log('empty page')</script>
        <div class="ad">广告位</div>
      </body>
    </html>
    """

    respx.get("https://example.com/bad-page").respond(text=mock_bad_page, status_code=200)

    result = SearchResult(
        title="无效页面测试",
        url="https://example.com/bad-page",
        source="web",
    )

    file_path = await web_adapter.download(result, str(tmp_path))

    assert os.path.exists(file_path)
    assert file_path.endswith(".html")

    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    content = await asyncio.to_thread(_read_file, file_path)
    assert "广告位" in content


@pytest.mark.asyncio
@respx.mock
async def test_download_safe_filename(web_adapter, tmp_path):
    """测试特殊字符标题文件名清洗，避免系统报错"""
    mock_page = "<html><body><p>test content for validation</p></body></html>"
    respx.get("https://example.com/test").respond(text=mock_page, status_code=200)

    result = SearchResult(
        title="测试: 特殊/字符\\文件名*?<>|",
        url="https://example.com/test",
        source="web",
    )

    file_path = await web_adapter.download(result, str(tmp_path))

    assert os.path.exists(file_path)
    assert ":" not in os.path.basename(file_path)
    assert "/" not in os.path.basename(file_path)
