from __future__ import annotations

import pytest
import respx

from retriever.adapters.pdf import PdfAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


@pytest.fixture
def http_client():
    return HttpClient()


@pytest.fixture
def pdf_adapter(http_client):
    return PdfAdapter(http_client)


@pytest.mark.asyncio
@respx.mock
async def test_search_normal(pdf_adapter):
    """测试正常搜索场景，验证限定 PDF 类型与结果封装"""
    mock_search_html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/doc1.pdf">Python 异步编程指南.pdf</a>
          <div class="result__snippet">本文详细讲解 Python 异步编程的核心原理与实战...</div>
        </div>
        <div class="result">
          <a class="result__a" href="https://example.com/doc2.pdf">FastAPI 开发手册.pdf</a>
          <div class="result__snippet">FastAPI 高性能 Web 框架开发完整指南...</div>
        </div>
      </body>
    </html>
    """

    route = respx.get("https://html.duckduckgo.com/html/").respond(
        text=mock_search_html, status_code=200
    )

    query = SearchQuery(
        intent="paper",
        raw_query="python async",
        keywords=["python", "async"],
    )
    results = await pdf_adapter.search(query)

    assert route.called
    params = route.calls[0].request.url.params
    assert "filetype:pdf" in params["q"]

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].source == "pdf"
    assert results[0].title == "Python 异步编程指南.pdf"
    assert results[0].url == "https://example.com/doc1.pdf"
    assert "异步编程" in results[0].abstract


@pytest.mark.asyncio
@respx.mock
async def test_search_empty_result(pdf_adapter):
    """测试无搜索结果场景，验证返回空列表不报错"""
    mock_empty_html = "<html><body><div class='no-results'>无匹配结果</div></body></html>"
    respx.get("https://html.duckduckgo.com/html/").respond(text=mock_empty_html, status_code=200)

    query = SearchQuery(
        intent="paper",
        raw_query="nonexistent_pdf_file_xyz",
        keywords=["nonexistent_pdf_file_xyz"],
    )
    results = await pdf_adapter.search(query)

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
@respx.mock
async def test_download_text_success(pdf_adapter, tmp_path, monkeypatch):
    """测试 PDF 文本提取成功场景，验证保存为纯文本文件"""
    mock_pdf = b"%PDF-1.4 mock pdf content"
    respx.get("https://example.com/doc1.pdf").respond(content=mock_pdf, status_code=200)

    # Mock PDF 解析器，直接返回指定文本
    def mock_reader(*args, **kwargs):
        class MockPage:
            def extract_text(self):
                return "Python 异步编程最佳实践指南，包含事件循环、协程、任务调度等核心内容。"

        class MockReader:
            def __init__(self):
                self.pages = [MockPage()]

        return MockReader()

    monkeypatch.setattr("retriever.adapters.pdf.PdfReader", mock_reader)

    result = SearchResult(
        title="Python 异步编程指南",
        url="https://example.com/doc1.pdf",
        source="pdf",
    )

    outcome = await pdf_adapter.download(result, tmp_path)

    assert outcome.success is True
    assert outcome.local_path is not None
    assert outcome.local_path.exists()
    assert outcome.local_path.suffix == ".txt"
    assert outcome.sha256

    content = outcome.local_path.read_text(encoding="utf-8")
    assert "Python 异步编程最佳实践指南" in content
    assert "事件循环、协程、任务调度" in content


@pytest.mark.asyncio
@respx.mock
async def test_download_fallback_pdf(pdf_adapter, tmp_path, monkeypatch):
    """测试文本提取失败降级，验证保存原始 PDF 文件"""
    bad_pdf = b"This is not a valid PDF file, corrupted data"
    respx.get("https://example.com/bad.pdf").respond(content=bad_pdf, status_code=200)

    # Mock 解析器抛出读取异常，触发降级逻辑
    def mock_reader_error(*args, **kwargs):
        from pypdf.errors import PdfReadError

        raise PdfReadError("Invalid PDF")

    monkeypatch.setattr("retriever.adapters.pdf.PdfReader", mock_reader_error)

    result = SearchResult(
        title="损坏的测试文档",
        url="https://example.com/bad.pdf",
        source="pdf",
    )

    outcome = await pdf_adapter.download(result, tmp_path)

    assert outcome.success is True
    assert outcome.local_path is not None
    assert outcome.local_path.suffix == ".pdf"
    assert b"This is not a valid PDF" in outcome.local_path.read_bytes()


@pytest.mark.asyncio
@respx.mock
async def test_download_safe_filename(pdf_adapter, tmp_path, monkeypatch):
    """测试特殊字符标题文件名清洗，避免系统报错"""
    mock_pdf = b"%PDF-1.4 test content"
    respx.get("https://example.com/test.pdf").respond(content=mock_pdf, status_code=200)

    # Mock 解析失败，走降级分支，验证文件名清洗
    def mock_reader_error(*args, **kwargs):
        from pypdf.errors import PdfReadError

        raise PdfReadError("Invalid PDF")

    monkeypatch.setattr("retriever.adapters.pdf.PdfReader", mock_reader_error)

    result = SearchResult(
        title="测试: 特殊/字符\\文件名*?<>|",
        url="https://example.com/test.pdf",
        source="pdf",
    )

    outcome = await pdf_adapter.download(result, tmp_path)

    assert outcome.success is True
    assert outcome.local_path is not None
    name = outcome.local_path.name
    assert ":" not in name
    assert "/" not in name
    assert "?" not in name


@pytest.mark.asyncio
@respx.mock
async def test_download_network_error_returns_failure_result(pdf_adapter, tmp_path):
    """网络失败必须返回 success=False，而不是抛异常（PRD §2.5）。"""
    respx.get("https://example.com/missing.pdf").respond(status_code=404)

    result = SearchResult(
        title="不存在的文件",
        url="https://example.com/missing.pdf",
        source="pdf",
    )

    outcome = await pdf_adapter.download(result, tmp_path)

    assert outcome.success is False
    assert outcome.error
    assert outcome.local_path is None


@pytest.mark.asyncio
@respx.mock
async def test_search_resolves_duckduckgo_redirect(pdf_adapter):
    """DDG 的 /l/?uddg= 跳转包装要还原成真实地址，否则去重与下载都会出错。"""
    html = (
        '<div class="result">'
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Freal.pdf">'
        "真实文档</a>"
        '<a class="result__snippet">摘要</a>'
        "</div>"
    )
    respx.get("https://html.duckduckgo.com/html/").respond(text=html, status_code=200)

    query = SearchQuery(intent="paper", raw_query="real", keywords=["real"])
    results = await pdf_adapter.search(query)

    assert len(results) == 1
    assert results[0].url == "https://example.com/real.pdf"
