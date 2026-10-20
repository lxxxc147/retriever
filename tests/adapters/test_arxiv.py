import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from retriever.adapters.arxiv import ARXIV_API_URL, ArxivAdapter, sanitize_filename
from retriever.adapters.base import AdapterError, classify_http_error
from retriever.http import HttpClient
from retriever.models import Filters, SearchQuery, SearchResult

MOCK_SEARCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <title>Test LLM Agent Paper</title>
    <summary>This is a test paper about LLM agents.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Test Author</name></author>
    <arxiv:id>2401.01234</arxiv:id>
    <arxiv:primary_category term="cs.AI"/>
    <link title="pdf" href="https://arxiv.org/pdf/2401.01234.pdf" type="application/pdf"/>
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_arxiv_search_basic_fields():
    """测试搜索接口返回字段完全符合SearchResult契约"""
    with respx.mock(base_url="http://export.arxiv.org") as mock:
        mock.get("/api/query").return_value = httpx.Response(200, text=MOCK_SEARCH_XML)

        adapter = ArxivAdapter()
        query = SearchQuery(
            intent="paper",
            raw_query="llm agent",
            keywords=["llm", "agent"],
            limit=5,
        )
        results = await adapter.search(query)

        assert len(results) == 1
        assert results[0].source == "arxiv"
        assert results[0].title == "Test LLM Agent Paper"
        assert results[0].abstract == "This is a test paper about LLM agents."
        assert results[0].url == "https://arxiv.org/abs/2401.01234"
        assert results[0].download_url == "https://arxiv.org/pdf/2401.01234.pdf"
        assert results[0].published_at is not None
        assert results[0].published_at.year == 2024
        assert results[0].authors == ["Test Author"]
        assert results[0].extra["arxiv_id"] == "2401.01234"
        assert "cs.AI" in results[0].extra["categories"]


def test_extract_arxiv_id_all_formats():
    """测试ID提取覆盖新旧格式、URL格式、非法输入"""
    # 新格式纯ID
    assert ArxivAdapter.extract_arxiv_id("2401.03462") == "2401.03462"
    # 带版本号
    assert ArxivAdapter.extract_arxiv_id("2401.03462v2") == "2401.03462v2"
    # abs页面URL
    assert ArxivAdapter.extract_arxiv_id("https://arxiv.org/abs/2401.03462") == "2401.03462"
    # PDF链接URL
    assert ArxivAdapter.extract_arxiv_id("https://arxiv.org/pdf/2401.03462.pdf") == "2401.03462"
    # 旧格式ID
    assert ArxivAdapter.extract_arxiv_id("cs/0601001") == "cs/0601001"
    # 普通无意义文本
    assert ArxivAdapter.extract_arxiv_id("random text without id") is None


@pytest.mark.asyncio
async def test_download_success_creates_file():
    """测试下载成功：正确生成文件、返回字段完整"""
    test_pdf_content = b"%PDF-1.4 test binary content"
    with respx.mock as mock:
        mock.get("https://arxiv.org/pdf/2401.01234.pdf").return_value = httpx.Response(
            200, content=test_pdf_content
        )

        adapter = ArxivAdapter()
        result = SearchResult(
            source="arxiv",
            title="Test Paper Title",
            abstract="",
            url="https://arxiv.org/abs/2401.01234",
            download_url="https://arxiv.org/pdf/2401.01234.pdf",
            authors=[],
            extra={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dest_dir = Path(tmpdir)
            download_res = await adapter.download(result, dest_dir)

            assert download_res.success is True
            assert download_res.local_path is not None
            assert download_res.size_bytes == len(test_pdf_content)
            assert download_res.sha256 is not None
            assert download_res.local_path.exists()
            assert download_res.local_path.read_bytes() == test_pdf_content


@pytest.mark.asyncio
async def test_download_failure_returns_error_no_raise():
    """测试下载失败：不抛异常，返回success=False带错误信息"""
    with respx.mock as mock:
        mock.get("https://arxiv.org/pdf/invalid.pdf").return_value = httpx.Response(404)

        adapter = ArxivAdapter()
        result = SearchResult(
            source="arxiv",
            title="Invalid Paper",
            abstract="",
            url="",
            download_url="https://arxiv.org/pdf/invalid.pdf",
            authors=[],
            extra={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            download_res = await adapter.download(result, Path(tmpdir))
            assert download_res.success is False
            assert download_res.error is not None


@pytest.mark.asyncio
async def test_get_by_id_returns_single_result():
    """测试ID直达功能：跳过搜索，直接获取单条结果"""
    with respx.mock(base_url="http://export.arxiv.org") as mock:
        mock.get("/api/query").return_value = httpx.Response(200, text=MOCK_SEARCH_XML)

        adapter = ArxivAdapter()
        res = await adapter.get_by_id("2401.01234")

        assert res is not None
        assert res.extra["arxiv_id"] == "2401.01234"
        assert res.source == "arxiv"


def test_sanitize_filename_truncates_and_removes_illegal():
    """测试文件名安全化：截断80字符，移除非法字符"""
    long_title = "a" * 100
    assert len(sanitize_filename(long_title)) == 80

    bad_title = 'paper: with / illegal \\ chars *?'
    safe = sanitize_filename(bad_title)
    assert ":" not in safe
    assert "/" not in safe
    assert "\\" not in safe


# ---------------------------------------------------------------- 失败语义与降级
@pytest.mark.asyncio
@respx.mock
async def test_search_downgrades_filters_after_upstream_500():
    """arXiv 对 "作者 + 年份" 组合返回 500 时，自动放宽条件重试并拿到结果。"""
    adapter = ArxivAdapter(http=HttpClient(rate_limit_qps=1000.0, max_retries=1))
    route = respx.get(ARXIV_API_URL)
    route.side_effect = [
        httpx.Response(500, text="internal error"),
        httpx.Response(200, text=MOCK_SEARCH_XML),
    ]
    query = SearchQuery(
        intent="paper",
        raw_query="Vaswani 2017 attention",
        keywords=["attention"],
        filters=Filters(authors=["Vaswani"], year_from=2017, year_to=2017),
    )

    results = await adapter.search(query)

    assert len(results) == 1
    assert route.call_count == 2
    first = route.calls[0].request.url.params["search_query"]
    second = route.calls[1].request.url.params["search_query"]
    assert "submittedDate" in first and "au:" in first
    assert "submittedDate" not in second and "au:" in second
    # 降级顺序：先去 submittedDate，作者条件还在；再不行才去掉作者


@pytest.mark.asyncio
@respx.mock
async def test_search_raises_classified_error_on_rate_limit():
    """429 不该静默返回空列表，而是抛可读的 AdapterError(kind=rate_limit)。"""
    adapter = ArxivAdapter(http=HttpClient(rate_limit_qps=1000.0, max_retries=1))
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(429))

    query = SearchQuery(intent="paper", raw_query="q", keywords=["q"])

    with pytest.raises(AdapterError) as excinfo:
        await adapter.search(query)

    assert excinfo.value.kind == "rate_limit"
    assert excinfo.value.source == "arxiv"


@pytest.mark.asyncio
@respx.mock
async def test_search_raises_classified_error_on_forbidden():
    """403 归为 auth，提示用户去检查鉴权/反爬，而不是重试。"""
    adapter = ArxivAdapter(http=HttpClient(rate_limit_qps=1000.0, max_retries=1))
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(403))

    query = SearchQuery(intent="paper", raw_query="q", keywords=["q"])

    with pytest.raises(AdapterError) as excinfo:
        await adapter.search(query)

    assert excinfo.value.kind == "auth"


@pytest.mark.asyncio
@respx.mock
async def test_search_returns_empty_when_every_variant_is_empty():
    """所有条件都成功但都没结果时才返回空列表（不是错误）。"""
    adapter = ArxivAdapter(http=HttpClient(rate_limit_qps=1000.0, max_retries=1))
    empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    respx.get(ARXIV_API_URL).mock(return_value=httpx.Response(200, text=empty))

    query = SearchQuery(
        intent="paper", raw_query="q", keywords=["q"], filters=Filters(authors=["Nobody"])
    )

    assert await adapter.search(query) == []


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (403, "auth"), (429, "rate_limit"), (404, "not_found"), (503, "network")],
)
def test_classify_http_status(status: int, kind: str):
    """HTTP 状态码到失败类别的映射（CLI 用它给出可读原因）。"""
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(status, request=request)
    error = classify_http_error(
        httpx.HTTPStatusError("boom", request=request, response=response), "arxiv"
    )
    assert error.kind == kind


def test_classify_timeout_is_network():
    """超时归为 network。"""
    error = classify_http_error(httpx.ReadTimeout("slow"), "arxiv")
    assert error.kind == "network"
