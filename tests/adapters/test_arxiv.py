import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from retriever.adapters.arxiv import ArxivAdapter, sanitize_filename
from retriever.models import SearchQuery, SearchResult

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
