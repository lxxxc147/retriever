from __future__ import annotations

import pytest
import respx

from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult
from retriever.router import AdapterRouter


@pytest.fixture
def http_client():
    return HttpClient()


@pytest.fixture
def router(http_client):
    return AdapterRouter(http_client)


@pytest.mark.asyncio
async def test_router_init_builtin_adapters(router):
    """测试路由初始化，验证内置适配器双向映射正确注册"""
    # 验证 intent 映射
    assert "web" in router._intent_adapters
    assert "paper" in router._intent_adapters
    # 验证 source 映射
    assert "web" in router._source_adapters
    assert "pdf" in router._source_adapters


@pytest.mark.asyncio
async def test_get_adapter_by_intent_success(router):
    """测试正常获取已知意图的适配器"""
    web_adapter = router.get_adapter_by_intent("web")
    assert web_adapter.source_name == "web"

    pdf_adapter = router.get_adapter_by_intent("paper")
    assert pdf_adapter.source_name == "pdf"


@pytest.mark.asyncio
async def test_get_adapter_by_intent_unknown(router):
    """测试未知意图抛出明确异常"""
    with pytest.raises(ValueError, match="Unsupported search intent"):
        router.get_adapter_by_intent("unknown_intent")


@pytest.mark.asyncio
async def test_get_adapter_by_source_success(router):
    """测试正常获取已知来源的适配器"""
    web_adapter = router.get_adapter_by_source("web")
    assert web_adapter.source_name == "web"

    pdf_adapter = router.get_adapter_by_source("pdf")
    assert pdf_adapter.source_name == "pdf"


@pytest.mark.asyncio
async def test_get_adapter_by_source_unknown(router):
    """测试未知来源抛出明确异常"""
    with pytest.raises(ValueError, match="Unsupported result source"):
        router.get_adapter_by_source("unknown_source")


@pytest.mark.asyncio
@respx.mock
async def test_search_route_web(router):
    """测试搜索请求正确路由到 Web 适配器"""
    mock_html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/test">测试页面</a>
          <div class="result__snippet">测试摘要内容</div>
        </div>
      </body>
    </html>
    """
    respx.get("https://html.duckduckgo.com/html/").respond(text=mock_html, status_code=200)

    query = SearchQuery(
        intent="web",
        raw_query="test query",
        keywords=["test"],
    )
    results = await router.search(query)

    assert len(results) == 1
    assert results[0].source == "web"
    assert results[0].title == "测试页面"


@pytest.mark.asyncio
@respx.mock
async def test_search_route_pdf(router):
    """测试搜索请求正确路由到 PDF 适配器"""
    mock_html = """
    <html>
      <body>
        <div class="result">
          <a class="result__a" href="https://example.com/doc.pdf">测试文档.pdf</a>
          <div class="result__snippet">PDF 文档摘要</div>
        </div>
      </body>
    </html>
    """
    respx.get("https://html.duckduckgo.com/html/").respond(text=mock_html, status_code=200)

    query = SearchQuery(
        intent="paper",
        raw_query="test pdf",
        keywords=["test"],
    )
    results = await router.search(query)

    assert len(results) == 1
    assert results[0].source == "pdf"
    assert results[0].title == "测试文档.pdf"


@pytest.mark.asyncio
async def test_download_route_by_source(router, tmp_path, monkeypatch):
    """测试下载请求根据 source 字段正确路由"""
    called = False

    async def mock_download(*args, **kwargs):
        nonlocal called
        called = True
        return str(tmp_path / "test.txt")

    monkeypatch.setattr(
        "retriever.adapters.pdf.PdfAdapter.download",
        mock_download,
    )

    result = SearchResult(
        title="测试文档",
        url="https://example.com/doc.pdf",
        source="pdf",
    )
    path = await router.download(result, str(tmp_path))

    assert called is True
    assert path.endswith("test.txt")


@pytest.mark.asyncio
async def test_router_extra_adapters(http_client):
    """测试支持注册自定义扩展适配器"""
    from retriever.adapters.base import SearchAdapter

    class CustomAdapter(SearchAdapter):
        source_name = "custom"

        def __init__(self, http_client: HttpClient):
            self.http_client = http_client

        async def search(self, query):
            return []

        async def download(self, result, output_dir):
            return ""

    router = AdapterRouter(
        http_client,
        extra_intent_adapters={"custom_intent": CustomAdapter},
    )

    # 验证 intent 映射
    adapter = router.get_adapter_by_intent("custom_intent")
    assert isinstance(adapter, CustomAdapter)
    # 验证 source 映射
    adapter2 = router.get_adapter_by_source("custom")
    assert isinstance(adapter2, CustomAdapter)
