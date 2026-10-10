from __future__ import annotations

from retriever.adapters.base import SearchAdapter
from retriever.adapters.web import WebAdapter

# 可选导入 PDF 适配器，模块不存在则跳过注册
try:
    from retriever.adapters.pdf import PdfAdapter
except ModuleNotFoundError:
    PdfAdapter = None

from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


class AdapterRouter:
    """适配器路由调度器，根据查询意图自动分发到对应数据源适配器。

    统一管理所有已注册的适配器，提供标准的 search/download 入口，
    上层调用无需感知具体适配器实现细节。
    """

    def __init__(
        self,
        http_client: HttpClient,
        extra_intent_adapters: dict[str, type[SearchAdapter]] | None = None,
    ):
        """初始化适配器路由调度器。

        Args:
            http_client: 统一 HTTP 客户端实例，透传给所有适配器
            extra_intent_adapters: 可选扩展适配器映射，intent 到适配器类
        """
        self._http_client = http_client
        self._intent_adapters: dict[str, SearchAdapter] = {}
        self._source_adapters: dict[str, SearchAdapter] = {}

        # 内置意图到适配器的映射
        default_intent_map: dict[str, type[SearchAdapter]] = {
            "web": WebAdapter,
        }
        # PDF 模块存在时才注册
        if PdfAdapter is not None:
            default_intent_map["paper"] = PdfAdapter

        # 注册内置适配器
        for intent, adapter_cls in default_intent_map.items():
            self._register_adapter(intent, adapter_cls)

        # 注册扩展适配器
        if extra_intent_adapters:
            for intent, adapter_cls in extra_intent_adapters.items():
                self._register_adapter(intent, adapter_cls)

    def _register_adapter(self, intent: str, adapter_cls: type[SearchAdapter]) -> None:
        """注册适配器，同时建立 intent 和 source 双向映射。"""
        adapter = adapter_cls(self._http_client)
        self._intent_adapters[intent] = adapter
        self._source_adapters[adapter.source_name] = adapter

    def get_adapter_by_intent(self, intent: str) -> SearchAdapter:
        """根据查询意图获取对应适配器实例。

        Args:
            intent: 查询意图标识

        Returns:
            对应适配器实例

        Raises:
            ValueError: 未知的查询意图
        """
        adapter = self._intent_adapters.get(intent)
        if not adapter:
            supported = ", ".join(sorted(self._intent_adapters.keys()))
            raise ValueError(
                f"Unsupported search intent: {intent}. "
                f"Supported intents: {supported}"
            )
        return adapter

    def get_adapter_by_source(self, source: str) -> SearchAdapter:
        """根据结果来源获取对应适配器实例。

        Args:
            source: 结果来源标识

        Returns:
            对应适配器实例

        Raises:
            ValueError: 未知的结果来源
        """
        adapter = self._source_adapters.get(source)
        if not adapter:
            supported = ", ".join(sorted(self._source_adapters.keys()))
            raise ValueError(
                f"Unsupported result source: {source}. "
                f"Supported sources: {supported}"
            )
        return adapter

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """统一搜索入口，根据意图自动路由到对应适配器执行搜索。

        Args:
            query: 标准化查询对象

        Returns:
            标准化搜索结果列表
        """
        adapter = self.get_adapter_by_intent(query.intent)
        return await adapter.search(query)

    async def download(self, result: SearchResult, output_dir: str) -> str:
        """统一下载入口，根据结果来源自动路由到对应适配器执行下载。

        Args:
            result: 搜索结果项
            output_dir: 本地保存目录

        Returns:
            本地文件绝对路径
        """
        adapter = self.get_adapter_by_source(result.source)
        return await adapter.download(result, output_dir)
