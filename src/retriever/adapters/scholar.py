from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


class ScholarRateLimitError(Exception):
    """Google Scholar 数据源限流/封禁异常，供 Router 降级识别"""


class ScholarAdapter(SearchAdapter):
    """
    Google Scholar 数据源适配器，通过 SerpAPI 免费层实现学术论文搜索。
    支持引用数、发表来源、年份等扩展字段，内置限流识别与下载降级机制。
    """

    source_name = "scholar"

    def __init__(self, http_client: HttpClient, api_key: str | None = None):
        """
        初始化 Scholar 适配器。

        Args:
            http_client: 项目统一 HTTP 客户端，所有网络请求必须经此发起
            api_key: SerpAPI 密钥，由上层配置注入
        """
        self.http = http_client
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        执行 Google Scholar 搜索，返回标准化结果列表。

        Args:
            query: 标准化查询对象，含关键词、作者、年份范围等参数

        Returns:
            符合 SearchResult 契约的结果列表，extra 携带 citations、venue、年份

        Raises:
            ScholarRateLimitError: 触发限流/封禁时抛出，供上层降级处理
        """
        # 1. 构建 SerpAPI 请求参数
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "q": query.raw_query,
            "api_key": self.api_key,
            "hl": "zh-CN",
        }

        # 映射年份范围筛选（从 filters 嵌套对象中读取）
        if query.filters.year_from:
            params["as_ylo"] = str(query.filters.year_from)
        if query.filters.year_to:
            params["as_yhi"] = str(query.filters.year_to)

        # 映射作者筛选参数（从 filters 嵌套对象中读取）
        if query.filters.authors:
            author_query = " ".join(f"author:{a}" for a in query.filters.authors)
            params["q"] = f"{query.raw_query} {author_query}"

        # 2. 统一通过 HttpClient 发起请求
        try:
            response = await self.http.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            # 识别 429 限流状态，抛出可识别异常供 Router 降级
            if hasattr(e, "response") and e.response.status_code == 429:
                raise ScholarRateLimitError("Google Scholar 触发限流，请稍后重试") from e
            raise

        # 3. 解析响应并标准化为 SearchResult
        results: list[SearchResult] = []
        organic_results = data.get("organic_results", [])

        for item in organic_results:
            title = item.get("title", "")
            url = item.get("link", "")
            abstract = item.get("snippet", "")

            # 提取扩展字段：引用数、发表来源、年份
            pub_info = item.get("publication_info", {})
            cited_by = item.get("inline_links", {}).get("cited_by", {}).get("total", 0)
            venue = pub_info.get("summary", "")

            # 从发表信息中正则提取 4 位数字年份
            year: int | None = None
            year_match = re.search(r"\b(19|20)\d{2}\b", venue)
            if year_match:
                year = int(year_match.group())

            # 扩展字段严格对齐验收标准
            extra = {
                "citations": cited_by,
                "venue": venue,
                "year": year,
            }

            results.append(SearchResult(
                title=title,
                url=url,
                abstract=abstract,
                source=self.source_name,
                extra=extra,
            ))

        return results

    async def download(self, result: SearchResult, output_dir: str) -> str:
        """
        下载论文资源并返回本地文件路径。
        Google Scholar 无统一 PDF 直链，优先尝试下载结果链接；
        下载失败时降级保存 HTML 快照，避免流程中断。

        Args:
            result: 单条搜索结果
            output_dir: 本地保存目录

        Returns:
            本地文件绝对路径

        Raises:
            ScholarRateLimitError: 触发限流/封禁时抛出
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 清洗文件名，去除系统非法字符
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', result.title)
        file_path = Path(output_dir) / f"{safe_title}.pdf"

        try:
            # 统一通过 HttpClient 发起下载请求，自动跟随重定向
            response = await self.http.get(result.url, follow_redirects=True)
            response.raise_for_status()

            # 异步线程中执行文件写入，避免阻塞事件循环
            def _write_pdf(path: str, content: bytes) -> None:
                with open(path, "wb") as f:
                    f.write(content)

            await asyncio.to_thread(_write_pdf, str(file_path), response.content)
            return str(file_path.resolve())

        except Exception as e:
            # 识别限流异常并向上抛出
            if hasattr(e, "response") and e.response.status_code == 429:
                raise ScholarRateLimitError("Google Scholar 下载触发限流") from e

            # 通用下载失败降级：保存 HTML 快照
            html_path = Path(output_dir) / f"{safe_title}.html"

            def _write_html(path: str, content: str) -> None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

            html_content = f"<!-- 原始链接: {result.url} -->\n"
            if hasattr(e, "response"):
                html_content += getattr(e.response, "text", "")

            await asyncio.to_thread(_write_html, str(html_path), html_content)
            return str(html_path.resolve())
