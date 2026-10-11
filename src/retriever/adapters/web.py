from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


class WebAdapter(SearchAdapter):
    """通用网页搜索适配器，对接 DuckDuckGo 免费搜索引擎 + trafilatura 正文提取。

    覆盖技术文档、博客、资料类检索意图，自动清洗导航/广告等冗余内容。
    正文提取失败时降级保存原始 HTML，保证流程不中断。
    """

    source_name = "web"
    _MIN_CONTENT_LENGTH = 30

    def __init__(self, http_client: HttpClient):
        """初始化 Web 适配器。

        Args:
            http_client: 项目统一 HTTP 客户端，所有网络请求必须经此发起
        """
        self.http = http_client
        self.search_base = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行通用网页搜索，返回标准化搜索结果列表。

        Args:
            query: 标准化查询对象

        Returns:
            符合 SearchResult 契约的结果列表，source 统一标记为 web
        """
        params = {
            "q": query.raw_query,
            "kl": "cn-zh",
        }

        response = await self.http.get(self.search_base, params=params)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        result_items = soup.select(".result")

        results: list[SearchResult] = []
        for item in result_items:
            title_elem = item.select_one(".result__a")
            snippet_elem = item.select_one(".result__snippet")

            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")
            abstract = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if not url or url.startswith("/"):
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    abstract=abstract,
                    source=self.source_name,
                    extra={},
                )
            )

        return results

    async def download(self, result: SearchResult, output_dir: str) -> str:
        """提取网页正文并保存为 Markdown 文件。

        正文提取失败或内容过短时，降级保存原始 HTML 快照。

        Args:
            result: 单条搜索结果
            output_dir: 本地保存目录

        Returns:
            本地文件绝对路径
        """
        os.makedirs(output_dir, exist_ok=True)

        safe_title = re.sub(r'[\\/:*?"<>|]', "_", result.title)
        md_path = Path(output_dir) / f"{safe_title}.md"

        response = await self.http.get(result.url, follow_redirects=True)
        response.raise_for_status()
        html_content = response.text

        markdown_content = trafilatura.extract(
            html_content,
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=False,
            fast=True,
        )

        if markdown_content and len(markdown_content.strip()) >= self._MIN_CONTENT_LENGTH:

            def _write_md(path: str, content: str) -> None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

            await asyncio.to_thread(_write_md, str(md_path), markdown_content)
            return str(md_path.resolve())

        html_path = Path(output_dir) / f"{safe_title}.html"

        def _write_html(path: str, content: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write_html, str(html_path), html_content)
        return str(html_path.resolve())
