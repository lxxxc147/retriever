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
    """
    通用网页搜索适配器，对接 DuckDuckGo 免费搜索引擎 + trafilatura 正文提取。
    覆盖技术文档、博客、资料类检索意图，自动清洗导航/广告等冗余内容。
    正文提取失败时降级保存原始 HTML，保证流程不中断。
    """

    source_name = "web"
    _MIN_CONTENT_LENGTH = 30  # 有效正文最小长度阈值，低于此值视为提取失败

    def __init__(self, http_client: HttpClient):
        """
        初始化 Web 适配器。

        Args:
            http_client: 项目统一 HTTP 客户端，所有网络请求必须经此发起
        """
        self.http = http_client
        self.search_base = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        执行通用网页搜索，返回标准化搜索结果列表。

        Args:
            query: 标准化查询对象

        Returns:
            符合 SearchResult 契约的结果列表，source 统一标记为 web
        """
        # 1. 构建搜索请求参数
        params = {
            "q": query.raw_query,
            "kl": "cn-zh",
        }

        # 2. 统一通过 HttpClient 发起搜索请求
        response = await self.http.get(self.search_base, params=params)
        response.raise_for_status()

        # 3. 解析 HTML 搜索结果
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

            # 过滤无效相对链接
            if not url or url.startswith("/"):
                continue

            results.append(SearchResult(
                title=title,
                url=url,
                abstract=abstract,
                source=self.source_name,
                extra={},
            ))

        return results

    async def download(self, result: SearchResult, output_dir: str) -> str:
        """
        提取网页正文并保存为 Markdown 文件。
        正文提取失败或内容过短时，降级保存原始 HTML 快照。

        Args:
            result: 单条搜索结果
            output_dir: 本地保存目录

        Returns:
            本地文件绝对路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 清洗文件名，去除系统非法字符
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', result.title)
        md_path = Path(output_dir) / f"{safe_title}.md"

        # 1. 请求目标网页，统一走 HttpClient
        response = await self.http.get(result.url, follow_redirects=True)
        response.raise_for_status()
        html_content = response.text

        # 2. 提取正文为 Markdown 格式，自动过滤导航、广告
        markdown_content = trafilatura.extract(
            html_content,
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=False,
            fast=True,
        )

        # 3. 内容有效则保存 Markdown
        if markdown_content and len(markdown_content.strip()) >= self._MIN_CONTENT_LENGTH:
            def _write_md(path: str, content: str) -> None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

            await asyncio.to_thread(_write_md, str(md_path), markdown_content)
            return str(md_path.resolve())

        # 4. 提取失败 / 内容过短 降级：保存原始 HTML
        html_path = Path(output_dir) / f"{safe_title}.html"

        def _write_html(path: str, content: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write_html, str(html_path), html_content)
        return str(html_path.resolve())
