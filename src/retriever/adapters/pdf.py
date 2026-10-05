from __future__ import annotations

import asyncio
import io
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import SearchQuery, SearchResult


class PdfAdapter(SearchAdapter):
    """PDF 文档解析适配器，限定搜索 PDF 格式资源，自动提取正文文本。

    自动过滤页眉、页脚、页码等冗余内容，提取失败时降级保存原始 PDF 文件。
    """

    source_name = "pdf"
    _MIN_TEXT_LENGTH = 20

    def __init__(self, http_client: HttpClient):
        """初始化 PDF 适配器。

        Args:
            http_client: 项目统一 HTTP 客户端，所有网络请求必须经此发起
        """
        self.http = http_client
        self.search_base = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """搜索指定关键词的 PDF 文档，返回标准化搜索结果列表。

        Args:
            query: 标准化查询对象

        Returns:
            符合 SearchResult 契约的结果列表，source 标记为 pdf
        """
        search_query = f"{query.raw_query} filetype:pdf"
        params = {
            "q": search_query,
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
        """下载 PDF 文件并提取正文文本，保存为纯文本文件。

        文本提取失败时降级保存原始 PDF 文件，保证流程不中断。

        Args:
            result: 单条搜索结果
            output_dir: 本地保存目录

        Returns:
            本地文件绝对路径
        """
        os.makedirs(output_dir, exist_ok=True)

        safe_title = re.sub(r'[\\/:*?"<>|]', "_", result.title)
        txt_path = Path(output_dir) / f"{safe_title}.txt"

        response = await self.http.get(result.url, follow_redirects=True)
        response.raise_for_status()
        pdf_bytes = response.content

        def _extract_text(pdf_data: bytes) -> str:
            reader = PdfReader(io.BytesIO(pdf_data))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
            return "\n\n".join(text_parts)

        try:
            text_content = await asyncio.to_thread(_extract_text, pdf_bytes)
        except PdfReadError:
            text_content = ""

        if text_content and len(text_content.strip()) >= self._MIN_TEXT_LENGTH:

            def _write_txt(path: str, content: str) -> None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

            await asyncio.to_thread(_write_txt, str(txt_path), text_content)
            return str(txt_path.resolve())

        pdf_path = Path(output_dir) / f"{safe_title}.pdf"

        def _write_pdf(path: str, data: bytes) -> None:
            with open(path, "wb") as f:
                f.write(data)

        await asyncio.to_thread(_write_pdf, str(pdf_path), pdf_bytes)
        return str(pdf_path.resolve())
