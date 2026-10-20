"""PdfAdapter — 用 DuckDuckGo 搜 PDF 并在下载时提取正文（A-6 的搜索/解析侧）。

设计要点：

- 只做 ``filetype:pdf`` 的检索 + 下载时抽正文，正文抽不出来就降级存原始 PDF
  （PRD §9 风险表："解析失败降级为仅下载不解析，不阻塞主流程"）；
- 网络全部经 ``HttpClient``（铁律 #2），失败返回 ``DownloadResult(success=False)``
  而不是抛异常（PRD §2.5）；
- DuckDuckGo 的 ``href`` 常是 ``/l/?uddg=<URL 编码目标>`` 跳转包装，必须解出真地址，
  否则存进 ``SearchResult.url`` 会让 M4 的去重键失效、下载多绕一跳。
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup
from loguru import logger
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.utils import sanitize_filename, sha256_of, unique_path


def resolve_duckduckgo_url(href: str) -> str:
    """把 DuckDuckGo 的跳转链接还原成真实目标地址。

    Args:
        href: 搜索结果里的 ``href``，可能是绝对地址、``//`` 开头的协议相对地址，
            或 ``/l/?uddg=<urlencoded>`` 形式的跳转包装。

    Returns:
        解包后的真实地址；没有 ``uddg`` 参数时原样返回。
    """
    if not href:
        return href
    candidate = f"https:{href}" if href.startswith("//") else href
    target = parse_qs(urlparse(candidate).query).get("uddg")
    if target:
        return unquote(target[0])
    return href


class PdfAdapter(SearchAdapter):
    """PDF 文档搜索适配器：搜 ``filetype:pdf``，下载时抽正文，失败降级存原始 PDF。"""

    name: str = "pdf"
    rate_limit_qps: float = 0.5
    _MIN_TEXT_LENGTH = 20

    def __init__(self, http_client: HttpClient | None = None) -> None:
        """初始化 PDF 适配器。

        Args:
            http_client: 项目统一 HTTP 客户端；``None`` 时按本适配器的限流自建。
        """
        self.http = http_client or HttpClient(rate_limit_qps=self.rate_limit_qps)
        self.search_base = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """检索指定关键词的 PDF 文档。

        Args:
            query: 标准化查询对象。

        Returns:
            符合 SearchResult 契约的结果列表（最多 ``query.limit`` 条）；网络失败
            时返回空列表并打 warning，不抛异常。
        """
        params = {"q": f"{query.raw_query} filetype:pdf", "kl": "cn-zh"}
        try:
            response = await self.http.get(self.search_base, params=params)
        except Exception as exc:  # noqa: BLE001 — 单源失败不能中断整次检索
            logger.warning("PdfAdapter.search 失败: {}", exc)
            return []

        results = self._parse_html(response.text)
        logger.debug("PdfAdapter.search 返回 {} 条结果", len(results))
        return results[: query.limit]

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """下载 PDF 并把正文抽成 ``.txt``；抽取失败则保存原始 PDF。

        Args:
            result: 单条搜索结果（使用 ``result.url``）。
            dest_dir: 目标目录，不存在时自动创建。

        Returns:
            :class:`DownloadResult`；失败时 ``success=False`` 并带原因，绝不抛异常。
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = await self.http.get(result.url, follow_redirects=True)
        except Exception as exc:  # noqa: BLE001 — 报错而不是抛出
            logger.warning("PdfAdapter.download 失败 {}: {}", result.url, exc)
            return DownloadResult(success=False, error=str(exc))

        data = response.content
        text = await self._extract_text(data)
        stem = sanitize_filename(result.title)

        if text and len(text.strip()) >= self._MIN_TEXT_LENGTH:
            target = unique_path(dest_dir, f"{stem}.txt")
            await asyncio.to_thread(target.write_text, text, encoding="utf-8")
            return DownloadResult(
                success=True,
                local_path=target,
                size_bytes=len(text.encode("utf-8")),
                sha256=sha256_of(text.encode("utf-8")),
            )

        target = unique_path(dest_dir, f"{stem}.pdf")
        await asyncio.to_thread(target.write_bytes, data)
        logger.debug("PdfAdapter 正文提取失败，降级保存原始 PDF: {}", target)
        return DownloadResult(
            success=True,
            local_path=target,
            size_bytes=len(data),
            sha256=sha256_of(data),
        )

    # ----------------------------------------------------------------- helpers
    def _parse_html(self, html: str) -> list[SearchResult]:
        """解析 DuckDuckGo 结果页，抽出标题 / 真实地址 / 摘要。"""
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
        for item in soup.select(".result"):
            title_elem = item.select_one(".result__a")
            if not title_elem:
                continue
            url = resolve_duckduckgo_url(title_elem.get("href", ""))
            if not url or url.startswith("/"):
                continue
            snippet_elem = item.select_one(".result__snippet")
            results.append(
                SearchResult(
                    title=title_elem.get_text(strip=True),
                    url=url,
                    abstract=snippet_elem.get_text(strip=True) if snippet_elem else "",
                    source=self.name,
                    extra={},
                )
            )
        return results

    async def _extract_text(self, pdf_bytes: bytes) -> str:
        """在阻塞线程里用 pypdf 抽取正文；损坏文件返回空串而不是抛出。"""

        def _extract(data: bytes) -> str:
            reader = PdfReader(io.BytesIO(data))
            parts: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    parts.append(page_text.strip())
            return "\n\n".join(parts)

        try:
            return await asyncio.to_thread(_extract, pdf_bytes)
        except (PdfReadError, ValueError, OSError) as exc:
            logger.warning("PDF 正文提取失败（降级保存原始文件）: {}", exc)
            return ""
