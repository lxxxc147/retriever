"""WebAdapter — DuckDuckGo 通用搜索 + trafilatura 正文提取（A-5）。

- 搜索：``html.duckduckgo.com/html/``（免费、无需 Key），解析结果页；
- 下载：抓页面 → trafilatura 提正文存 Markdown，提不出来降级存原始 HTML；
- DuckDuckGo 的 ``href`` 常是 ``/l/?uddg=<URL 编码目标>`` 跳转包装，必须解出真地址，
  否则 ``SearchResult.url`` 存的是包装串，M4 的跨源去重键会失效、下载也要多绕一跳。

网络全部经 ``HttpClient``（铁律 #2），失败走返回式错误（PRD §2.5）而不是抛异常。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import trafilatura
from bs4 import BeautifulSoup
from loguru import logger

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.utils import sanitize_filename, sha256_of, unique_path


def resolve_duckduckgo_url(href: str) -> str:
    """把 DuckDuckGo 的跳转链接还原成真实目标地址。

    与 ``pdf.py`` 里的同名函数是同一份逻辑：PRD 铁律 #1 禁止 adapter 之间互相 import，
    所以两个抓 DDG 的适配器各自持有一份（逻辑只有 6 行，不引入共享依赖）。

    Args:
        href: 结果页里的 ``href``，可能是绝对地址、``//`` 协议相对地址，
            或 ``/l/?uddg=<urlencoded>`` 跳转包装。

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


class WebAdapter(SearchAdapter):
    """通用网页搜索适配器：DuckDuckGo 检索 + trafilatura 正文提取。"""

    name: str = "web"
    rate_limit_qps: float = 0.5
    _MIN_CONTENT_LENGTH = 30

    def __init__(self, http_client: HttpClient | None = None) -> None:
        """初始化适配器。

        Args:
            http_client: 项目统一 HTTP 客户端；``None`` 时按本适配器的限流自建。
        """
        self.http_client = http_client or HttpClient(rate_limit_qps=self.rate_limit_qps)
        self.search_base = "https://html.duckduckgo.com/html/"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行通用网页搜索。

        Args:
            query: 标准化查询对象。

        Returns:
            符合 SearchResult 契约的结果列表（最多 ``query.limit`` 条）；网络失败时
            返回空列表并打 warning。
        """
        params = {"q": query.raw_query, "kl": "cn-zh"}
        try:
            response = await self.http_client.get(self.search_base, params=params)
        except Exception as exc:  # noqa: BLE001 — 单源失败不能中断整次检索
            logger.warning("WebAdapter 搜索失败: {}", exc)
            return []

        results = self._parse_html(response.text)
        logger.debug("WebAdapter.search 返回 {} 条结果", len(results))
        return results[: query.limit]

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """抓取网页并保存为 Markdown；正文提取失败则降级保存原始 HTML。

        Args:
            result: 单条搜索结果（使用 ``result.url``）。
            dest_dir: 目标目录，不存在时自动创建。

        Returns:
            :class:`DownloadResult`；失败时 ``success=False`` 并带原因。
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_filename(result.title)

        try:
            response = await self.http_client.get(result.url, follow_redirects=True)
        except Exception as exc:  # noqa: BLE001 — 报错而不是抛出
            logger.warning("WebAdapter 下载失败 {}: {}", result.url, exc)
            return DownloadResult(success=False, error=str(exc))

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["nav", "footer", "header", "aside", "script", "style", "iframe"]):
            tag.decompose()

        markdown = trafilatura.extract(
            str(soup),
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=False,
            fast=True,
        )
        if markdown and len(markdown.strip()) >= self._MIN_CONTENT_LENGTH:
            target = unique_path(dest_dir, f"{stem}.md")
            target.write_text(markdown, encoding="utf-8")
            return DownloadResult(
                success=True,
                local_path=target,
                size_bytes=len(markdown.encode("utf-8")),
                sha256=sha256_of(markdown.encode("utf-8")),
            )

        target = unique_path(dest_dir, f"{stem}.html")
        target.write_text(html_content, encoding="utf-8")
        logger.debug("WebAdapter 正文提取失败，降级保存原始 HTML: {}", target)
        return DownloadResult(
            success=True,
            local_path=target,
            size_bytes=len(html_content.encode("utf-8")),
            sha256=sha256_of(html_content.encode("utf-8")),
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
