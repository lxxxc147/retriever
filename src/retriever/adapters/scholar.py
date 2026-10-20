"""ScholarAdapter — Google Scholar 检索（A-4：引用数、发表年份、来源期刊）。

抓取 ``scholar.google.com`` 能拿到引用数/年份/venue，但反爬严格且违反其 ToS
（PRD §3 NF6）：本适配器只做检索、**不支持下载**（``download`` 返回失败结果，
由 arXiv/PDF 适配器负责取全文），限流 0.2 QPS，429/403 时给出可识别告警。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from retriever.adapters.base import SearchAdapter
from retriever.http import HttpClient
from retriever.models import DownloadResult, SearchQuery, SearchResult


class ScholarRateLimitError(Exception):
    """Google Scholar 限流异常，可被 Router 识别并降级。"""


class ScholarBlockedError(Exception):
    """Google Scholar 封禁异常，可被 Router 识别并降级。"""


class ScholarAdapter(SearchAdapter):
    """Google Scholar 学术搜索适配器：论文标题、作者、年份检索与引用数提取。"""

    name: str = "scholar"
    rate_limit_qps: float = 0.2

    def __init__(self, http_client: HttpClient | None = None) -> None:
        """初始化适配器。

        Args:
            http_client: 项目统一 HTTP 客户端；``None`` 时按本适配器的限流自建。
        """
        self.http_client = http_client or HttpClient(rate_limit_qps=self.rate_limit_qps)
        self._base_url = "https://scholar.google.com/scholar"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行学术搜索，返回标准化结果，``extra`` 含 citations / venue / year。

        Args:
            query: 结构化查询对象。

        Returns:
            SearchResult 列表（最多 ``query.limit`` 条）。

        Raises:
            ScholarRateLimitError: 触发限流（HTTP 429）。
            ScholarBlockedError: IP 被封禁（HTTP 403）。
        """
        params = {"q": query.raw_query, "hl": "zh-CN", "num": query.limit}

        try:
            response = await self.http_client.get(self._base_url, params=params)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 429:
                raise ScholarRateLimitError("Google Scholar 触发限流，请稍后重试") from exc
            if status_code == 403:
                raise ScholarBlockedError("Google Scholar 访问被封禁") from exc
            logger.warning("ScholarAdapter 请求失败: {}", exc)
            return []
        except Exception as exc:  # noqa: BLE001 — 单源失败不能中断整次检索
            logger.warning("ScholarAdapter 请求失败: {}", exc)
            return []

        results = self._parse_html(response.text)
        logger.debug("ScholarAdapter.search 返回 {} 条结果", len(results))
        return results[: query.limit]

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Scholar 只做检索：下载交给 arXiv/PDF 适配器，这里返回失败而不抛异常。

        Args:
            result: 单条搜索结果（未使用）。
            dest_dir: 目标目录（未使用）。

        Returns:
            恒为 ``success=False`` 的结果，``error`` 说明去哪儿取全文。
        """
        return DownloadResult(
            success=False,
            error="scholar 仅支持检索；请用 arXiv adapter 或 PDF adapter 下载全文",
        )

    # ----------------------------------------------------------------- helpers
    def _parse_html(self, html: str) -> list[SearchResult]:
        """解析搜索结果页，提取标题 / 链接 / 摘要 / 引用数 / venue / 年份。"""
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []

        for item in soup.select(".gs_r.gs_or.gs_scl"):
            title_elem = item.select_one(".gs_rt a")
            if not title_elem:
                continue
            title = title_elem.get_text(strip=True)
            url = title_elem.get("href", "")
            abstract_elem = item.select_one(".gs_rs")
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""

            citations = 0
            venue = ""
            year = ""
            bottom_elem = item.select_one(".gs_a")
            if bottom_elem:
                bottom_text = bottom_elem.get_text()
                cite_match = re.search(r"被引用次数：\s*(\d+)", bottom_text) or re.search(
                    r"Cited by\s+(\d+)", bottom_text
                )
                if cite_match:
                    citations = int(cite_match.group(1))
                year_match = re.search(r"\b(19|20)\d{2}\b", bottom_text)
                if year_match:
                    year = year_match.group()
                parts = bottom_text.split(" - ")
                if len(parts) >= 2:
                    venue = parts[1].strip().split(",")[0].strip()

            results.append(
                SearchResult(
                    source=self.name,
                    title=title,
                    url=url,
                    abstract=abstract,
                    published_at=self._published_at(year),
                    extra={"citations": citations, "venue": venue, "year": year},
                )
            )

        return results

    @staticmethod
    def _published_at(year: str) -> datetime | None:
        """把页面上的年份转成 ``published_at``（M4 时效性打分需要它）。"""
        if not year.isdigit():
            return None
        try:
            return datetime(int(year), 1, 1, tzinfo=UTC)
        except ValueError:
            return None
