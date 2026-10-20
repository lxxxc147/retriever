"""ArxivAdapter - arXiv data source adapter.

Supports keyword/author/category/year filtering, PDF download,
and direct ID/URL lookup (skips search flow).
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..http import HttpClient
from ..models import DownloadResult, SearchQuery, SearchResult
from .base import RETRYABLE_KINDS, AdapterError, SearchAdapter, classify_http_error

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Reuse project-standard filename sanitization logic
_SAFE_CHARS = re.compile(r"[^0-9A-Za-z一-鿿._-]+")


def sanitize_filename(title: str, max_len: int = 80) -> str:
    """Make a file name safe: keep alphanumerics/CJK/._-, spaces to "_", truncate.

    Args:
        title: Raw title with potentially illegal characters.
        max_len: Maximum length before extension.

    Returns:
        Sanitized filename-safe string.
    """
    name = title.replace(" ", "_")
    name = _SAFE_CHARS.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name[:max_len] or "untitled"


class ArxivAdapter(SearchAdapter):
    """arXiv data source adapter implementing SearchAdapter contract."""

    name: str = "arxiv"
    rate_limit_qps: float = 1.0

    def __init__(self, http: HttpClient | None = None) -> None:
        """Initialize adapter with optional shared HTTP client.

        Args:
            http: Shared HttpClient instance; created on demand if None.
        """
        self._http = http or HttpClient(rate_limit_qps=self.rate_limit_qps)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search arXiv API and return unified SearchResult list.

        过滤条件是**逐级降级**的：arXiv 对 `au:"作者" + submittedDate` 这类组合会直接
        返回 500，或返回 0 条。遇到这种情况就自动放宽条件重试（去年份 → 去作者 →
        只留关键词），而不是让用户拿到空结果（PRD §2.1 F1.2 / §9 风险表）。

        Args:
            query: Structured search task.

        Returns:
            List of SearchResult, capped at query.limit.

        Raises:
            AdapterError: 鉴权/限流/资源不存在这类"重试也没用"的失败，携带可读原因。
        """
        last_error: AdapterError | None = None

        for label, search_query in self._query_variants(query):
            params = {"search_query": search_query, "start": 0, "max_results": query.limit}
            try:
                response = await self._http.get(ARXIV_API_URL, params=params)
            except Exception as exc:  # 先分类，再决定是否降级重试
                error = classify_http_error(exc, self.name)
                logger.warning("ArxivAdapter.search 失败（{}）: {}", label, error)
                if error.kind not in RETRYABLE_KINDS:
                    raise error from exc
                last_error = error
                continue

            last_error = None  # 请求成功
            results = self._parse_xml(response.text)[: query.limit]
            if results:
                if label != "完整条件":
                    logger.info("arXiv 放宽为「{}」后拿到 {} 条结果", label, len(results))
                return results

        if last_error is not None:
            raise last_error
        logger.debug("ArxivAdapter.search 返回 0 条结果")
        return []

    def _query_variants(self, query: SearchQuery) -> list[tuple[str, str]]:
        """构造"完整 → 去年份 → 去作者 → 仅关键词"的降级查询序列。

        Args:
            query: 用户的结构化查询。

        Returns:
            ``(说明, search_query)`` 列表，顺序即降级顺序；完全相同的查询只保留一次。
        """
        keyword_part = [f"all:{keyword}" for keyword in query.keywords] or ["all:*"]
        author_part = [f'au:"{author}"' for author in (query.filters.authors or [])]

        categories = query.filters.categories or []
        category_part: list[str] = []
        if categories:
            cat_str = " OR ".join(f"cat:{code}" for code in categories)
            category_part = [f"({cat_str})"]

        year_part: list[str] = []
        year_from = query.filters.year_from
        year_to = query.filters.year_to
        if year_from or year_to:
            start = f"{year_from}-01-01" if year_from else "1900-01-01"
            end = f"{year_to}-12-31" if year_to else "2099-12-31"
            year_part = [f"submittedDate:[{start} TO {end}]"]

        variants: list[tuple[str, str]] = [
            ("完整条件", " AND ".join([*keyword_part, *author_part, *category_part, *year_part]))
        ]
        if year_part:
            variants.append(
                ("去掉年份过滤", " AND ".join([*keyword_part, *author_part, *category_part]))
            )
        if author_part:
            variants.append(
                ("去掉作者过滤", " AND ".join([*keyword_part, *category_part]))
            )
        if category_part:
            variants.append(("仅关键词", " AND ".join(keyword_part)))

        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for label, value in variants:
            if value in seen:
                continue
            seen.add(value)
            unique.append((label, value))
        return unique

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download PDF from result.download_url into dest_dir.

        Args:
            result: Search result with download_url.
            dest_dir: Destination directory, created if missing.

        Returns:
            DownloadResult; errors are returned as success=False, never raised.
        """
        if not result.download_url:
            return DownloadResult(success=False, error="no download_url on result")

        try:
            response = await self._http.get(result.download_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ArxivAdapter.download failed for {}: {}", result.url, exc)
            return DownloadResult(success=False, error=str(exc))

        content = response.content
        filename = sanitize_filename(result.title) + ".pdf"

        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / filename
        local_path.write_bytes(content)

        return DownloadResult(
            success=True,
            local_path=local_path,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def extract_arxiv_id(text: str) -> str | None:
        """Extract arXiv ID from plain text or URL.

        Supports:
        - New format: 2401.03462 / 2401.03462v1
        - Old format: cs/0601001
        - Full URL: /abs/ or /pdf/ links

        Args:
            text: Input text possibly containing an arXiv ID or URL.

        Returns:
            Clean arXiv ID string, or None if not found.
        """
        # Match URL pattern
        url_pattern = r"arxiv\.org/(?:abs|pdf)/([\w./-]+)"
        url_match = re.search(url_pattern, text)
        if url_match:
            return url_match.group(1).removesuffix(".pdf")

        # Match plain ID patterns
        id_pattern = r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[\w.-]+/\d{7})\b"
        id_match = re.search(id_pattern, text)
        if id_match:
            return id_match.group(1)

        return None

    async def get_by_id(self, arxiv_id: str) -> SearchResult | None:
        """Fetch a single paper directly by arXiv ID, skips search flow.

        Args:
            arxiv_id: Clean arXiv identifier.

        Returns:
            Single SearchResult, or None if not found.
        """
        params = {"id_list": arxiv_id, "max_results": 1}

        try:
            response = await self._http.get(
                "http://export.arxiv.org/api/query",
                params=params,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ArxivAdapter.get_by_id failed for {}: {}", arxiv_id, exc)
            return None

        results = self._parse_xml(response.text)
        return results[0] if results else None

    def _parse_xml(self, xml_text: str) -> list[SearchResult]:
        """Parse arXiv Atom XML response into SearchResult list.

        Args:
            xml_text: Raw XML string from arXiv API.

        Returns:
            Parsed list of SearchResult.
        """
        root = ET.fromstring(xml_text)
        entries = root.findall(f"{ATOM_NS}entry")
        results: list[SearchResult] = []

        for entry in entries:
            title = entry.findtext(f"{ATOM_NS}title", "").strip()
            abstract = entry.findtext(f"{ATOM_NS}summary", "").strip()
            published_str = entry.findtext(f"{ATOM_NS}published", "").strip()
            # <id> lives in the Atom namespace (NOT the arXiv extension
            # namespace), e.g. "http://arxiv.org/abs/1706.03762v7".
            arxiv_id = (
                entry.findtext(f"{ATOM_NS}id", "")
                .strip()
                .removesuffix("/")
                .rsplit("/abs/", maxsplit=1)[-1]
            )
            # Strip the version suffix ("1706.03762v7" -> "1706.03762").
            arxiv_id = arxiv_id.partition("v")[0]

            # Parse published datetime
            published_at: datetime | None = None
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str)
                except ValueError:
                    published_at = None

            # Authors
            authors = [
                author.findtext(f"{ATOM_NS}name", "").strip()
                for author in entry.findall(f"{ATOM_NS}author")
            ]

            # PDF link
            pdf_url = ""
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            # Primary categories
            categories = [
                cat.get("term", "")
                for cat in entry.findall(f"{ARXIV_NS}primary_category")
            ]

            results.append(
                SearchResult(
                    source=self.name,
                    title=title,
                    abstract=abstract,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    download_url=pdf_url,
                    published_at=published_at,
                    authors=authors,
                    extra={
                        "arxiv_id": arxiv_id,
                        "categories": categories,
                    },
                )
            )

        return results
