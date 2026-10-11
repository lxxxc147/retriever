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
from .base import SearchAdapter

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# Reuse project-standard filename sanitization logic
_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


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

        Supports keyword, author, category, and submitted-date range filters.

        Args:
            query: Structured search task.

        Returns:
            List of SearchResult, capped at query.limit.
        """
        query_parts: list[str] = []

        # Keywords
        if query.keywords:
            keyword_str = " AND ".join(f"all:{kw}" for kw in query.keywords)
            query_parts.append(keyword_str)

        # Author filter
        if query.filters.authors:
            author_str = " AND ".join(f'au:"{a}"' for a in query.filters.authors)
            query_parts.append(author_str)

        # Category filter
        categories = query.filters.categories if hasattr(query.filters, "categories") else None
        if categories:
            cat_str = " OR ".join(f"cat:{c}" for c in categories)
            query_parts.append(f"({cat_str})")

        # Year range (submitted date)
        year_from = query.filters.year_from
        year_to = query.filters.year_to
        if year_from or year_to:
            start = f"{year_from}-01-01" if year_from else "1900-01-01"
            end = f"{year_to}-12-31" if year_to else "2099-12-31"
            query_parts.append(f"submittedDate:[{start} TO {end}]")

        search_query = " AND ".join(query_parts) if query_parts else "all:*"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": query.limit,
        }

        try:
            response = await self._http.get(
                "http://export.arxiv.org/api/query",
                params=params,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ArxivAdapter.search failed: {}", exc)
            return []

        results = self._parse_xml(response.text)
        logger.debug("ArxivAdapter.search returned {} results", len(results))
        return results[: query.limit]

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
            arxiv_id = entry.findtext(f"{ARXIV_NS}id", "").strip()

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
