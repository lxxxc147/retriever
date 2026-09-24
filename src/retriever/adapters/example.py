"""ExampleAdapter — a minimal reference implementation of SearchAdapter.

New team members should copy this file as the starting point for their own
adapter (arXiv, GitHub, ...). It demonstrates:

- how to call the shared ``HttpClient`` (never raw httpx),
- how to parse a JSON response into ``SearchResult`` models,
- how to download bytes safely: sanitize the file name, create the
  destination directory, compute sha256 and size.

The search endpoint (``https://example.org/api/search``) is intentionally a
placeholder — the point is the *structure*, not the endpoint.
"""

import hashlib
import re
from pathlib import Path

from loguru import logger

from ..http import HttpClient
from ..models import DownloadResult, SearchQuery, SearchResult
from .base import SearchAdapter

SEARCH_URL = "https://example.org/api/search"
# Characters kept in sanitized file names: ASCII alphanumerics, CJK chars,
# dot, underscore, dash. Spaces become underscores. Result is capped at
# 80 characters.
_SAFE_CHARS = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def sanitize_filename(title: str, max_len: int = 80) -> str:
    """Make a file name safe: keep alphanumerics/CJK/._-, spaces to "_", truncate.

    Args:
        title: Raw title, potentially containing illegal characters.
        max_len: Maximum length of the sanitized name (before extension).

    Returns:
        A file-name-safe string.
    """
    name = title.replace(" ", "_")
    name = _SAFE_CHARS.sub("_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name[:max_len] or "untitled"


class ExampleAdapter(SearchAdapter):
    """Reference adapter implementation. See module docstring."""

    name: str = "example"
    rate_limit_qps: float = 2.0

    def __init__(self, http: HttpClient | None = None) -> None:
        """Initialize the adapter.

        Args:
            http: Optional shared HTTP client; one is created on demand.
        """
        self._http = http or HttpClient(rate_limit_qps=self.rate_limit_qps)

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search the (placeholder) endpoint and parse results.

        Args:
            query: Structured search task.

        Returns:
            Parsed list of :class:`SearchResult`, capped at ``query.limit``.
        """
        params = {"q": " ".join(query.keywords) or query.raw_query, "limit": query.limit}
        response = await self._http.get(SEARCH_URL, params=params)
        payload = response.json()
        results = [
            SearchResult(
                source=self.name,
                title=item.get("title", ""),
                abstract=item.get("summary", ""),
                url=item.get("url", ""),
                download_url=item.get("download_url"),
            )
            for item in payload.get("results", [])
        ]
        logger.debug("ExampleAdapter.search returned {} results", len(results))
        return results[: query.limit]

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download ``result.download_url`` into ``dest_dir`` with a safe name.

        Args:
            result: Search result whose ``download_url`` will be fetched.
            dest_dir: Destination directory (created if missing).

        Returns:
            A :class:`DownloadResult`; HTTP errors are reported as
            ``success=False`` instead of raising.
        """
        if not result.download_url:
            return DownloadResult(success=False, error="no download_url on result")

        try:
            response = await self._http.get(result.download_url)
        except Exception as exc:  # noqa: BLE001 — failures must not raise
            logger.warning("ExampleAdapter.download failed for {}: {}", result.url, exc)
            return DownloadResult(success=False, error=str(exc))

        content = response.content
        filename = sanitize_filename(result.title)
        # Preserve a meaningful extension from the URL path if present.
        url_suffix = Path(result.download_url.split("?")[0]).suffix
        if url_suffix and len(url_suffix) <= 10:
            filename += url_suffix.lower()

        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / filename
        local_path.write_bytes(content)

        return DownloadResult(
            success=True,
            local_path=local_path,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
