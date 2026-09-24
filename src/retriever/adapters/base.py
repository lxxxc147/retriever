"""Abstract base class for all search adapters (PRD §4.3).

Every data source (arXiv, GitHub, Scholar, Web, ...) implements
:class:`SearchAdapter`. Adapters:

- must NOT import each other (data flows only via ``models.py``),
- must NOT perform raw HTTP calls (use the shared ``HttpClient``),
- declare their own ``rate_limit_qps`` so the framework can pace requests.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import DownloadResult, SearchQuery, SearchResult


class SearchAdapter(ABC):
    """Contract every data-source adapter must fulfil."""

    name: str  # data-source identifier, e.g. "arxiv"
    rate_limit_qps: float  # max requests per second for this source

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search the data source and return unified results.

        Args:
            query: Structured search task (intent/keywords/filters/limit).

        Returns:
            A list of :class:`SearchResult`, at most ``query.limit`` items.
        """

    @abstractmethod
    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download the file referenced by a search result into ``dest_dir``.

        Args:
            result: The search result to download (uses ``download_url``).
            dest_dir: Destination directory; must be created if missing.

        Returns:
            A :class:`DownloadResult` — failures are reported in the result,
            not raised, so one bad file never crashes a batch.
        """
