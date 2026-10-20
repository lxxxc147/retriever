"""Abstract base class for all search adapters (PRD §4.3).

Every data source (arXiv, GitHub, Scholar, Web, ...) implements
:class:`SearchAdapter`. Adapters:

- must NOT import each other (data flows only via ``models.py``),
- must NOT perform raw HTTP calls (use the shared ``HttpClient``),
- declare their own ``rate_limit_qps`` so the framework can pace requests.

Adapters report *classified* failures through :class:`AdapterError` so the CLI
can tell the user "限流 / 鉴权 / 网络不可达" instead of a raw traceback
(PRD §8.1 V5, §2.7 F7.5).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from ..models import DownloadResult, SearchQuery, SearchResult

# Failure kinds that are worth retrying with a looser query.
RETRYABLE_KINDS = frozenset({"network", "parse", "unknown"})


class AdapterError(RuntimeError):
    """A data-source failure with a user-readable classification.

    Attributes:
        source: Adapter name (``arxiv`` / ``github`` / ...).
        message: What went wrong, phrased for the user.
        kind: One of ``auth``, ``rate_limit``, ``network``, ``not_found``,
            ``parse``, ``unknown``.
    """

    KINDS = ("auth", "rate_limit", "network", "not_found", "parse", "unknown")

    def __init__(self, source: str, message: str, kind: str = "unknown") -> None:
        """Build a classified adapter failure.

        Args:
            source: Adapter name.
            message: User-facing explanation.
            kind: Failure class, used by callers to decide whether to retry.
        """
        super().__init__(message)
        self.source = source
        self.message = message
        self.kind = kind

    def __str__(self) -> str:
        """Render as ``<message>（<kind>）`` for the CLI warning line."""
        return f"{self.message}（{self.kind}）"


def classify_http_error(exc: BaseException, source: str) -> AdapterError:
    """Turn a transport/HTTP exception into a classified :class:`AdapterError`.

    Args:
        exc: The exception raised by :class:`~retriever.http.HttpClient`.
        source: Adapter name, recorded on the error.

    Returns:
        An :class:`AdapterError` whose ``kind`` says whether retrying could help.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return AdapterError(source, f"鉴权失败或访问受限（HTTP {status}）", "auth")
        if status == 429:
            return AdapterError(source, "触发源站限流（HTTP 429）", "rate_limit")
        if status == 404:
            return AdapterError(source, f"资源不存在（HTTP {status}）", "not_found")
        if status >= 500:
            return AdapterError(source, f"源站暂时不可用（HTTP {status}）", "network")
        return AdapterError(source, f"请求被拒绝（HTTP {status}）", "unknown")
    if isinstance(exc, httpx.TimeoutException):
        return AdapterError(source, "请求超时", "network")
    if isinstance(exc, httpx.TransportError):
        return AdapterError(source, "网络不可达", "network")
    return AdapterError(source, str(exc) or exc.__class__.__name__, "unknown")


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
