"""Unified HTTP client — the ONLY network exit of this project.

Iron rule (PRD §4.3 #2): adapters must never call ``httpx.get`` directly.
All requests go through :class:`HttpClient`, which provides:

- per-host rate limiting (simple token-bucket via ``asyncio.Lock`` +
  monotonic timestamps; no extra heavy dependencies),
- retries with exponential backoff via tenacity for 429 / 5xx / connection
  errors (never for other 4xx),
- full loguru logging of request URLs, retries, and failures.

Usage::

    client = HttpClient(rate_limit_qps=2.0, max_retries=3)
    resp = await client.get("https://example.org/paper.pdf")
"""

import asyncio
import time
from collections.abc import Callable
from typing import Self
from urllib.parse import urlparse

import httpx
from loguru import logger
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# 4xx statuses (other than 429) are permanent client errors — never retry.
NON_RETRYABLE_STATUSES = {400, 401, 403, 404, 405, 406, 410, 422, 451}


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception deserves a retry.

    Retries connection/transport errors, timeouts, HTTP 429 and 5xx.
    Does NOT retry other 4xx statuses (they fail permanently).
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in NON_RETRYABLE_STATUSES:
            return False
        return status == 429 or status >= 500
    return False


class HttpClient:
    """Async HTTP client with per-host rate limiting and retry + logging."""

    def __init__(
        self,
        rate_limit_qps: float = 2.0,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        sleeper: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        """Initialize the client.

        Args:
            rate_limit_qps: Maximum requests per second per host.
            max_retries: Maximum total attempts for retryable failures.
            timeout_seconds: Default request timeout.
            client: Optional pre-built ``httpx.AsyncClient`` (mainly for tests).
            sleeper: Async sleep callable used for rate-limit pacing
                (injectable so tests can avoid real waiting).
        """
        self.rate_limit_qps = rate_limit_qps
        self.max_retries = max_retries
        self._min_interval = 1.0 / rate_limit_qps if rate_limit_qps > 0 else 0.0
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            http2=True,
            follow_redirects=True,
        )
        self._owns_client = client is None
        self._sleeper = sleeper
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request_at: dict[str, float] = {}

    @staticmethod
    def _host_of(url: str) -> str:
        """Extract the normalized host key used for per-host rate limiting."""
        parsed = urlparse(url)
        return parsed.netloc.lower() or "default"

    def _lock_for(self, host: str) -> asyncio.Lock:
        """Return (creating if needed) the asyncio.Lock for a host."""
        if host not in self._locks:
            self._locks[host] = asyncio.Lock()
        return self._locks[host]

    async def _respect_rate_limit(self, host: str) -> None:
        """Enforce the minimum interval between requests to the same host."""
        if self._min_interval <= 0:
            return
        async with self._lock_for(host):
            last = self._last_request_at.get(host)
            now = time.monotonic()
            wait = (last + self._min_interval) - now if last is not None else 0.0
            if wait > 0:
                logger.debug("Rate limit: sleeping {:.2f}s before {}", wait, host)
                await self._sleeper(wait)
            self._last_request_at[host] = time.monotonic()

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        """Send a request with per-host rate limiting, retries, and logging.

        Retries up to ``max_retries`` attempts with exponential backoff when
        the failure is retryable (429 / 5xx / connection errors). Other 4xx
        statuses raise ``httpx.HTTPStatusError`` immediately without retry.

        Args:
            method: HTTP method, e.g. ``"GET"``.
            url: Request URL.
            **kwargs: Extra arguments forwarded to ``httpx.AsyncClient.request``.

        Returns:
            The successful ``httpx.Response`` (2xx after ``raise_for_status``).

        Raises:
            httpx.HTTPStatusError: For non-retryable HTTP error statuses.
            RetryError: When retryable failures persist after all attempts.
        """
        host = self._host_of(url)
        await self._respect_rate_limit(host)
        logger.debug("HTTP {} {}", method, url)

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception(_is_retryable),
            reraise=False,
        )
        async def _do_request() -> httpx.Response:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        try:
            return await _do_request()
        except RetryError as exc:
            logger.warning("HTTP request failed after retries: {} {}", method, url)
            raise exc.reraise() from exc

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        """Convenience wrapper around :meth:`request` for GET requests."""
        return await self.request("GET", url, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
