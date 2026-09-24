"""Tests for the unified HttpClient. 测试不打真实 API，全部 mock。"""

from unittest.mock import AsyncMock

import httpx
import pytest
import respx
import tenacity

from retriever.http import HttpClient


@pytest.fixture
def fast_http() -> HttpClient:
    """HttpClient with a generous rate limit so tests never really wait."""
    return HttpClient(rate_limit_qps=1000.0, max_retries=3)


@respx.mock
async def test_successful_request(fast_http: HttpClient) -> None:
    """① A 200 response is returned as-is."""
    respx.get("https://example.org/api/search").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    response = await fast_http.get("https://example.org/api/search")
    assert response.status_code == 200


@respx.mock
async def test_retry_on_503_then_success(
    fast_http: HttpClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """② A 503 is retried with backoff and a later 200 succeeds."""
    # Keep the test fast: replace exponential backoff with no waiting.
    monkeypatch.setattr(
        "retriever.http.wait_exponential",
        lambda *args, **kwargs: tenacity.wait_none(),
    )
    route = respx.get("https://example.org/flaky")
    route.side_effect = [httpx.Response(503), httpx.Response(200, text="ok")]

    response = await fast_http.get("https://example.org/flaky")

    assert response.status_code == 200
    assert route.call_count == 2  # first 503, second 200


@respx.mock
async def test_404_not_retried(fast_http: HttpClient) -> None:
    """③ A 404 (non-retryable 4xx) raises HTTPStatusError immediately."""
    route = respx.get("https://example.org/missing").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await fast_http.get("https://example.org/missing")
    assert route.call_count == 1


@respx.mock
async def test_per_host_rate_limit(respx_mock: respx.MockRouter) -> None:
    """④ Two sequential requests to one host wait at least 1/qps in between."""
    respx_mock.get("https://example.org/a").mock(return_value=httpx.Response(200))
    respx_mock.get("https://example.org/b").mock(return_value=httpx.Response(200))

    sleeper = AsyncMock()  # records requested sleeps without actually waiting
    client = HttpClient(rate_limit_qps=2.0, sleeper=sleeper)

    await client.get("https://example.org/a")  # first request: no pacing
    await client.get("https://example.org/b")  # second request: paced

    assert sleeper.call_count == 1
    requested_wait = sleeper.call_args.args[0]
    assert 0 < requested_wait <= 0.5  # 1 / 2 qps, minus elapsed time
