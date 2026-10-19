"""Tests for the OpenAI-compatible LLM client.

All HTTP is mocked with ``respx`` — no test here touches a real endpoint, and
every client is built with an explicit key/config so ambient ``.env`` values
cannot leak into the assertions (PRD §7.4).
"""

import json

import httpx
import pytest
import respx

from retriever.config import LLMConfig
from retriever.http import HttpClient
from retriever.llm import LLMClient, LLMError

BASE = "https://llm.test/v1"
ENDPOINT = f"{BASE}/chat/completions"


def _client(api_key: str = "sk-test", **overrides: object) -> LLMClient:
    """Build a client with a fast HTTP stub-friendly transport and fixed config."""
    config = LLMConfig(base_url=BASE, model="test-model", **overrides)  # type: ignore[arg-type]
    http = HttpClient(rate_limit_qps=1000.0, max_retries=1)
    return LLMClient(api_key=api_key, config=config, http=http)


def _answer(content: str) -> dict:
    """A minimal chat-completions response body."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@respx.mock
async def test_chat_sends_bearer_token_and_model() -> None:
    """① The request carries the key, the model name, and the messages."""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_answer("收到")))

    result = await _client().chat([{"role": "user", "content": "hi"}])

    assert result == "收到"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@respx.mock
async def test_json_mode_requests_json_response_format() -> None:
    """② ``json_mode=True`` asks the provider to constrain the answer to JSON."""
    route = respx.post(ENDPOINT).mock(
        return_value=httpx.Response(200, json=_answer('{"intent": "paper"}'))
    )

    await _client().chat([{"role": "user", "content": "x"}], json_mode=True)

    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}


@respx.mock
async def test_temperature_and_max_tokens_override_config() -> None:
    """③ Explicit call arguments win over the configured defaults."""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_answer("ok")))

    await _client().chat([{"role": "user", "content": "x"}], temperature=0.7, max_tokens=64)

    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 64


async def test_missing_api_key_raises_without_network() -> None:
    """④ Without a key the client fails fast and performs no request."""
    with pytest.raises(LLMError, match="API key"):
        await _client(api_key="").chat([{"role": "user", "content": "x"}])


@respx.mock
async def test_http_error_becomes_llm_error() -> None:
    """⑤ A non-retryable HTTP status is wrapped in :class:`LLMError`."""
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(401))

    with pytest.raises(LLMError, match="failed"):
        await _client().chat([{"role": "user", "content": "x"}])
    assert route.call_count == 1  # 401 is never retried


@respx.mock
async def test_non_json_body_becomes_llm_error() -> None:
    """⑥ A 200 answer that is not JSON is reported as an :class:`LLMError`."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, text="<html>oops</html>"))

    with pytest.raises(LLMError, match="non-JSON"):
        await _client().chat([{"role": "user", "content": "x"}])


@respx.mock
async def test_unexpected_shape_becomes_llm_error() -> None:
    """⑦ A response without ``choices[0].message.content`` is rejected."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json={"choices": []}))

    with pytest.raises(LLMError, match="no choices"):
        await _client().chat([{"role": "user", "content": "x"}])


@respx.mock
async def test_empty_answer_becomes_llm_error() -> None:
    """⑧ An empty assistant message is unusable and reported as an error."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_answer("   ")))

    with pytest.raises(LLMError, match="empty"):
        await _client().chat([{"role": "user", "content": "x"}])


@respx.mock
def test_chat_sync_drives_the_cli_flow() -> None:
    """⑨ ``chat_sync`` is the blocking entry point used by the intent parser."""
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_answer("ok")))

    assert _client().chat_sync([{"role": "user", "content": "x"}]) == "ok"


def test_configured_reflects_the_key_and_model_is_exposed() -> None:
    """⑩ ``configured``/``model`` describe the client without any I/O."""
    client = _client()
    assert client.configured is True
    assert client.model == "test-model"
    assert client.base_url == BASE
    assert _client(api_key="  ").configured is False
