"""LLM client for the optional Phase 2 LLM features (PRD §2.1 F1.5/F1.6).

The LLM is an *optional upgrade*, never a hard dependency: Phase 1 works fully
with the rule-based parser, and so does the CLI when no API key is configured
(PRD "关键设计决策 #2" / 风险表 "Phase 2 LLM 解析成本").

Every request goes through :class:`~retriever.http.HttpClient`, so iron rule #2
(PRD §4.3) keeps holding: rate limiting, retries, proxy support and request
logging are inherited instead of re-implemented in a second network stack.

The client speaks the OpenAI-compatible ``/chat/completions`` dialect, which
DeepSeek, Moonshot and most self-hosted gateways expose, so swapping providers
is a ``.env`` change rather than a code change::

    LLM_API_KEY=sk-...
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_MODEL=deepseek-chat
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Self

from loguru import logger

from .config import LLMConfig, resolve_llm_api_key, resolve_llm_config
from .http import HttpClient

# A chat message in the OpenAI wire format.
Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the LLM is not configured or returns an unusable response.

    Callers treat this as a *recoverable* failure: the intent parser falls back
    to the rule-based implementation instead of aborting the CLI run.
    """


class LLMClient:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        http: HttpClient | None = None,
    ) -> None:
        """Initialize the client without performing any network I/O.

        Args:
            api_key: Explicit API key. ``None`` resolves ``LLM_API_KEY`` /
                ``DEEPSEEK_API_KEY`` from the environment or ``.env``.
            config: Model and endpoint settings. ``None`` resolves them from
                ``config/settings.yaml`` merged with environment overrides.
            http: Shared :class:`HttpClient`; one is created lazily on the
                first call when omitted (mainly injected in tests).
        """
        self._config = config or resolve_llm_config()
        resolved = resolve_llm_api_key() if api_key is None else api_key
        self._api_key = (resolved or "").strip()
        self._http = http
        self._owns_http = False

    # ------------------------------------------------------------------ state
    @property
    def config(self) -> LLMConfig:
        """The effective model/endpoint configuration."""
        return self._config

    @property
    def model(self) -> str:
        """Model name sent with every request."""
        return self._config.model

    @property
    def base_url(self) -> str:
        """Normalized endpoint base URL (no trailing slash)."""
        return self._config.base_url.rstrip("/")

    @property
    def configured(self) -> bool:
        """Whether an API key is available (i.e. the LLM can actually be used)."""
        return bool(self._api_key)

    # ------------------------------------------------------------------- calls
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Run one chat-completion round trip and return the assistant text.

        Args:
            messages: Conversation messages in the OpenAI wire format.
            temperature: Sampling temperature; defaults to ``llm.temperature``.
            max_tokens: Answer length cap; defaults to ``llm.max_tokens``.
            json_mode: Ask the provider to constrain the answer to JSON
                (``response_format={"type": "json_object"}``).

        Returns:
            The assistant message content (never ``None``; empty string is
            turned into an :class:`LLMError` because it is unusable).

        Raises:
            LLMError: When the key is missing, the request fails, or the
                response does not follow the chat-completions contract.
        """
        if not self.configured:
            raise LLMError(
                "LLM API key is not configured; set LLM_API_KEY (or DEEPSEEK_API_KEY) in .env"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": self._config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.debug("LLM request: model={} messages={}", self.model, len(payload["messages"]))
        try:
            response = await self._http_client().request(
                "POST", url, json=payload, headers=headers
            )
        except Exception as exc:  # every transport failure is one LLMError
            raise LLMError(f"LLM request to {url} failed: {exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"LLM returned a non-JSON response from {url}") from exc

        content = self._extract_content(body)
        if not content.strip():
            raise LLMError("LLM returned an empty answer")
        logger.debug("LLM answer: {} chars", len(content))
        return content

    def chat_sync(self, messages: Sequence[Message], **kwargs: Any) -> str:
        """Blocking wrapper around :meth:`chat` for the synchronous CLI flow."""
        return asyncio.run(self.chat(messages, **kwargs))

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None
            self._owns_http = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ----------------------------------------------------------------- helpers
    def _http_client(self) -> HttpClient:
        """Return the shared HTTP client, creating it on first use."""
        if self._http is None:
            self._http = HttpClient(
                rate_limit_qps=self._config.rate_limit_qps,
                max_retries=self._config.max_retries,
                timeout_seconds=float(self._config.timeout_seconds),
            )
            self._owns_http = True
        return self._http

    @staticmethod
    def _extract_content(body: Any) -> str:
        """Pull ``choices[0].message.content`` out of a chat-completions body.

        Args:
            body: Decoded JSON response body.

        Returns:
            The assistant content string (may be empty).

        Raises:
            LLMError: When the body does not follow the expected shape.
        """
        if not isinstance(body, dict):
            raise LLMError("LLM response is not a JSON object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM response carries no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("LLM response carries no message content")
        return content
