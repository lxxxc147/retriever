"""One-sentence Chinese summaries for search results (PRD §2.4 F4.3).

The pipeline (``pipeline.summarize``) takes *any* callable that returns one
summary per result, so the LLM lives here instead of inside ``pipeline.py``:

- :class:`LLMSummarizer` batches every result of one search into a **single**
  chat call (cheaper and faster than N calls), asks for a strict JSON answer and
  falls back to the deterministic extractive summary when the model is not
  configured, fails, or returns something unusable;
- the prompt forbids inventing facts: a summary may only restate what the title
  and abstract already say.
"""

from __future__ import annotations

import re

from loguru import logger

from .llm import LLMClient, LLMError, extract_json_object
from .models import SearchResult
from .pipeline import SUMMARY_MAX_CHARS, fallback_summary

# Leading list markers a model sometimes keeps ("1. ", "1、", "- ", "• ").
_ENUMERATION_RE = re.compile(r"^\s*(?:\d+\s*[.、)）]|[-•*])\s*")

SUMMARY_SYSTEM_PROMPT = """You write one-line Chinese summaries of search results for a research assistant.

You receive a numbered list of results, each with its source, title and (possibly) abstract.
Return ONE JSON object, no other text:
{"items": [{"i": 1, "s": "一句话中文摘要"}]}

Rules:
- One sentence per result, at most 40 Chinese characters, no bullet points, no numbering.
- State what the work actually is or does (主题、方法、结论), not that it is "interesting".
- Never invent facts that are absent from the title/abstract. If the abstract is missing,
  summarize from the title only.
- Keep proper nouns and model names as written (Transformer, RAG, PyTorch).
- Cover EVERY input item exactly once, keyed by its number in "i"."""


class LLMSummarizer:
    """Batch summarizer backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        client: LLMClient | None = None,
        max_chars: int = SUMMARY_MAX_CHARS,
        max_items: int = 10,
        abstract_chars: int = 300,
    ) -> None:
        """Initialize the summarizer without any network I/O.

        Args:
            client: LLM client; ``None`` resolves the configured one.
            max_chars: Length cap applied to each returned summary.
            max_items: Maximum number of results per batch (and per call).
            abstract_chars: How much of each abstract to send to the model.
        """
        self._client = client or LLMClient()
        self.max_chars = max_chars
        self.max_items = max_items
        self.abstract_chars = abstract_chars

    @property
    def configured(self) -> bool:
        """Whether a key is available (i.e. real model summaries are possible)."""
        return self._client.configured

    def __call__(self, results: list[SearchResult]) -> list[str]:
        """Return one summary per result, in order.

        Never raises: any failure degrades to :func:`fallback_summary`.

        Args:
            results: The ranked results to summarize.

        Returns:
            Exactly ``len(results)`` summaries.
        """
        items = list(results)
        if not items:
            return []
        if not self.configured:
            return [fallback_summary(item, self.max_chars) for item in items]

        try:
            answer = self._client.chat_sync(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(items)},
                ],
                json_mode=True,
            )
            produced = self._parse(answer, len(items))
        except (LLMError, ValueError, TypeError) as exc:
            logger.warning("LLM summarization failed, using extractive summaries: {}", exc)
            produced = {}

        return [
            self._clean(produced.get(index)) or fallback_summary(item, self.max_chars)
            for index, item in enumerate(items, start=1)
        ]

    # ----------------------------------------------------------------- helpers
    def _build_prompt(self, items: list[SearchResult]) -> str:
        """Render the numbered result list sent to the model."""
        lines: list[str] = []
        for index, item in enumerate(items[: self.max_items], start=1):
            abstract = " ".join((item.abstract or "").split())[: self.abstract_chars]
            lines.append(f'{index}. [{item.source}] {item.title}')
            if abstract:
                lines.append(f"   abstract: {abstract}")
        return "\n".join(lines)

    @staticmethod
    def _parse(answer: str, count: int) -> dict[int, str]:
        """Extract ``{index: summary}`` from the model answer.

        Args:
            answer: Raw model output.
            count: Number of items asked about (indices outside 1..count are
                dropped so a hallucinated index cannot corrupt the mapping).

        Returns:
            Mapping of 1-based index to summary text.

        Raises:
            TypeError: When the answer carries no ``items`` list.
        """
        payload = extract_json_object(answer)
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise TypeError("LLM answer has no 'items' list")
        parsed: dict[int, str] = {}
        for entry in raw_items:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("i"))
            except (TypeError, ValueError):
                continue
            summary = entry.get("s")
            if isinstance(summary, str) and 1 <= index <= count and summary.strip():
                parsed[index] = summary.strip()
        return parsed

    def _clean(self, summary: str | None) -> str:
        """Trim a model summary to one line, drop enumeration, cap the length."""
        if not summary:
            return ""
        text = " ".join(summary.split()).strip().strip('"“”\'')
        text = _ENUMERATION_RE.sub("", text).strip().strip('"“”\'')
        return text[: self.max_chars]
