"""Tests for the LLM summarizer. The model client is faked — no network."""

import pytest

from retriever.llm import LLMError
from retriever.models import SearchResult
from retriever.summarizer import LLMSummarizer


class _FakeClient:
    """Stand-in for :class:`retriever.llm.LLMClient` with a canned answer."""

    def __init__(self, answer: str | None = None, error: Exception | None = None,
                 configured: bool = True) -> None:
        self.answer = answer
        self.error = error
        self._configured = configured
        self.model = "fake-model"
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def chat_sync(self, messages: list[dict], **kwargs: object) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        assert self.answer is not None
        return self.answer


def _result(**overrides: object) -> SearchResult:
    """Build a search result for summarizer tests."""
    data: dict = {
        "source": "arxiv",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex "
                    "recurrent or convolutional neural networks.",
        "url": "https://arxiv.org/abs/1706.03762",
    }
    data.update(overrides)
    return SearchResult(**data)  # type: ignore[arg-type]


def test_single_batched_call_with_json_mode() -> None:
    """All results go into ONE call, and JSON mode is requested."""
    fake = _FakeClient('{"items": [{"i": 1, "s": "提出 Transformer 架构"}]}')
    summarizer = LLMSummarizer(client=fake)

    summaries = summarizer([_result(), _result(title="Second paper")])

    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"] == {"json_mode": True}
    prompt = fake.calls[0]["messages"][1]["content"]
    assert "1. [arxiv] Attention Is All You Need" in prompt
    assert "2. [arxiv] Second paper" in prompt
    assert summaries[0] == "提出 Transformer 架构"
    assert summaries[1]  # the un-answered item falls back to an extractive summary


def test_summaries_are_mapped_by_index() -> None:
    """The ``i`` field decides which result a summary belongs to."""
    fake = _FakeClient('{"items": [{"i": 2, "s": "第二个"}, {"i": 1, "s": "第一个"}]}')
    summaries = LLMSummarizer(client=fake)([_result(), _result(title="Second")])

    assert summaries == ["第一个", "第二个"]


def test_out_of_range_and_malformed_items_are_ignored() -> None:
    """Hallucinated indices and junk entries cannot corrupt the mapping."""
    answer = (
        '{"items": [{"i": 9, "s": "越界"}, {"i": "x", "s": "坏索引"}, '
        '{"i": 1, "s": "  有效摘要 "}, "junk"]}'
    )
    summaries = LLMSummarizer(client=_FakeClient(answer))([_result()])
    assert summaries[0] == "有效摘要"


def test_fenced_json_answer_is_accepted() -> None:
    """Markdown fences around the JSON do not break parsing."""
    answer = '```json\n{"items": [{"i": 1, "s": "带围栏"}]}\n```'
    assert LLMSummarizer(client=_FakeClient(answer))([_result()])[0] == "带围栏"


def test_missing_items_key_falls_back() -> None:
    """An answer without ``items`` degrades to extractive summaries."""
    summaries = LLMSummarizer(client=_FakeClient('{"other": 1}'))([_result()])
    assert summaries[0].startswith("The dominant sequence")


def test_llm_error_falls_back_to_extractive() -> None:
    """A dead endpoint never blocks the run."""
    summaries = LLMSummarizer(client=_FakeClient(error=LLMError("timeout")))([_result()])
    assert summaries[0].startswith("The dominant sequence")


def test_unconfigured_client_makes_no_call() -> None:
    """With no API key the summarizer does not even try to call the model."""
    fake = _FakeClient(configured=False)
    summaries = LLMSummarizer(client=fake)([_result()])

    assert fake.calls == []
    assert summaries[0].startswith("The dominant sequence")


def test_summary_is_cleaned_and_truncated() -> None:
    """Leading numbering/quotes are stripped and the length is capped."""
    answer = '{"items": [{"i": 1, "s": "1. “这是一个很长很长的摘要，需要被截断处理的情况”"}]}'
    summary = LLMSummarizer(client=_FakeClient(answer), max_chars=10)([_result()])[0]

    assert not summary.startswith(("1.", "“"))
    assert len(summary) <= 10


def test_max_items_bounds_the_prompt() -> None:
    """Only ``max_items`` results are sent; the rest fall back extractively."""
    fake = _FakeClient('{"items": [{"i": 1, "s": "只总结第一条"}]}')
    summarizer = LLMSummarizer(client=fake, max_items=1)

    summaries = summarizer([_result(title="First"), _result(title="Second")])

    prompt = fake.calls[0]["messages"][1]["content"]
    assert "2. [arxiv] Second" not in prompt
    assert summaries[0] == "只总结第一条"
    assert summaries[1].startswith("The dominant sequence")


def test_empty_input_returns_empty_and_calls_nothing() -> None:
    """Summarizing nothing performs no request."""
    fake = _FakeClient('{"items": []}')
    assert LLMSummarizer(client=fake)([]) == []
    assert fake.calls == []


@pytest.mark.parametrize("abstract", ["", "   ", "短摘要"])
def test_short_or_missing_abstract_is_handled(abstract: str) -> None:
    """Empty or very short abstracts still produce a usable one-liner."""
    fake = _FakeClient('{"items": []}')
    summary = LLMSummarizer(client=fake)([_result(abstract=abstract)])[0]
    assert summary
    assert len(summary) <= 60
