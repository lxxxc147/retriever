"""Tests for the LLM intent parser and its rule-based fallback.

The model is faked at the client boundary, so these tests are fully offline and
deterministic; the LLM's own HTTP behaviour is covered in ``test_llm.py``.
"""

import pytest
from pydantic import ValidationError

from retriever.intent import parse_intent
from retriever.intent_llm import (
    build_parsed_intent,
    clean_keywords,
    extract_json_object,
    parse_intent_auto,
    sanitize_filters,
)
from retriever.llm import LLMError
from retriever.models import Filters

PAPER_ANSWER = (
    '{"intent": "paper", "keywords": ["transformer"], '
    '"filters": {"authors": null, "year_from": null, "year_to": null, '
    '"language": null, "min_stars": null, "categories": null}, "limit": 5}'
)


class _FakeLLM:
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


# --------------------------------------------------------------------- parsing
def test_valid_answer_is_used_as_is() -> None:
    """① A well-formed answer yields the LLM's intent, keywords, and limit."""
    outcome = parse_intent_auto("帮我找 transformer 的论文", client=_FakeLLM(PAPER_ANSWER))

    assert outcome.source == "llm"
    assert outcome.note == "fake-model"
    assert outcome.fallback is False
    assert outcome.parsed.intent == "paper"
    assert outcome.parsed.keywords == ["transformer"]
    assert outcome.parsed.raw_query == "帮我找 transformer 的论文"
    assert outcome.parsed.limit == 5


def test_prompt_is_sent_with_json_mode() -> None:
    """② The system prompt and the raw query are sent, with JSON mode on."""
    fake = _FakeLLM(PAPER_ANSWER)
    parse_intent_auto("transformer survey", client=fake)

    messages = fake.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "transformer survey"}
    assert fake.calls[0]["kwargs"] == {"json_mode": True}


def test_fenced_and_prose_wrapped_json_are_accepted() -> None:
    """③ Markdown fences and surrounding prose do not break extraction."""
    fenced = f"好的，结果如下：\n```json\n{PAPER_ANSWER}\n```\n希望有帮助。"
    assert parse_intent_auto("x", client=_FakeLLM(fenced)).parsed.intent == "paper"

    prose = f"Sure — {PAPER_ANSWER} that's the plan."
    assert parse_intent_auto("x", client=_FakeLLM(prose)).parsed.keywords == ["transformer"]


def test_chinese_intent_alias_is_normalized() -> None:
    """④ A friendly Chinese intent label maps onto the contract enum."""
    answer = '{"intent": "仓库", "keywords": ["pytorch"], "filters": {}, "limit": 3}'
    outcome = parse_intent_auto("pytorch 仓库", client=_FakeLLM(answer))

    assert outcome.source == "llm"
    assert outcome.parsed.intent == "repo"


def test_limit_and_filters_are_sanitized() -> None:
    """⑤ Years are ordered/clamped, the language is lowercased, the limit capped."""
    answer = (
        '{"intent": "repo", "keywords": ["RAG"], "limit": 99, "filters": '
        '{"authors": ["", "  "], "year_from": 2024, "year_to": 2020, '
        '"language": "Python", "min_stars": -5}}'
    )
    parsed = parse_intent_auto("RAG repos", client=_FakeLLM(answer)).parsed

    assert parsed.limit == 20
    assert parsed.filters.year_from == 2020
    assert parsed.filters.year_to == 2024
    assert parsed.filters.language == "python"
    assert parsed.filters.min_stars == 0
    assert parsed.filters.authors is None


def test_impossible_year_is_dropped() -> None:
    """⑥ A year outside 1900-2100 is discarded instead of reaching an adapter."""
    filters = sanitize_filters(Filters(year_from=1800, year_to=2101))
    assert filters.year_from is None
    assert filters.year_to is None


def test_keyword_cleaning_dedupes_and_truncates() -> None:
    """⑦ Keywords are stripped, deduped case-insensitively, and capped."""
    assert clean_keywords(["  LLM ", "llm", "", "agents", "a", "b", "c", "d", "e", "f"]) == [
        "LLM",
        "agents",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]


def test_extract_json_object_rejects_plain_text() -> None:
    """⑧ Text without any JSON object raises ``ValueError``."""
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("I cannot help with that.")


# -------------------------------------------------------------------- fallback
def test_invalid_json_falls_back_to_rules() -> None:
    """⑨ A non-JSON answer degrades to the deterministic rule parser."""
    outcome = parse_intent_auto("帮我找 transformer 的论文", client=_FakeLLM("sorry, no idea"))

    assert outcome.source == "rules"
    assert outcome.fallback is True
    assert "unusable LLM answer" in outcome.note
    assert outcome.parsed.intent == parse_intent("帮我找 transformer 的论文").intent


def test_schema_violation_falls_back_to_rules() -> None:
    """⑩ An unknown intent value is a schema violation, so the rules answer."""
    answer = '{"intent": "blog_post", "keywords": ["x"], "filters": {}, "limit": 5}'
    outcome = parse_intent_auto("transformer survey", client=_FakeLLM(answer))

    assert outcome.source == "rules"
    assert outcome.fallback is True
    assert outcome.parsed.intent == "paper"


def test_empty_keywords_fall_back_to_rules() -> None:
    """⑪ A keyword-less plan is useless for search and is rejected."""
    answer = '{"intent": "paper", "keywords": [], "filters": {}, "limit": 5}'
    outcome = parse_intent_auto("transformer survey", client=_FakeLLM(answer))

    assert outcome.source == "rules"
    assert outcome.fallback is True
    assert outcome.parsed.keywords


def test_llm_error_falls_back_to_rules() -> None:
    """⑫ A transport/API failure never breaks the run."""
    outcome = parse_intent_auto("transformer survey", client=_FakeLLM(error=LLMError("timeout")))

    assert outcome.source == "rules"
    assert outcome.fallback is True
    assert "LLM call failed" in outcome.note
    assert outcome.parsed.intent == "paper"


def test_unexpected_exception_falls_back_to_rules() -> None:
    """⑬ Even an unexpected exception is absorbed by the rule parser."""
    outcome = parse_intent_auto("transformer survey", client=_FakeLLM(error=RuntimeError("boom")))

    assert outcome.source == "rules"
    assert outcome.fallback is True


def test_missing_key_uses_rules_without_a_call() -> None:
    """⑭ With no key configured the rules run and nothing is reported as failure."""
    fake = _FakeLLM(PAPER_ANSWER, configured=False)
    outcome = parse_intent_auto("transformer survey", client=fake)

    assert outcome.source == "rules"
    assert outcome.fallback is False
    assert "no LLM API key" in outcome.note
    assert fake.calls == []


def test_forced_rule_mode_skips_the_model() -> None:
    """⑮ ``--parser rule`` never calls the model, even when a key exists."""
    fake = _FakeLLM(PAPER_ANSWER)
    outcome = parse_intent_auto("transformer survey", mode="rule", client=fake)

    assert outcome.source == "rules"
    assert fake.calls == []
    assert outcome.parsed.intent == "paper"


def test_exact_url_skips_the_model_in_auto_mode() -> None:
    """⑯ An arXiv/GitHub URL is already exact, so auto mode costs no tokens."""
    fake = _FakeLLM(PAPER_ANSWER)
    outcome = parse_intent_auto("https://arxiv.org/abs/1706.03762", client=fake)

    assert outcome.source == "rules"
    assert fake.calls == []
    assert "1706.03762" in outcome.parsed.keywords


def test_forced_llm_mode_still_parses_urls_with_the_model() -> None:
    """⑰ ``--parser llm`` keeps the model in the loop even for a URL query."""
    fake = _FakeLLM(PAPER_ANSWER)
    outcome = parse_intent_auto(
        "https://arxiv.org/abs/1706.03762", mode="llm", client=fake
    )

    assert outcome.source == "llm"
    assert len(fake.calls) == 1


# ------------------------------------------------------------------- contract
def test_build_parsed_intent_rejects_garbage() -> None:
    """⑱ The builder is strict: callers get a ``ValueError``/``ValidationError``."""
    with pytest.raises(ValueError):
        build_parsed_intent("q", "no json here")
    with pytest.raises(ValidationError):
        build_parsed_intent("q", '{"intent": "not-an-intent", "keywords": ["x"]}')
