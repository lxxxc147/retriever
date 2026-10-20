"""M1 LLM intent parser — the Phase 2 upgrade path of the rule parser.

PRD §2.1 lists LLM-based slot extraction and multi-turn clarification as the
Phase 2 upgrade; the rule parser (:mod:`retriever.intent`) stays the
deterministic default so the project keeps working with no key, no network and
no cost.

Design contract:

- output is the SAME :class:`~retriever.intent.ParsedIntent` the rule parser
  returns, so the router, CLI and tests cannot tell the two apart;
- every failure mode (missing key, timeout, HTTP error, malformed JSON, schema
  violation, empty keywords) degrades to the rule parser *with a recorded
  reason* instead of raising — a flaky model must never break a search run
  (PRD §3 NF2);
- the model is only asked to classify and fill slots, never to invent
  identifiers: the prompt forbids inventing years/authors, and the answer is
  re-validated with pydantic before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from .intent import ParsedIntent, has_source_url, parse_intent
from .llm import LLMClient, LLMError, extract_json_object
from .models import Filters, Intent

# Upper bound for the result count a model may request (PRD §2.1 default: 5).
MAX_LIMIT = 20
MAX_KEYWORDS = 8

_SYSTEM_PROMPT = """You are the query-understanding stage (M1) of Retriever, an agent that searches
arXiv papers and GitHub repositories and then downloads what it finds.

Turn the user's request into ONE JSON object, with no other text:
{"intent": "paper" | "repo" | "code_file" | "dataset" | "web",
 "keywords": ["..."],
 "filters": {"authors": ["..."], "year_from": null, "year_to": null,
             "language": null, "min_stars": null, "categories": null},
 "limit": 5}

Rules:
- intent: paper = 论文/综述/survey; repo = GitHub 仓库/项目; code_file = 仓库里的代码文件/源码;
  dataset = 数据集/语料; web = anything else (网页/教程/unknown).
- keywords: 1-5 short bare search terms. Drop intent words such as 帮我找/论文/下载/仓库/repo/paper.
  Keep the user's own language. Keep an arXiv ID (1706.03762) or an owner/repo pair
  (pytorch/pytorch) exactly as written and put it in keywords.
- filters: fill ONLY what the user actually said, otherwise null. authors are Latin-script
  personal names. year_from/year_to are integers; "2024 年" means both are 2024, "2023 年以后"
  means year_from=2023. language is lowercase (python, go, rust, c++). min_stars is an integer.
  categories are arXiv category codes (cs.CL, cs.LG) only when the user names the field.
- limit: integer 1-20, default 5.
- Never invent authors, years, stars or identifiers.

Examples:
user: 帮我找 transformer 的论文
assistant: {"intent": "paper", "keywords": ["transformer"], "filters": {"authors": null, "year_from": null, "year_to": null, "language": null, "min_stars": null, "categories": null}, "limit": 5}
user: 2023 年以后 RAG 相关的 github 项目，stars 超过 500，用 python 写的
assistant: {"intent": "repo", "keywords": ["RAG"], "filters": {"authors": null, "year_from": 2023, "year_to": null, "language": "python", "min_stars": 500, "categories": null}, "limit": 5}
user: Vaswani 2017 那篇 attention 的论文
assistant: {"intent": "paper", "keywords": ["attention"], "filters": {"authors": ["Vaswani"], "year_from": 2017, "year_to": 2017, "language": null, "min_stars": null, "categories": null}, "limit": 5}
"""

# LLM answers sometimes use a friendlier label than the contract's enum value.
_INTENT_ALIASES: dict[str, str] = {
    "论文": "paper",
    "文章": "paper",
    "paper_search": "paper",
    "repository": "repo",
    "repositories": "repo",
    "github": "repo",
    "git_repo": "repo",
    "仓库": "repo",
    "代码": "code_file",
    "code": "code_file",
    "source_code": "code_file",
    "数据集": "dataset",
    "网页": "web",
    "search": "web",
}

MIN_YEAR = 1900
MAX_YEAR = 2100


class _LLMIntentPayload(BaseModel):
    """Validated shape of the JSON object the model is instructed to return."""

    intent: Intent
    keywords: list[str] = Field(default_factory=list)
    filters: Filters = Field(default_factory=Filters)
    limit: int = 5


@dataclass(frozen=True)
class ParseOutcome:
    """Result of parsing one query, plus which parser actually produced it.

    Attributes:
        parsed: The structured intent, in the shared :class:`ParsedIntent` contract.
        source: ``"llm"`` or ``"rules"``.
        note: Model name on success, or the reason the rules were used instead.
            Empty when the rules were simply requested or no key is configured.
        fallback: ``True`` only when the LLM path was attempted and failed, so
            the CLI can warn the user without nagging on every rule-based run.
    """

    parsed: ParsedIntent
    source: Literal["llm", "rules"]
    note: str = ""
    fallback: bool = False


def parse_intent_auto(
    user_input: str,
    *,
    mode: Literal["auto", "rule", "llm"] = "auto",
    client: LLMClient | None = None,
) -> ParseOutcome:
    """Parse a query with the LLM when it makes sense, else with the rules.

    Args:
        user_input: The user's original query (Chinese or English).
        mode: ``"auto"`` uses the LLM when a key is configured (and the query is
            not already an exact source URL); ``"rule"`` forces the deterministic
            rule parser; ``"llm"`` forces the LLM and falls back to the rules
            only when the call or the answer is unusable.
        client: Pre-built LLM client (tests inject one; ``None`` resolves the
            default client from ``.env`` / ``config/settings.yaml``).

    Returns:
        A :class:`ParseOutcome` whose ``parsed`` field is always populated —
        this function never raises for LLM failures.
    """
    if mode == "rule":
        return ParseOutcome(parse_intent(user_input), "rules", "rule parser requested")

    # An exact arXiv/GitHub URL is already an unambiguous plan: no need to pay
    # for a model call, and the rules keep the extracted ID exact.
    if mode == "auto" and has_source_url(user_input):
        return ParseOutcome(parse_intent(user_input), "rules", "exact source URL")

    return _parse_with_llm(user_input, client or LLMClient())


def _parse_with_llm(user_input: str, client: LLMClient) -> ParseOutcome:
    """Run one LLM round trip and convert failures into a rule-parser fallback."""
    if not client.configured:
        return ParseOutcome(parse_intent(user_input), "rules", "no LLM API key configured")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    try:
        answer = client.chat_sync(messages, json_mode=True)
    except LLMError as exc:
        logger.warning("LLM intent parsing failed, using the rule parser: {}", exc)
        return ParseOutcome(
            parse_intent(user_input), "rules", f"LLM call failed: {exc}", fallback=True
        )
    except Exception as exc:  # noqa: BLE001 — the rules are the last safety net
        logger.warning("Unexpected LLM failure, using the rule parser: {}", exc)
        return ParseOutcome(
            parse_intent(user_input), "rules", f"LLM call failed: {exc}", fallback=True
        )

    try:
        parsed = build_parsed_intent(user_input, answer)
    except (ValueError, ValidationError) as exc:
        logger.warning("Unusable LLM answer, using the rule parser: {}", exc)
        return ParseOutcome(
            parse_intent(user_input), "rules", f"unusable LLM answer: {exc}", fallback=True
        )

    return ParseOutcome(parsed, "llm", client.model)


def build_parsed_intent(user_input: str, answer: str) -> ParsedIntent:
    """Convert a raw LLM answer into a validated :class:`ParsedIntent`.

    Args:
        user_input: The original query (echoed into ``raw_query``).
        answer: The raw model output, possibly wrapped in a `````json`` fence.

    Returns:
        The validated, sanitized intent.

    Raises:
        ValueError: When no JSON object is found or no keyword survives cleaning.
        ValidationError: When the object violates the payload schema.
    """
    data = extract_json_object(answer)
    intent = data.get("intent")
    if isinstance(intent, str) and intent.strip().lower() in _INTENT_ALIASES:
        data["intent"] = _INTENT_ALIASES[intent.strip().lower()]

    payload = _LLMIntentPayload.model_validate(data)
    keywords = clean_keywords(payload.keywords)
    if not keywords:
        raise ValueError("the LLM returned no usable keywords")

    return ParsedIntent(
        intent=payload.intent,
        keywords=keywords,
        filters=sanitize_filters(payload.filters),
        raw_query=user_input,
        limit=min(MAX_LIMIT, max(1, payload.limit)),
    )


def clean_keywords(keywords: list[str]) -> list[str]:
    """Strip, drop empties, and order-preserving dedupe (case-insensitive).

    Args:
        keywords: Whatever the model put in ``keywords``.

    Returns:
        At most :data:`MAX_KEYWORDS` usable search terms.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for keyword in keywords or []:
        term = str(keyword).strip()
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        cleaned.append(term)
        if len(cleaned) >= MAX_KEYWORDS:
            break
    return cleaned


def sanitize_filters(filters: Filters) -> Filters:
    """Normalize slot values into the ranges the adapters can actually use.

    Drops impossible years, swaps an inverted range, lowercases the language and
    removes empty author/category entries.

    Args:
        filters: Filters built from the model answer.

    Returns:
        A new :class:`Filters` with normalized values.
    """
    year_from = _clamp_year(filters.year_from)
    year_to = _clamp_year(filters.year_to)
    if year_from is not None and year_to is not None and year_from > year_to:
        year_from, year_to = year_to, year_from

    authors = [name.strip() for name in filters.authors or [] if name and name.strip()]
    categories = [code.strip() for code in filters.categories or [] if code and code.strip()]
    language = (filters.language or "").strip().lower() or None
    min_stars = filters.min_stars if filters.min_stars is None else max(0, int(filters.min_stars))

    return Filters(
        authors=authors or None,
        year_from=year_from,
        year_to=year_to,
        language=language,
        min_stars=min_stars,
        categories=categories or None,
    )


def _clamp_year(value: int | None) -> int | None:
    """Return ``value`` when it looks like a plausible year, else ``None``."""
    if value is None:
        return None
    year = int(value)
    return year if MIN_YEAR <= year <= MAX_YEAR else None
