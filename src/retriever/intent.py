"""M1 rule-based intent parser (task X-1, PRD §2.1).

Purely rule-based for Phase 1 — no LLM dependency (the LLM parser is a
Phase 2 upgrade path). The parser extracts, in order:

1. exact source URLs (arXiv ``/abs/`` & ``/pdf/`` links, ``github.com`` repos),
2. structured filter slots (author, year range, language, minimum stars),
3. a coarse intent classification,
4. leftover content words as keywords (stopwords removed).

Rules are deliberately simple and deterministic so they stay easy to test;
complex sentences are allowed to parse imperfectly (PRD §2.1).
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .models import Filters, Intent


class ParsedIntent(BaseModel):
    """Structured output of parsing one natural-language user query."""

    intent: Intent
    keywords: list[str]
    filters: Filters
    raw_query: str
    limit: int = 5


# --------------------------------------------------------------------------- RE
# Source URLs (exact recognition wins over everything else).
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\S+)", re.IGNORECASE)
_GITHUB_URL_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)")
_URL_RE = re.compile(r"https?://[^\s，。；）)】\]]+")

# Filter slots.
_AUTHOR_ANY_RE = re.compile(
    r"author\s*:\s*([A-Za-z][\w.-]*)"
    r"|\bby\s+([A-Za-z][\w.-]*)"
    r"|([A-Za-z][\w.-]*)\s+et\s+al\b\.?",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-~–—]\s*((?:19|20)\d{2})\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_LANGUAGE_NAMES = (
    "python", "java", "golang", "go", "rust", "ruby",
    "javascript", "typescript", "c++", "cpp", "c#",
    "swift", "kotlin", "matlab", "scala", "julia", "r",
)
_LANGUAGE_ALT = "|".join(re.escape(name) for name in _LANGUAGE_NAMES)
_LANGUAGE_SLOT_RE = re.compile(rf"language\s*:\s*({_LANGUAGE_ALT})", re.IGNORECASE)
_IN_LANGUAGE_RE = re.compile(rf"\bin\s+({_LANGUAGE_ALT})\b", re.IGNORECASE)
_ZH_LANGUAGE_RE = re.compile(rf"({_LANGUAGE_ALT})\s*实现", re.IGNORECASE)
_CODE_FILE_RE = re.compile(r"\.(?:py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|rb|cs)\b", re.IGNORECASE)
_STARS_SLOT_RE = re.compile(r"stars?\s*[:>]?=\s*>\s*(\d+)", re.IGNORECASE)
_STARS_GT_RE = re.compile(r"[>≥]\s*(\d+)\s*stars?\b", re.IGNORECASE)
_STARS_ZH_RE = re.compile(r"至少\s*(\d+)\s*(?:个)?\s*(?:stars?|星)", re.IGNORECASE)

_LANGUAGE_ALIASES = {"cpp": "c++", "golang": "go"}

# Intent markers (substring match on the lowercased remainder).
_PAPER_MARKERS = ("论文", "综述", "文献", "预印本", "paper", "survey", "preprint")
_REPO_MARKERS = ("repo", "repository", "github", "仓库", "开源项目")
_CODE_MARKERS = ("源码", "源代码", "代码文件", "source code", "code file")
_DATASET_MARKERS = ("数据集", "dataset", "语料", "corpus")

# Tokens that must never survive into the keyword list (intent markers and
# slot furniture), checked case-insensitively.
_KEYWORD_DROP = {
    "论文", "综述", "文献", "预印本", "实现", "仓库", "开源项目", "数据集", "语料",
    "源码", "源代码", "代码文件", "项目", "代码",
    "paper", "papers", "survey", "surveys", "preprint",
    "repo", "repos", "repository", "github",
    "dataset", "datasets", "corpus",
    "source", "code", "file", "web", "website", "网页",
}

_STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "at", "to", "and", "or", "with",
    "by", "about", "some", "any", "all", "me", "my", "i", "is", "are", "how",
    "what", "which", "who", "please", "find", "search", "download", "get",
    "show", "list", "give", "want", "need", "latest", "new", "best", "using",
    "use",
}

# Chinese noise fragments removed before tokenization (longer first matters:
# "帮我" is removed before the single char "找"/"看").
_ZH_NOISE = (
    "帮我", "麻烦", "帮忙", "给我", "我想", "我要", "有没有", "怎么样",
    "查找", "搜索", "下载", "相关", "一下", "看看", "看下", "至少",
    "的", "了", "和", "与", "及", "是", "在", "找", "看", "用",
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*|[一-鿿]+")


def _extract_url_slots(user_input: str) -> tuple[str, Intent | None, list[str]]:
    """Pull source URLs out of the raw input.

    Returns:
        ``(remainder, intent_or_none, url_keywords)`` — the remainder has the
        matched URLs blanked out, so URL fragments never leak into keywords.
    """
    remaining = user_input
    intent: Intent | None = None
    keywords: list[str] = []

    for match in _URL_RE.finditer(user_input):
        url = match.group(0)
        arxiv_match = _ARXIV_URL_RE.search(url)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1).split("?")[0].rstrip(")】]>")
            arxiv_id = re.sub(r"\.pdf$", "", arxiv_id, flags=re.IGNORECASE)
            intent = "paper"
            keywords.append(arxiv_id)
            remaining = remaining.replace(url, " ")
            continue
        github_match = _GITHUB_URL_RE.search(url)
        if github_match:
            full_name = f"{github_match.group(1)}/{github_match.group(2)}".removesuffix(".git")
            intent = "repo"
            keywords.append(full_name)
            remaining = remaining.replace(url, " ")

    return remaining, intent, keywords


def _extract_authors(remaining: str, authors: list[str]) -> str:
    """Collect author names from ``author:`` / ``by X`` / ``X et al`` slots.

    A single combined pass over the original text (instead of sequential
    sub-extractions) keeps the patterns from interacting: removing
    ``by Vaswani`` first would otherwise promote the preceding word into an
    accidental ``X et al`` match.
    """
    for match in _AUTHOR_ANY_RE.finditer(remaining):
        name = next(group for group in match.groups() if group is not None)
        if name not in authors:
            authors.append(name)
    return _AUTHOR_ANY_RE.sub(" ", remaining)


def _extract_year(remaining: str) -> tuple[str, int | None, int | None]:
    """Collect a single year or a ``2020-2024`` style year range."""
    match = _YEAR_RANGE_RE.search(remaining)
    if match:
        year_from, year_to = int(match.group(1)), int(match.group(2))
        return _YEAR_RANGE_RE.sub(" ", remaining), year_from, year_to
    match = _YEAR_RE.search(remaining)
    if match:
        year = int(match.group(1))
        return _YEAR_RE.sub(" ", remaining), year, year
    return remaining, None, None


def _extract_language(remaining: str) -> tuple[str, str | None]:
    """Collect a language slot (``language:python`` / ``in python`` / ``python 实现``)."""
    for pattern in (_LANGUAGE_SLOT_RE, _IN_LANGUAGE_RE, _ZH_LANGUAGE_RE):
        match = pattern.search(remaining)
        if match:
            name = match.group(1).lower()
            return pattern.sub(" ", remaining), _LANGUAGE_ALIASES.get(name, name)
    return remaining, None


def _extract_min_stars(remaining: str) -> tuple[str, int | None]:
    """Collect a minimum-stars slot (``stars:>100`` / ``>100 stars`` / ``至少 100 stars``)."""
    for pattern in (_STARS_SLOT_RE, _STARS_GT_RE, _STARS_ZH_RE):
        match = pattern.search(remaining)
        if match:
            return pattern.sub(" ", remaining), int(match.group(1))
    return remaining, None


def _classify(remaining: str, url_intent: Intent | None) -> Intent:
    """Coarse intent classification over the leftover text (URL intent wins)."""
    if url_intent is not None:
        return url_intent
    lowered = remaining.lower()
    if any(marker in lowered for marker in _DATASET_MARKERS):
        return "dataset"
    if any(marker in lowered for marker in _CODE_MARKERS) or _CODE_FILE_RE.search(remaining):
        return "code_file"
    if any(marker in lowered for marker in _REPO_MARKERS):
        return "repo"
    if any(marker in lowered for marker in _PAPER_MARKERS):
        return "paper"
    return "web"


def _extract_keywords(remaining: str, url_keywords: list[str]) -> list[str]:
    """Tokenize the leftover text into content keywords.

    Chinese noise fragments and English stopwords are removed first; intent
    markers and slot furniture are dropped token-wise. Order-preserving dedupe.
    """
    for noise in _ZH_NOISE:
        remaining = remaining.replace(noise, " ")
    keywords: list[str] = list(url_keywords)
    for token in _TOKEN_RE.findall(remaining):
        if token.lower() in _STOPWORDS or token.lower() in _KEYWORD_DROP:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords


def parse_intent(user_input: str) -> ParsedIntent:
    """Parse a natural-language query into a structured :class:`ParsedIntent`.

    Pure rules — deterministic, no LLM and no NLP library (PRD §2.1, Phase 1).

    Args:
        user_input: The user's original query (Chinese or English).

    Returns:
        The detected intent, extracted filter slots, and content keywords.
    """
    remaining, url_intent, url_keywords = _extract_url_slots(user_input)

    authors: list[str] = []
    remaining = _extract_authors(remaining, authors)
    remaining, year_from, year_to = _extract_year(remaining)
    remaining, language = _extract_language(remaining)
    remaining, min_stars = _extract_min_stars(remaining)

    intent = _classify(remaining, url_intent)
    keywords = _extract_keywords(remaining, url_keywords)

    filters = Filters(
        authors=authors or None,
        year_from=year_from,
        year_to=year_to,
        language=language,
        min_stars=min_stars,
    )
    return ParsedIntent(
        intent=intent,
        keywords=keywords,
        filters=filters,
        raw_query=user_input,
        limit=5,
    )
