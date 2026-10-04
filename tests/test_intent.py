"""Tests for the rule-based intent parser (task X-1). No network access."""

from retriever.intent import parse_intent


def test_arxiv_abs_url_recognized() -> None:
    """An arXiv /abs/ URL pins intent=paper and yields the bare arXiv ID."""
    parsed = parse_intent("please read https://arxiv.org/abs/1706.03762")
    assert parsed.intent == "paper"
    assert "1706.03762" in parsed.keywords


def test_arxiv_pdf_url_recognized() -> None:
    """An arXiv /pdf/ URL (with .pdf suffix) also yields the bare ID."""
    parsed = parse_intent("下载 https://arxiv.org/pdf/1706.03762.pdf 这篇")
    assert parsed.intent == "paper"
    assert "1706.03762" in parsed.keywords
    assert "https" not in " ".join(parsed.keywords)


def test_github_url_recognized() -> None:
    """A github.com/owner/repo URL pins intent=repo and yields owner/repo."""
    parsed = parse_intent("看一下 https://github.com/pytorch/pytorch 这个仓库")
    assert parsed.intent == "repo"
    assert "pytorch/pytorch" in parsed.keywords


def test_author_and_year_extracted() -> None:
    """``by X et al`` and a bare four-digit year fill the filter slots."""
    parsed = parse_intent("papers by Vaswani et al 2017")
    assert parsed.intent == "paper"
    assert parsed.filters.authors == ["Vaswani"]
    assert parsed.filters.year_from == 2017
    assert parsed.filters.year_to == 2017


def test_author_colon_slot() -> None:
    """``author:NAME`` fills the authors slot without an LLM."""
    parsed = parse_intent("transformer author:lecun")
    assert parsed.filters.authors == ["lecun"]
    assert "lecun" not in parsed.keywords


def test_year_range_extracted() -> None:
    """A ``2020-2024`` range fills year_from/year_to."""
    parsed = parse_intent("transformer survey 2020-2024")
    assert parsed.intent == "paper"
    assert parsed.filters.year_from == 2020
    assert parsed.filters.year_to == 2024


def test_language_slot_variants() -> None:
    """``language:X``, ``in X`` and ``X 实现`` all fill Filters.language."""
    assert parse_intent("language:python transformer").filters.language == "python"
    assert parse_intent("web crawler in go").filters.language == "go"
    assert parse_intent("transformer 的 python 实现").filters.language == "python"


def test_min_stars_variants() -> None:
    """``stars:>N``, ``>N stars`` and ``至少 N stars`` fill Filters.min_stars."""
    assert parse_intent("llm agents stars:>100").filters.min_stars == 100
    assert parse_intent("llm agents >100 stars").filters.min_stars == 100
    assert parse_intent("至少 500 stars 的 rag 项目").filters.min_stars == 500


def test_chinese_paper_sentence() -> None:
    """中文句式：“帮我找 transformer 的论文” → paper, keyword 只剩 transformer。"""
    parsed = parse_intent("帮我找 transformer 的论文")
    assert parsed.intent == "paper"
    assert parsed.keywords == ["transformer"]


def test_code_file_intent_from_source_code() -> None:
    """“xxx 的源码” routes to code_file."""
    parsed = parse_intent("transformer 的源码")
    assert parsed.intent == "code_file"


def test_code_file_intent_from_extension() -> None:
    """A literal “xxx.py” file mention routes to code_file."""
    parsed = parse_intent("find the transformer.py code file")
    assert parsed.intent == "code_file"


def test_dataset_intent() -> None:
    """“数据集 / dataset” routes to dataset and is dropped from keywords."""
    parsed = parse_intent("imagenet 数据集")
    assert parsed.intent == "dataset"
    assert parsed.keywords == ["imagenet"]


def test_repo_intent() -> None:
    """Bare “github / repo / 仓库” mentions route to repo."""
    assert parse_intent("pytorch 仓库").intent == "repo"
    assert parse_intent("a git repo for rag").intent == "repo"


def test_web_fallback_for_plain_keywords() -> None:
    """Sentences with no marker fall back to web, keywords preserved."""
    parsed = parse_intent("how old is the moon")
    assert parsed.intent == "web"
    assert "moon" in parsed.keywords


def test_stopword_cleaning() -> None:
    """中英文 stopword 与意图标记词不进入 keywords。"""
    parsed = parse_intent("please find the survey about attention mechanisms")
    assert parsed.intent == "paper"
    assert "please" not in parsed.keywords
    assert "find" not in parsed.keywords
    assert "the" not in parsed.keywords
    assert "survey" not in parsed.keywords
    assert "attention" in parsed.keywords
    assert "mechanisms" in parsed.keywords


def test_raw_query_and_limit_defaults() -> None:
    """raw_query is echoed verbatim and the default limit stays 5."""
    parsed = parse_intent("transformer survey")
    assert parsed.raw_query == "transformer survey"
    assert parsed.limit == 5


def test_empty_input_is_safe() -> None:
    """Degenerate input parses without crashing and falls back to web."""
    parsed = parse_intent("")
    assert parsed.intent == "web"
    assert parsed.keywords == []
