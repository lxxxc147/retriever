"""Tests for the argparse CLI main flow.

Adapters are replaced with in-memory stubs via monkeypatching — no real
network access in any test here.
"""

from pathlib import Path

import pytest

from retriever import cli, intent_llm
from retriever.adapters.base import SearchAdapter
from retriever.llm import LLMError
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.router import Router

_LLM_ANSWER = (
    '{"intent": "paper", "keywords": ["llm", "agent", "survey"], '
    '"filters": {"authors": null, "year_from": 2024, "year_to": 2024, '
    '"language": null, "min_stars": null, "categories": null}, "limit": 5}'
)


class _StubAdapter(SearchAdapter):
    """In-memory adapter recording the last query it was asked to search."""

    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.name = name
        self.rate_limit_qps = 100.0
        self._results = results
        self.last_query: SearchQuery | None = None

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        self.last_query = query
        return list(self._results)

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / f"{result.source}.bin"
        local_path.write_bytes(b"stub")
        return DownloadResult(success=True, local_path=local_path, size_bytes=4, sha256="aa")


def _stubs() -> dict[str, _StubAdapter]:
    return {
        "arxiv": _StubAdapter(
            "arxiv",
            [SearchResult(source="arxiv", title="Paper A", url="https://arxiv.org/abs/2401.00001")],
        ),
        "github": _StubAdapter(
            "github",
            [SearchResult(source="github", title="org/repo", url="https://github.com/org/repo")],
        ),
    }


@pytest.fixture
def stub_router(monkeypatch: pytest.MonkeyPatch) -> dict[str, _StubAdapter]:
    """Swap ``cli.Router`` for one backed by in-memory stub adapters."""
    stubs = _stubs()
    monkeypatch.setattr(cli, "Router", lambda: Router(adapters=list(stubs.values())))
    return stubs


class _FakeLLMClient:
    """Stand-in for :class:`retriever.llm.LLMClient` — no network, canned answer."""

    def __init__(
        self,
        configured: bool = True,
        answer: str = _LLM_ANSWER,
        error: Exception | None = None,
    ) -> None:
        self.configured = configured
        self.model = "fake-model"
        self._answer = answer
        self._error = error
        self.calls: list[dict] = []

    def chat_sync(self, messages: list[dict], **kwargs: object) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        return self._answer


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    """Install a fake LLM client wherever the CLI and the parser resolve one."""

    def _install(
        configured: bool = True,
        answer: str = _LLM_ANSWER,
        error: Exception | None = None,
    ) -> _FakeLLMClient:
        fake = _FakeLLMClient(configured=configured, answer=answer, error=error)
        monkeypatch.setattr(cli, "LLMClient", lambda: fake)
        monkeypatch.setattr(intent_llm, "LLMClient", lambda: fake)
        return fake

    return _install


def test_main_search_flow_with_source_filter(stub_router: dict[str, _StubAdapter], capsys: pytest.CaptureFixture) -> None:
    """``retrieve "query" --source arxiv --limit 2`` renders results, queries
    only the arXiv stub, and applies the limit."""
    exit_code = cli.main(["transformer survey", "--source", "arxiv", "--limit", "2"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[arXiv] Paper A" in out
    assert "https://arxiv.org/abs/2401.00001" in out
    assert stub_router["arxiv"].last_query is not None
    assert stub_router["arxiv"].last_query.limit == 2
    assert stub_router["github"].last_query is None


def test_main_default_source_queries_both(stub_router: dict[str, _StubAdapter], capsys: pytest.CaptureFixture) -> None:
    """Default --source all routes by intent; a dataset query fans out to both."""
    exit_code = cli.main(["imagenet 数据集"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[arXiv] Paper A" in out
    assert "[GitHub] org/repo" in out
    assert stub_router["arxiv"].last_query is not None
    assert stub_router["github"].last_query is not None
    assert stub_router["arxiv"].last_query.limit == 5  # default limit
    assert stub_router["arxiv"].last_query.intent == "dataset"


def test_main_intent_routing_with_default_source(
    stub_router: dict[str, _StubAdapter], capsys: pytest.CaptureFixture
) -> None:
    """A paper-intent query under --source all skips the GitHub adapter."""
    exit_code = cli.main(["transformer survey"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[arXiv] Paper A" in out
    assert "[GitHub]" not in out
    assert stub_router["arxiv"].last_query is not None
    assert stub_router["github"].last_query is None


def test_main_download_writes_manifest(
    stub_router: dict[str, _StubAdapter],
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """``--download`` downloads every result and writes manifest.json."""
    exit_code = cli.main(
        ["imagenet 数据集", "--download", "--out", str(tmp_path)]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Downloading [1/2]" in out
    assert "2 succeeded / 0 failed" in out

    manifest_paths = list(tmp_path.glob("*/manifest.json"))
    assert len(manifest_paths) == 1
    manifest_text = manifest_paths[0].read_text(encoding="utf-8")
    assert "Paper A" in manifest_text
    assert "org/repo" in manifest_text
    assert "task_id" in manifest_text


def test_main_no_download_flag_skips_downloads(
    stub_router: dict[str, _StubAdapter], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Without ``--download`` nothing is written to --out."""
    exit_code = cli.main(["transformer survey", "--out", str(tmp_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Downloading" not in out
    assert list(tmp_path.iterdir()) == []


def test_parser_rule_flag_reports_the_rule_parser(
    stub_router: dict[str, _StubAdapter], capsys: pytest.CaptureFixture
) -> None:
    """``--parser rule`` keeps the deterministic parser and says so on the plan line."""
    exit_code = cli.main(["transformer survey", "--parser", "rule"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "parser=rules" in out
    assert "intent=paper" in out
    assert stub_router["arxiv"].last_query is not None


def test_parser_llm_without_a_key_is_a_usage_error(
    stub_router: dict[str, _StubAdapter],
    fake_llm,
    capsys: pytest.CaptureFixture,
) -> None:
    """``--parser llm`` with no key fails loudly instead of silently using rules."""
    fake_llm(configured=False)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["transformer survey", "--parser", "llm"])

    assert excinfo.value.code == 2
    assert "LLM_API_KEY" in capsys.readouterr().err


def test_auto_parser_uses_the_llm_when_configured(
    stub_router: dict[str, _StubAdapter],
    fake_llm,
    capsys: pytest.CaptureFixture,
) -> None:
    """With a key present, auto mode takes the model's keywords and filter slots."""
    fake = fake_llm()

    exit_code = cli.main(["帮我找 llm agent 的论文"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "parser=llm(fake-model)" in out
    assert len(fake.calls) == 1
    query = stub_router["arxiv"].last_query
    assert query is not None
    # The rule parser would yield ["llm", "agent"]; these come from the model answer.
    assert query.keywords == ["llm", "agent", "survey"]
    assert query.filters.year_from == 2024


def test_llm_failure_falls_back_to_rules_with_a_warning(
    stub_router: dict[str, _StubAdapter],
    fake_llm,
    capsys: pytest.CaptureFixture,
) -> None:
    """A dead model endpoint degrades to the rules and warns — the run still completes."""
    fake_llm(error=LLMError("connection refused"))

    exit_code = cli.main(["transformer survey"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "warning:" in captured.err
    assert "connection refused" in captured.err
    assert "parser=rules" in captured.out
    assert stub_router["arxiv"].last_query is not None


def test_summary_flag_renders_extractive_summaries(
    stub_router: dict[str, _StubAdapter],
    capsys: pytest.CaptureFixture,
) -> None:
    """``--summary`` without a key still adds summaries and says so on stderr."""
    exit_code = cli.main(["transformer survey", "--summary", "--source", "arxiv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Summary:" in captured.out
    assert "extractive summaries" in captured.err


def test_summary_flag_uses_the_llm_when_configured(
    stub_router: dict[str, _StubAdapter],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """With a key configured the LLM summaries are rendered instead."""
    fake = _FakeLLMClient(
        answer='{"items": [{"i": 1, "s": "模型生成的一句话摘要"}, {"i": 2, "s": "第二条"}]}'
    )
    monkeypatch.setattr("retriever.summarizer.LLMClient", lambda: fake)

    exit_code = cli.main(["transformer survey", "--summary"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Summary: 模型生成的一句话摘要" in out
    assert len(fake.calls) == 1  # one batched call, not one per result
