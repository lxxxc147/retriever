"""Tests for Router (task X-3). All adapters are in-memory stubs — no network."""

from pathlib import Path

import pytest

from retriever.adapters.base import SearchAdapter
from retriever.intent import ParsedIntent
from retriever.models import DownloadResult, Filters, SearchQuery, SearchResult
from retriever.router import Router


class _StubAdapter(SearchAdapter):
    """In-memory adapter: returns canned results, or raises when told to."""

    def __init__(
        self,
        name: str,
        results: list[SearchResult] | None = None,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.rate_limit_qps = 100.0
        self._results = results or []
        self._fail = fail

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if self._fail:
            raise RuntimeError(f"{self.name} exploded")
        return list(self._results)

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        return DownloadResult(success=True, local_path=dest_dir / "stub", size_bytes=1, sha256="x")


def _result(source: str, title: str) -> SearchResult:
    return SearchResult(source=source, title=title, url=f"https://{source}.example/{title}")


def _query(intent: str = "web") -> SearchQuery:
    return SearchQuery(
        intent=intent,  # type: ignore[arg-type]
        raw_query="q",
        keywords=["kw"],
        filters=Filters(),
        limit=5,
    )


def _parsed(intent: str) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,  # type: ignore[arg-type]
        keywords=[],
        filters=Filters(),
        raw_query="q",
    )


def test_route_paper_selects_arxiv_only() -> None:
    router = Router()
    assert [a.name for a in router.route(_parsed("paper"))] == ["arxiv"]


def test_route_repo_selects_github_only() -> None:
    router = Router()
    assert [a.name for a in router.route(_parsed("repo"))] == ["github"]


def test_route_code_file_selects_github_only() -> None:
    router = Router()
    assert [a.name for a in router.route(_parsed("code_file"))] == ["github"]


def test_route_dataset_and_web_select_both() -> None:
    router = Router()
    assert [a.name for a in router.route(_parsed("dataset"))] == ["arxiv", "github"]
    assert [a.name for a in router.route(_parsed("web"))] == ["arxiv", "github"]


def test_search_all_aggregates_with_canonical_group_order() -> None:
    """Parallel aggregation works and results are grouped arXiv-first even when
    the adapters were registered in reverse order."""
    arxiv = _StubAdapter("arxiv", [_result("arxiv", "Paper A")])
    github = _StubAdapter("github", [_result("github", "Repo B"), _result("github", "Repo C")])
    router = Router(adapters=[github, arxiv])

    results = router.search_all(_query("dataset"))

    assert [(r.source, r.title) for r in results] == [
        ("arxiv", "Paper A"),
        ("github", "Repo B"),
        ("github", "Repo C"),
    ]


def test_search_all_survives_failing_adapter(capsys: pytest.CaptureFixture) -> None:
    """One adapter raising must not interrupt the other sources."""
    failing = _StubAdapter("arxiv", fail=True)
    ok = _StubAdapter("github", [_result("github", "Repo B")])
    router = Router(adapters=[failing, ok])

    results = router.search_all(_query("web"))

    assert [r.title for r in results] == ["Repo B"]
    assert "arxiv" in capsys.readouterr().err


def test_search_all_direct_arxiv_id_lookup() -> None:
    """A bare arXiv ID keyword bypasses keyword search via get_by_id."""

    class _ArxivStub(_StubAdapter):
        async def get_by_id(self, arxiv_id: str) -> SearchResult | None:
            self.requested_id = arxiv_id
            return _result("arxiv", "Direct paper")

    stub = _ArxivStub("arxiv", [_result("arxiv", "Keyword paper")])
    router = Router(adapters=[stub])
    query = SearchQuery(
        intent="paper",
        raw_query="https://arxiv.org/abs/1706.03762",
        keywords=["1706.03762"],
        filters=Filters(),
        limit=5,
    )

    results = router.search_all(query)

    assert [r.title for r in results] == ["Direct paper"]
    assert stub.requested_id == "1706.03762"


def test_search_all_no_adapters_returns_empty() -> None:
    router = Router(adapters=[])
    assert router.search_all(_query("paper")) == []
