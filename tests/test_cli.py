"""Tests for the argparse CLI main flow.

Adapters are replaced with in-memory stubs via monkeypatching — no real
network access in any test here.
"""

from pathlib import Path

import pytest

from retriever import cli
from retriever.adapters.base import SearchAdapter
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.router import Router


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
    monkeypatch.setattr(cli, "Router", lambda: Router(adapters=list(stubs.values()))
    return stubs


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
