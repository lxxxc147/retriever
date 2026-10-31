"""Tests for the interactive CLI mode — refine loop, detail view, downloads (F8.2).

Adapters are in-memory stubs (same pattern as ``test_cli.py``); the prompt reader
is a scripted fake, so no test touches the network or real stdin.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from retriever import cli
from retriever.adapters.base import SearchAdapter
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.router import Router


class _StubAdapter(SearchAdapter):
    """In-memory adapter recording every query it was asked to search."""

    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.name = name
        self.rate_limit_qps = 100.0
        self._results = results
        self.queries: list[SearchQuery] = []

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        self.queries.append(query)
        return list(self._results)

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / f"{result.source}.bin"
        local_path.write_bytes(b"stub")
        return DownloadResult(success=True, local_path=local_path, size_bytes=4, sha256="aa")


class _ScriptedInput:
    """Fake prompt reader feeding a fixed script, then raising EOF."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __call__(self, prompt: str = "") -> str:
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None


def _two_results() -> list[SearchResult]:
    return [
        SearchResult(
            source="arxiv",
            title="Paper A",
            url="https://arxiv.org/abs/2401.00001",
            published_at=datetime(2023, 5, 1, tzinfo=UTC),
        ),
        SearchResult(
            source="arxiv",
            title="Paper B",
            url="https://arxiv.org/abs/2401.00002",
            published_at=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    ]


@pytest.fixture
def stub_adapter(monkeypatch: pytest.MonkeyPatch) -> _StubAdapter:
    adapter = _StubAdapter("arxiv", _two_results())
    monkeypatch.setattr(cli, "Router", lambda: Router(adapters=[adapter]))
    return adapter


@pytest.fixture(autouse=True)
def isolated_history(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("history")
    monkeypatch.setattr("retriever.history.DEFAULT_HISTORY_DIR", directory)
    return directory


def run_interactive_script(
    argv: list[str], lines: list[str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> tuple[int, str]:
    parser = cli.build_parser()
    args = parser.parse_args(argv + ["--out", str(tmp_path / "downloads")])
    code = cli.run_interactive(args, parser, input_func=_ScriptedInput(lines))
    return code, capsys.readouterr().out


def test_welcome_help_and_quit(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``help`` lists the commands; ``q`` ends the session with exit code 0."""
    code, out = run_interactive_script([], ["help", "q"], tmp_path, capsys)
    assert code == 0
    assert "交互模式" in out
    assert "d N" in out


def test_eof_ends_the_session_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Running out of scripted input (EOF) exits quietly, like ``q``."""
    code, _ = run_interactive_script([], [], tmp_path, capsys)
    assert code == 0


def test_first_line_is_searched_then_quit(
    stub_adapter: _StubAdapter, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The opening line drives the first search and renders its results."""
    code, out = run_interactive_script([], ["transformer survey", "q"], tmp_path, capsys)
    assert code == 0
    assert "Plan:" in out
    assert "[arXiv] Paper A" in out
    assert "continue" in out or "refine" in out  # follow-up hint printed
    assert len(stub_adapter.queries) == 1


def test_refine_filters_results_locally(
    stub_adapter: _StubAdapter, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A narrowing follow-up filters the current list without a new search."""
    code, out = run_interactive_script(
        [], ["transformer survey", "只要 2024 的", "q"], tmp_path, capsys
    )
    assert code == 0
    assert "Found 1 result(s)" in out  # narrowed from 2 to 1
    assert "[arXiv] Paper B" in out  # the 2024 paper survives
    assert len(stub_adapter.queries) == 1  # no re-search happened


def test_refine_with_no_local_match_researches_with_merged_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """When nothing survives local filtering the merged query goes to the sources."""
    stale = [
        SearchResult(
            source="arxiv",
            title="Stale 2021 paper",
            url="https://arxiv.org/abs/2401.00008",
            published_at=datetime(2021, 9, 1, tzinfo=UTC),
        )
    ]
    fresh = [
        SearchResult(
            source="arxiv",
            title="Fresh 2024 paper",
            url="https://arxiv.org/abs/2401.00009",
            published_at=datetime(2024, 9, 1, tzinfo=UTC),
        )
    ]

    class _SwitchingStub(_StubAdapter):
        async def search(self, query: SearchQuery) -> list[SearchResult]:
            self.queries.append(query)
            return fresh if len(self.queries) > 1 else list(self._results)

    adapter = _SwitchingStub("arxiv", stale)
    monkeypatch.setattr(cli, "Router", lambda: Router(adapters=[adapter]))

    code, out = run_interactive_script(
        [], ["transformer survey", "只要 2024 的", "q"], tmp_path, capsys
    )
    assert code == 0
    assert len(adapter.queries) == 2  # initial search + merged re-query
    assert adapter.queries[1].filters.year_from == 2024
    assert "[arXiv] Fresh 2024 paper" in out


def test_detail_command_renders_one_result(
    stub_adapter: _StubAdapter, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Typing a bare number expands that result; out-of-range numbers complain.

    Ranking puts the fresher 2024 paper first, so #2 is Paper A.
    """
    code, out = run_interactive_script(
        [], ["transformer survey", "2", "9", "q"], tmp_path, capsys
    )
    assert code == 0
    assert "[2] [arXiv] Paper A" in out
    assert "no result #9" in out


def test_download_all_and_download_one(
    stub_adapter: _StubAdapter, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """``d`` downloads everything; ``d N`` downloads one; manifests land in --out."""
    code, out = run_interactive_script(
        [], ["transformer survey", "d 2", "d", "q"], tmp_path, capsys
    )
    assert code == 0
    assert "Downloading 1 file(s)" in out
    assert "Downloading 2 file(s)" in out
    manifests = list((tmp_path / "downloads").glob("*/manifest.json"))
    assert len(manifests) == 2


def test_download_without_results_warns(
    stub_adapter: _StubAdapter, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A download command before any search prints a hint instead of crashing."""
    code, out = run_interactive_script([], ["d", "q"], tmp_path, capsys)
    assert code == 0
    assert "nothing to download" in out


def test_history_records_every_turn(
    stub_adapter: _StubAdapter, isolated_history: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Each search and each refine lands one entry in the history log."""
    run_interactive_script(
        [], ["transformer survey", "只要 2024 的", "q"], tmp_path, capsys
    )
    files = list(isolated_history.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "transformer survey" in lines[0]
    assert "只要 2024 的" in lines[1]
