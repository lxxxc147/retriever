"""Integration tests for DownloadManager (task B-5).

All adapters here are in-memory fakes — no real network is touched.
"""

import asyncio as real_asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from retriever.adapters.base import SearchAdapter
from retriever.downloader import DownloadManager
from retriever.http import HttpClient
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.utils import sanitize_filename


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the downloader's view of asyncio.sleep to avoid real waits."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "retriever.downloader.asyncio",
        SimpleNamespace(
            sleep=_no_sleep,
            gather=real_asyncio.gather,
            Semaphore=real_asyncio.Semaphore,
        ),
    )


class FakeAdapter(SearchAdapter):
    """In-memory adapter returning canned bytes per result title."""

    name = "fake"
    rate_limit_qps = 1000.0

    def __init__(self, payloads: dict[str, bytes], fail_first: set[str] | None = None) -> None:
        self._payloads = dict(payloads)
        self._fail_first = set(fail_first or ())
        self._failed_once: set[str] = set()

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return []

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        payload = self._payloads.get(result.title)
        if payload is None:
            return DownloadResult(success=False, error="title not in fake payloads")
        if result.title in self._fail_first and result.title not in self._failed_once:
            self._failed_once.add(result.title)
            return DownloadResult(success=False, error="transient failure")
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{sanitize_filename(result.title)}.bin"
        path.write_bytes(payload)
        return DownloadResult(
            success=True,
            local_path=path,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )


class TrackingAdapter(FakeAdapter):
    """FakeAdapter that records observed download parallelism."""

    name = "tracked"

    def __init__(self) -> None:
        super().__init__({})
        self.active = 0
        self.peak = 0

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await real_asyncio.sleep(0.01)
        self.active -= 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{sanitize_filename(result.title)}.bin"
        path.write_bytes(result.title.encode("utf-8"))
        return DownloadResult(
            success=True,
            local_path=path,
            size_bytes=len(result.title),
            sha256=hashlib.sha256(result.title.encode("utf-8")).hexdigest(),
        )


def _results(source: str, titles: list[str]) -> list[SearchResult]:
    return [
        SearchResult(source=source, title=title, url=f"https://example.org/{title}")
        for title in titles
    ]


def _read_manifest(task_dir: Path) -> dict:
    return json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))


async def test_download_all_writes_numbered_files_and_manifest(tmp_path: Path) -> None:
    """① Files are saved as {seq}_{title}.bin and the manifest is complete (F5.1/F5.2/F5.8)."""
    payloads = {"Alpha": b"aaa", "Beta": b"bbb", "Gamma": b"ccc"}
    manager = DownloadManager(adapters={"fake": FakeAdapter(payloads)}, max_concurrent=2)
    task_dir = tmp_path / "task"

    outcomes = await manager.download_all(
        _results("fake", list(payloads)), task_dir, task_id="T-1", query="demo"
    )

    assert all(outcome.success for outcome in outcomes)
    names = sorted(path.name for path in (task_dir / "fake").iterdir())
    assert names == ["01_Alpha.bin", "02_Beta.bin", "03_Gamma.bin"]
    assert not (task_dir / "download_failures.log").exists()

    manifest = _read_manifest(task_dir)
    assert manifest["task_id"] == "T-1"
    assert manifest["query"] == "demo"
    assert len(manifest["files"]) == 3
    assert all(Path(entry["local_path"]).exists() for entry in manifest["files"])
    assert all(entry["sha256"] for entry in manifest["files"])


async def test_duplicate_content_is_stored_once(tmp_path: Path) -> None:
    """② Identical bytes downloaded twice are stored once and share a path (F5.3)."""
    payloads = {"First": b"same-bytes", "Second": b"same-bytes"}
    manager = DownloadManager(adapters={"fake": FakeAdapter(payloads)}, max_concurrent=1)
    task_dir = tmp_path / "task"

    outcomes = await manager.download_all(_results("fake", list(payloads)), task_dir)

    assert all(outcome.success for outcome in outcomes)
    assert outcomes[0].local_path == outcomes[1].local_path
    assert len(list((task_dir / "fake").iterdir())) == 1


async def test_failure_is_logged_and_excluded_from_manifest(tmp_path: Path) -> None:
    """③ A failed download lands in download_failures.log and not in the manifest (F5.6)."""
    manager = DownloadManager(adapters={"fake": FakeAdapter({"Good": b"ok"})}, max_retries=1)
    task_dir = tmp_path / "task"

    outcomes = await manager.download_all(
        _results("fake", ["Good", "Missing"]), task_dir, query="q"
    )

    assert outcomes[0].success
    assert not outcomes[1].success
    log = (task_dir / "download_failures.log").read_text(encoding="utf-8")
    assert "https://example.org/Missing" in log
    assert len(_read_manifest(task_dir)["files"]) == 1


async def test_retry_recovers_transient_failure(tmp_path: Path) -> None:
    """④ A transient failure is retried until it succeeds (F5.6)."""
    adapter = FakeAdapter({"Flaky": b"data"}, fail_first={"Flaky"})
    manager = DownloadManager(adapters={"fake": adapter}, max_retries=3, max_concurrent=1)

    outcomes = await manager.download_all(_results("fake", ["Flaky"]), tmp_path / "task")

    assert outcomes[0].success


async def test_manifest_written_even_when_all_fail(tmp_path: Path) -> None:
    """⑤ manifest.json always exists, even with zero successful files."""
    manager = DownloadManager(adapters={"fake": FakeAdapter({})}, max_retries=1)
    task_dir = tmp_path / "task"

    await manager.download_all(_results("fake", ["Ghost"]), task_dir)

    assert _read_manifest(task_dir)["files"] == []


async def test_max_concurrent_limits_parallelism(tmp_path: Path) -> None:
    """⑥ The semaphore caps the number of simultaneous downloads (F5.3 concurrency)."""
    adapter = TrackingAdapter()
    manager = DownloadManager(adapters={"tracked": adapter}, max_concurrent=2)

    await manager.download_all(_results("tracked", ["a", "b", "c", "d"]), tmp_path / "task")

    assert adapter.peak == 2


@respx.mock
async def test_generic_fallback_without_adapter(tmp_path: Path) -> None:
    """⑦ Sources without a registered adapter fall back to a direct HTTP fetch."""
    respx.get("https://example.org/files/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 body")
    )
    result = SearchResult(
        source="web",
        title="My Doc",
        url="https://example.org/doc",
        download_url="https://example.org/files/doc.pdf",
    )
    manager = DownloadManager(adapters={}, http=HttpClient(rate_limit_qps=1000.0), max_concurrent=1)
    task_dir = tmp_path / "task"

    outcomes = await manager.download_all([result], task_dir)

    assert outcomes[0].success
    assert outcomes[0].local_path is not None
    assert outcomes[0].local_path.name == "01_My_Doc.pdf"
    assert outcomes[0].local_path.read_bytes() == b"%PDF-1.4 body"
