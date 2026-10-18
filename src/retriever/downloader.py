"""M5 Download Manager (PRD §2.5 — task B-5).

Downloads a batch of :class:`SearchResult` items concurrently and writes the
per-task ``manifest.json``. Coverage of the PRD requirements:

- **F5.1** unified layout — the caller passes the per-task directory; results
  are grouped into a sub-directory named after their source,
- **F5.2** naming ``{seq}_{safe_title}{ext}`` with automatic ``_2`` suffixes,
- **F5.3** content-hash dedup — identical bytes are only stored once,
- **F5.4** single-file download, delegated to the source adapter when one is
  registered (generic HTTP fallback otherwise),
- **F5.6** retry with exponential backoff + ``download_failures.log``,
- **F5.8** ``manifest.json`` with metadata and real on-disk paths.

F5.5 (whole-repo ZIP / release assets) lives in ``GithubAdapter`` and flows
through the adapter path unchanged; F5.7 (large-file progress) is Phase 2.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .adapters.base import SearchAdapter
from .config import get_settings
from .http import HttpClient
from .models import DownloadResult, Manifest, ManifestFile, SearchResult
from .utils import (
    build_filename,
    ensure_extension,
    sanitize_filename,
    sha256_of,
    sha256_of_file,
    unique_path,
    url_extension,
)

FAILURE_LOG = "download_failures.log"
MANIFEST_FILE = "manifest.json"
GITHUB_API_HOST = "api.github.com"


def _default_task_id() -> str:
    """Return a timestamp-based task id, e.g. ``20260214-103000``."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _scan_existing_hashes(task_dir: Path) -> dict[str, Path]:
    """Map ``sha256 -> path`` for files already present under ``task_dir``."""
    existing: dict[str, Path] = {}
    for path in Path(task_dir).rglob("*"):
        if path.is_file() and path.name not in {MANIFEST_FILE, FAILURE_LOG}:
            existing.setdefault(sha256_of_file(path), path)
    return existing


class DownloadManager:
    """Concurrent, deduplicating download manager writing a per-task manifest."""

    def __init__(
        self,
        adapters: Mapping[str, SearchAdapter] | None = None,
        http: HttpClient | None = None,
        base_dir: Path | None = None,
        max_concurrent: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            adapters: Optional ``source name -> adapter`` registry. When a
                result's source is registered its ``download`` is used;
                otherwise a generic HTTP fetch of ``download_url`` happens.
            http: Optional shared HTTP client (used by the generic fallback).
            base_dir: Root download directory; defaults to ``settings.yaml``.
            max_concurrent: Concurrent download limit; defaults to settings.
            max_retries: Attempts per file; defaults to settings.
        """
        settings = get_settings()
        self.adapters: dict[str, SearchAdapter] = dict(adapters or {})
        self._http = http or HttpClient(rate_limit_qps=8.0)
        self._owns_http = http is None
        self.base_dir = Path(base_dir) if base_dir is not None else settings.download.base_dir
        self.max_concurrent = max_concurrent or settings.download.max_concurrent
        self.max_retries = max_retries if max_retries is not None else settings.download.max_retries
        self._token = settings.env.github_token

    async def download_all(
        self,
        results: Sequence[SearchResult],
        task_dir: Path,
        *,
        task_id: str | None = None,
        query: str = "",
    ) -> list[DownloadResult]:
        """Download every result concurrently and write ``manifest.json``.

        Args:
            results: Search results to download.
            task_dir: Per-task directory (created if missing).
            task_id: Optional task id for the manifest; auto-generated if omitted.
            query: Original user query, recorded in the manifest.

        Returns:
            One :class:`DownloadResult` per input result, in the same order.
        """
        task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)
        results = list(results)
        task_id = task_id or _default_task_id()
        semaphore = asyncio.Semaphore(max(1, self.max_concurrent))
        seen_hashes = _scan_existing_hashes(task_dir)

        async def _bounded(index: int, result: SearchResult) -> DownloadResult:
            async with semaphore:
                try:
                    return await self._download_one(index, result, task_dir, seen_hashes)
                except Exception as exc:  # noqa: BLE001 — one bad file must not kill the batch
                    logger.exception("Unexpected download failure for {}", result.url)
                    return DownloadResult(success=False, error=str(exc))

        outcomes = await asyncio.gather(
            *(_bounded(index, result) for index, result in enumerate(results))
        )

        entries: list[ManifestFile] = []
        failures: list[tuple[SearchResult, str]] = []
        for result, outcome in zip(results, outcomes, strict=True):
            if outcome.success and outcome.local_path is not None:
                entries.append(
                    ManifestFile(
                        title=result.title,
                        source=result.source,
                        url=result.url,
                        local_path=outcome.local_path,
                        size_bytes=outcome.size_bytes,
                        sha256=outcome.sha256 or sha256_of_file(outcome.local_path),
                    )
                )
            else:
                failures.append((result, outcome.error or "unknown error"))

        self._write_failures(task_dir, failures)
        self._write_manifest(task_dir, task_id, query, entries)
        logger.info(
            "DownloadManager: {}/{} files saved under {}", len(entries), len(results), task_dir
        )
        return list(outcomes)

    # ------------------------------------------------------------------ internals
    async def _download_one(
        self,
        index: int,
        result: SearchResult,
        task_dir: Path,
        seen_hashes: dict[str, Path],
    ) -> DownloadResult:
        """Fetch one result, finalize its name, and apply content dedup."""
        dest_dir = task_dir / result.source
        outcome = await self._fetch_with_retry(result, dest_dir)
        if not outcome.success or outcome.local_path is None:
            return outcome

        final_path = self._finalize_name(outcome.local_path, index, result, dest_dir)
        digest = outcome.sha256 or sha256_of_file(final_path)

        duplicate = seen_hashes.get(digest)
        if duplicate is not None and duplicate != final_path:
            logger.debug("Duplicate content detected, reusing {}", duplicate)
            final_path.unlink(missing_ok=True)
            return DownloadResult(
                success=True,
                local_path=duplicate,
                size_bytes=outcome.size_bytes,
                sha256=digest,
            )

        seen_hashes[digest] = final_path
        return DownloadResult(
            success=True,
            local_path=final_path,
            size_bytes=outcome.size_bytes,
            sha256=digest,
        )

    async def _fetch_with_retry(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download one result, retrying retryable failures with backoff."""
        attempts = max(1, self.max_retries)
        for attempt in range(1, attempts + 1):
            outcome = await self._fetch(result, dest_dir)
            if outcome.success or attempt == attempts:
                return outcome
            delay = min(2 ** (attempt - 1), 8)
            logger.warning(
                "Download attempt {}/{} failed for {}: {} — retrying in {}s",
                attempt,
                attempts,
                result.url or result.title,
                outcome.error,
                delay,
            )
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _fetch(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Fetch via the registered adapter, or the generic HTTP fallback."""
        adapter = self.adapters.get(result.source)
        if adapter is not None:
            try:
                return await adapter.download(result, dest_dir)
            except Exception as exc:  # noqa: BLE001 — report instead of raising
                logger.warning("Adapter '{}' raised while downloading: {}", result.source, exc)
                return DownloadResult(success=False, error=str(exc))
        return await self._generic_download(result, dest_dir)

    async def _generic_download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download ``result.download_url`` directly (fallback path)."""
        if not result.download_url:
            return DownloadResult(success=False, error="no download_url on result")
        try:
            response = await self._http.get(
                result.download_url, headers=self._headers_for(result.download_url)
            )
        except Exception as exc:  # noqa: BLE001 — report instead of raising
            logger.warning("Generic download failed for {}: {}", result.download_url, exc)
            return DownloadResult(success=False, error=str(exc))

        content = response.content
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = ensure_extension(
            sanitize_filename(result.title), url_extension(result.download_url)
        )
        target = unique_path(dest_dir, filename)
        target.write_bytes(content)
        return DownloadResult(
            success=True,
            local_path=target,
            size_bytes=len(content),
            sha256=sha256_of(content),
        )

    def _headers_for(self, url: str) -> dict[str, str] | None:
        """Return GitHub API headers when needed, else ``None``."""
        if GITHUB_API_HOST not in url:
            return None
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @staticmethod
    def _finalize_name(path: Path, index: int, result: SearchResult, dest_dir: Path) -> Path:
        """Rename a freshly written file to ``{seq}_{safe_title}{ext}``."""
        path = Path(path)
        ext = path.suffix or url_extension(result.download_url or "")
        target = unique_path(dest_dir, build_filename(index + 1, result.title, ext))
        if path != target and path.exists():
            path.replace(target)
        return target

    @staticmethod
    def _write_failures(task_dir: Path, failures: Sequence[tuple[SearchResult, str]]) -> None:
        """Append failed downloads to ``download_failures.log`` (F5.6)."""
        if not failures:
            return
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        log_path = Path(task_dir) / FAILURE_LOG
        with log_path.open("a", encoding="utf-8") as handle:
            for result, error in failures:
                handle.write(f"{stamp}\t{result.source}\t{result.url}\t{error}\n")
        logger.warning("{} download(s) failed — details in {}", len(failures), log_path)

    @staticmethod
    def _write_manifest(
        task_dir: Path, task_id: str, query: str, entries: Sequence[ManifestFile]
    ) -> Path:
        """Write ``manifest.json`` for the task (F5.8)."""
        manifest = Manifest(
            task_id=task_id,
            query=query,
            created_at=datetime.now(UTC),
            files=list(entries),
        )
        manifest_path = Path(task_dir) / MANIFEST_FILE
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    async def aclose(self) -> None:
        """Close the internally-owned HTTP client, if any."""
        if self._owns_http:
            await self._http.aclose()
