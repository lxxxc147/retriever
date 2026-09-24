"""M5 Download Manager — placeholder. Phase 1 implementation, see PRD §2.5."""

from pathlib import Path

from .models import DownloadResult, SearchResult


class DownloadManager:
    """Concurrent download manager (semaphore-limited, hash-dedup, manifest)."""

    async def download_all(
        self, results: list[SearchResult], task_dir: Path
    ) -> list[DownloadResult]:
        """Download all results concurrently and write manifest.json.

        Args:
            results: Search results to download.
            task_dir: Per-task directory under ``downloads/{date}/{source}/``.

        Returns:
            One :class:`DownloadResult` per input result.

        Raises:
            NotImplementedError: Until Phase 1 (task B-5).
        """
        raise NotImplementedError("Phase 1 实现，见 PRD §2.5")
