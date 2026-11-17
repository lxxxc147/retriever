"""HuggingFaceAdapter — Hugging Face Hub 数据集搜索 + dataset card 下载（Phase 4）。

- 搜索：``https://huggingface.co/api/datasets``（免费、无需 Key），按 downloads
  排序，支持 ``language:`` 标签的客户端过滤；
- 下载：dataset card（README.md）经 ``/raw/main/README.md`` 拉取存为 Markdown——
  数据集本体动辄 GB 级，Phase 4 只交付"元数据 + 卡片"级下载，整仓拉取走
  ``huggingface_hub`` 的用户可自行扩展（PRD §2.9 非目标）；
- 字段映射：``downloads``/``likes`` 进 ``extra``（M4 热度维度读 ``extra["stars"]``，
  这里用 likes），``lastModified`` → ``published_at``，tags 里的 ``language:xx``
  → ``extra["language"]``。

网络全部经 ``HttpClient``（铁律 #2），失败走返回式错误而不是抛异常。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from retriever.adapters.base import SearchAdapter, classify_http_error
from retriever.http import HttpClient
from retriever.models import DownloadResult, SearchQuery, SearchResult
from retriever.utils import sanitize_filename, sha256_of, unique_path

API_URL = "https://huggingface.co/api/datasets"
DATASET_PAGE = "https://huggingface.co/datasets/{repo_id}"
CARD_URL = "https://huggingface.co/datasets/{repo_id}/raw/main/README.md"


class HuggingfaceAdapter(SearchAdapter):
    """Hugging Face Hub 数据集适配器（Phase 4 数据集源）。"""

    name: str = "huggingface"
    rate_limit_qps: float = 2.0

    def __init__(self, http: HttpClient | None = None) -> None:
        """Initialize the adapter.

        Args:
            http: Optional shared HTTP client; one is created on demand.
        """
        self._http = http or HttpClient(rate_limit_qps=self.rate_limit_qps)

    # ------------------------------------------------------------------- search
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search public datasets via the Hub API, sorted by downloads.

        Args:
            query: Structured search task (keywords + optional language filter).

        Returns:
            Parsed list of :class:`SearchResult`, capped at ``query.limit``.

        Raises:
            AdapterError: Classified failure (network/rate-limit/...) so the
                router can degrade gracefully.
        """
        params = {
            "search": " ".join(query.keywords) or query.raw_query,
            "sort": "downloads",
            "direction": -1,
            "limit": max(1, min(query.limit * 2, 100)),  # 语言过滤会再砍一刀
        }
        try:
            response = await self._http.get(API_URL, params=params)
        except Exception as exc:
            raise classify_http_error(exc, self.name) from exc
        items = response.json()
        if not isinstance(items, list):
            raise TypeError(f"unexpected Hugging Face API payload: {type(items).__name__}")

        results = [
            self._to_result(item)
            for item in items
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ]
        if query.filters.language:
            wanted = query.filters.language.strip().lower()
            results = [
                result
                for result in results
                if str(result.extra.get("language") or "").lower() == wanted
            ]
        logger.debug("HuggingfaceAdapter.search returned {} result(s)", len(results))
        return results[: query.limit]

    def _to_result(self, item: dict[str, Any]) -> SearchResult:
        """Map one Hub API entry onto the unified contract (PRD §2.3)."""
        repo_id = str(item.get("id") or "").strip()
        tags = [str(tag) for tag in (item.get("tags") or [])]
        languages = [tag.split(":", 1)[1] for tag in tags if tag.startswith("language:")]
        downloads = int(item.get("downloads") or 0)
        likes = int(item.get("likes") or 0)
        published_at = self._parse_timestamp(str(item.get("lastModified") or ""))

        tag_preview = ", ".join(tags[:5])
        abstract = f"downloads {downloads} · likes {likes}"
        if tag_preview:
            abstract += f" · {tag_preview}"

        return SearchResult(
            source=self.name,
            title=repo_id,
            abstract=abstract,
            url=DATASET_PAGE.format(repo_id=repo_id),
            download_url=CARD_URL.format(repo_id=repo_id),
            published_at=published_at,
            extra={
                "downloads": downloads,
                "likes": likes,
                "stars": likes,  # M4 热度维度读 extra["stars"]
                "tags": tags,
                "language": languages[0] if languages else None,
            },
        )

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime | None:
        """Parse the Hub's ISO timestamp; unparseable values yield ``None``."""
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    # ----------------------------------------------------------------- download
    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download the dataset card (README.md) as Markdown.

        Args:
            result: Single search result (uses ``download_url``).
            dest_dir: Target directory; created when missing.

        Returns:
            :class:`DownloadResult`; failures are reported in the result, not
            raised, so one bad file never crashes a batch.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_filename(result.title) or "dataset_card"
        try:
            response = await self._http.get(result.download_url or CARD_URL.format(repo_id=result.title))
        except Exception as exc:  # noqa: BLE001 — 报错而不是抛出
            logger.warning("HuggingfaceAdapter 下载失败 {}: {}", result.url, exc)
            return DownloadResult(success=False, error=str(exc))

        markdown = response.text
        target = unique_path(dest_dir, f"{stem}.md")
        target.write_text(markdown, encoding="utf-8")
        return DownloadResult(
            success=True,
            local_path=target,
            size_bytes=len(markdown.encode("utf-8")),
            sha256=sha256_of(markdown.encode("utf-8")),
        )
