"""GitHub data-source adapter (M3, PRD §2.3 — tasks B-1 / B-2 / B-3 / B-4).

Implements :class:`SearchAdapter` for GitHub and covers four capabilities:

- **B-1** repository search — ``search_repos`` (``/search/repositories``)
- **B-3** code search — ``search_code`` (``/search/code``)
- **B-2** single-file download (``raw.githubusercontent.com``)
- **B-4** whole-repo ZIP (zipball) and release-asset downloads

:meth:`GithubAdapter.search` dispatches on :attr:`SearchQuery.intent`:
``code_file`` routes to the code-search endpoint, everything else to repository
search.

Design constraints (PRD §4.3): all HTTP goes through the shared ``HttpClient``
(iron rule #2), and this adapter never imports another adapter (iron rule #1).
Requests to ``api.github.com`` carry the optional ``GITHUB_TOKEN`` from ``.env``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger

from ..config import get_settings
from ..http import HttpClient
from ..models import DownloadResult, SearchQuery, SearchResult
from ..utils import ensure_extension, sanitize_filename, sha256_of, unique_path, url_extension
from .base import SearchAdapter

API_BASE = "https://api.github.com"
REPO_SEARCH_URL = f"{API_BASE}/search/repositories"
CODE_SEARCH_URL = f"{API_BASE}/search/code"
RAW_BASE = "https://raw.githubusercontent.com"

# Accept header required by the GitHub REST API; also needed so the zipball
# endpoint answers with a redirect instead of an error.
_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"
_DISPOSITION = re.compile(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", re.IGNORECASE)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp from the GitHub API.

    Args:
        value: Raw value from the API (usually ``"2024-06-01T10:00:00Z"``).

    Returns:
        A timezone-aware :class:`datetime`, or ``None`` when unparseable.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.debug("Could not parse GitHub timestamp: {!r}", value)
        return None


def _ref_from_blob_url(html_url: Any, full_name: str, path: str) -> str | None:
    """Extract the git ref from a GitHub blob URL.

    ``/search/code`` does not return ``default_branch``, but its ``html_url``
    pins the exact ref: ``https://github.com/<owner>/<repo>/blob/<ref>/<path>``.
    Deriving the ref from there is both simpler and more precise than guessing
    a branch name (which 404s whenever the repository does not use ``main``).

    Args:
        html_url: The ``html_url`` field of a code-search item.
        full_name: ``owner/repo`` of the containing repository.
        path: File path inside the repository.

    Returns:
        The ref (commit sha, branch, or tag), or ``None`` when not derivable.
    """
    if not isinstance(html_url, str) or not full_name or not path:
        return None
    prefix = f"https://github.com/{full_name}/blob/"
    suffix = f"/{path}"
    if html_url.startswith(prefix) and html_url.endswith(suffix):
        ref = html_url[len(prefix) : -len(suffix)]
        return ref or None
    return None


class GithubAdapter(SearchAdapter):
    """GitHub adapter: repository search, code search, and file/ZIP downloads."""

    name: str = "github"
    rate_limit_qps: float = 2.0

    def __init__(self, http: HttpClient | None = None, token: str | None = None) -> None:
        """Initialize the adapter.

        Args:
            http: Optional shared HTTP client; one is created on demand.
            token: Optional GitHub token; defaults to ``GITHUB_TOKEN`` from env.
        """
        self._http = http or HttpClient(rate_limit_qps=self.rate_limit_qps)
        self._token = token if token is not None else get_settings().env.github_token
        if not self._token:
            logger.debug("GITHUB_TOKEN not set — using unauthenticated GitHub API (low rate limit)")

    # ------------------------------------------------------------------ headers
    def _api_headers(self) -> dict[str, str]:
        """Build the headers required for ``api.github.com`` requests."""
        headers = {"Accept": _ACCEPT, "X-GitHub-Api-Version": _API_VERSION}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _headers_for_url(self, url: str) -> dict[str, str] | None:
        """Return API headers only for ``api.github.com`` URLs, else ``None``."""
        return self._api_headers() if url.startswith(API_BASE) else None

    # ------------------------------------------------------------------- search
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search GitHub; dispatch by intent (code file vs. repository).

        Args:
            query: Structured search task.

        Returns:
            Parsed list of :class:`SearchResult`, capped at ``query.limit``.
        """
        if query.intent == "code_file":
            return await self.search_code(query)
        return await self.search_repos(query)

    @staticmethod
    def _base_terms(query: SearchQuery) -> list[str]:
        """Keywords to search for, falling back to the raw query text."""
        if query.keywords:
            return list(query.keywords)
        return [query.raw_query] if query.raw_query else []

    def _repo_query_string(self, query: SearchQuery) -> str:
        """Translate a :class:`SearchQuery` into a GitHub repository-search ``q``.

        Supports the slots the PRD requires for repos: keywords, owner
        (``authors``), language, minimum stars, and an update-year range.
        """
        parts = self._base_terms(query)
        filters = query.filters
        if filters.authors:
            parts.extend(f"user:{author}" for author in filters.authors)
        if filters.language:
            parts.append(f"language:{filters.language}")
        if filters.min_stars:
            parts.append(f"stars:>={filters.min_stars}")
        if filters.year_from:
            parts.append(f"pushed:>={filters.year_from}-01-01")
        if filters.year_to:
            parts.append(f"pushed:<={filters.year_to}-12-31")
        return " ".join(part for part in parts if part).strip() or "stars:>0"

    async def search_repos(self, query: SearchQuery) -> list[SearchResult]:
        """Search repositories via ``/search/repositories`` (B-1, F3.2).

        Args:
            query: Structured search task (keywords + repo filters).

        Returns:
            Repository results sorted by stars.
        """
        params = {
            "q": self._repo_query_string(query),
            "sort": "stars",
            "order": "desc",
            "per_page": max(1, min(query.limit, 100)),
        }
        response = await self._http.get(REPO_SEARCH_URL, params=params, headers=self._api_headers())
        items = response.json().get("items", [])
        results = [self._repo_to_result(item) for item in items]
        logger.debug("GithubAdapter.search_repos returned {} results", len(results))
        return results[: query.limit]

    @staticmethod
    def _repo_to_result(item: dict) -> SearchResult:
        """Map one ``/search/repositories`` item onto :class:`SearchResult`.

        Note: the repository-search payload does **not** include
        ``zipball_url``, so the archive URL is built from ``full_name`` and
        ``default_branch``; without it a repo result would have no
        ``download_url`` at all.
        """
        owner = (item.get("owner") or {}).get("login")
        license_info = item.get("license") or {}
        full_name = item.get("full_name") or item.get("name") or ""
        default_branch = item.get("default_branch")
        zipball_url = item.get("zipball_url")
        if not zipball_url and full_name:
            suffix = f"/{default_branch}" if default_branch else ""
            zipball_url = f"{API_BASE}/repos/{full_name}/zipball{suffix}"
        return SearchResult(
            source="github",
            title=full_name,
            abstract=item.get("description") or "",
            url=item.get("html_url") or "",
            download_url=zipball_url,
            published_at=_parse_dt(item.get("pushed_at") or item.get("updated_at")),
            authors=[owner] if owner else [],
            extra={
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
                "open_issues": item.get("open_issues_count", 0),
                "language": item.get("language"),
                "license": license_info.get("spdx_id"),
                "default_branch": default_branch,
                "topics": item.get("topics", []),
                "updated_at": item.get("updated_at"),
            },
        )

    async def search_code(self, query: SearchQuery) -> list[SearchResult]:
        """Search code files via ``/search/code`` (B-3, F3.3).

        Args:
            query: Structured search task (keywords, optional language).

        Returns:
            File-level results; each ``download_url`` points at raw content.
        """
        parts = self._base_terms(query)
        if query.filters.language:
            parts.append(f"language:{query.filters.language}")
        params = {
            "q": " ".join(part for part in parts if part).strip() or "readme",
            "per_page": max(1, min(query.limit, 100)),
        }
        response = await self._http.get(CODE_SEARCH_URL, params=params, headers=self._api_headers())
        items = response.json().get("items", [])
        results = [self._code_to_result(item) for item in items]
        logger.debug("GithubAdapter.search_code returned {} results", len(results))
        return results[: query.limit]

    @staticmethod
    def _code_to_result(item: dict) -> SearchResult:
        """Map one ``/search/code`` item onto a file-level :class:`SearchResult`.

        The ref is taken from ``html_url`` (which pins the exact commit)
        because the embedded ``repository`` object carries no
        ``default_branch``.
        """
        repo = item.get("repository") or {}
        full_name = repo.get("full_name") or ""
        path = item.get("path") or ""
        html_url = item.get("html_url") or ""
        ref = _ref_from_blob_url(html_url, full_name, path) or repo.get("default_branch") or "main"
        raw_url = f"{RAW_BASE}/{full_name}/{ref}/{path}" if full_name and path else None
        owner = (repo.get("owner") or {}).get("login")
        return SearchResult(
            source="github",
            title=f"{full_name}/{path}".strip("/") or path,
            abstract="",
            url=html_url,
            download_url=raw_url,
            authors=[owner] if owner else [],
            extra={
                "kind": "code_file",
                "repo": full_name,
                "path": path,
                "ref": ref,
                "name": item.get("name") or Path(path).name,
            },
        )

    # ---------------------------------------------------------------- downloads
    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        """Download the file referenced by ``result.download_url`` (B-2/B-4).

        Handles raw files, ``zipball`` archives, and release assets alike: a
        ``Content-Disposition`` file name wins when present, otherwise the file
        name is derived from the result title plus the URL extension (``.zip``
        for zipballs).

        Args:
            result: Search result whose ``download_url`` will be fetched.
            dest_dir: Destination directory (created if missing).

        Returns:
            A :class:`DownloadResult`; failures are reported, never raised.
        """
        url = result.download_url
        if not url:
            return DownloadResult(success=False, error="no download_url on result")

        try:
            response = await self._http.get(url, headers=self._headers_for_url(url))
        except Exception as exc:  # noqa: BLE001 — failures must not raise
            logger.warning("GithubAdapter.download failed for {}: {}", url, exc)
            return DownloadResult(success=False, error=str(exc))

        content = response.content
        filename = self._filename_for(result, url, response)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = unique_path(dest_dir, filename)
        local_path.write_bytes(content)
        logger.debug("GithubAdapter downloaded {} bytes -> {}", len(content), local_path)
        return DownloadResult(
            success=True,
            local_path=local_path,
            size_bytes=len(content),
            sha256=sha256_of(content),
        )

    def _filename_for(self, result: SearchResult, url: str, response: Any) -> str:
        """Choose a safe, correctly-extensioned file name for a download."""
        from_disposition = self._disposition_filename(response)
        if from_disposition:
            return from_disposition
        if "/zipball" in url:
            return ensure_extension(sanitize_filename(result.title), ".zip")
        return ensure_extension(sanitize_filename(result.title), url_extension(url))

    @staticmethod
    def _disposition_filename(response: Any) -> str | None:
        """Extract a sanitized file name from a ``Content-Disposition`` header."""
        header = response.headers.get("content-disposition")
        if not header:
            return None
        match = _DISPOSITION.search(header)
        if not match:
            return None
        raw = unquote(match.group(1).strip().strip('"'))
        return f"{sanitize_filename(Path(raw).stem)}{Path(raw).suffix.lower()}"

    async def download_repo_zip(
        self, full_name: str, dest_dir: Path, ref: str | None = None
    ) -> DownloadResult:
        """Download an entire repository as a ZIP archive (B-4, F5.5).

        Args:
            full_name: ``owner/repo`` identifier.
            dest_dir: Destination directory.
            ref: Optional branch, tag, or commit; defaults to the default branch.

        Returns:
            A :class:`DownloadResult` pointing at the saved ``.zip``.
        """
        suffix = f"/{ref}" if ref else ""
        result = SearchResult(
            source=self.name,
            title=f"{full_name.replace('/', '-')}{'-' + ref if ref else ''}",
            url=f"https://github.com/{full_name}",
            download_url=f"{API_BASE}/repos/{full_name}/zipball{suffix}",
            extra={"kind": "repo_zip", "repo": full_name, "ref": ref},
        )
        return await self.download(result, dest_dir)

    async def latest_release_assets(self, full_name: str) -> list[dict]:
        """List the assets of a repository's latest release (B-4, F5.5).

        Args:
            full_name: ``owner/repo`` identifier.

        Returns:
            One mapping per asset with ``name``, ``url``, and ``size``.
        """
        url = f"{API_BASE}/repos/{full_name}/releases/latest"
        response = await self._http.get(url, headers=self._api_headers())
        assets = response.json().get("assets", [])
        return [
            {
                "name": asset.get("name"),
                "url": asset.get("browser_download_url"),
                "size": asset.get("size", 0),
            }
            for asset in assets
        ]

    async def download_release_asset(
        self, asset_url: str, dest_dir: Path, title: str | None = None
    ) -> DownloadResult:
        """Download a release asset by its ``browser_download_url`` (B-4, F5.5).

        Args:
            asset_url: Direct ``browser_download_url`` of the asset.
            dest_dir: Destination directory.
            title: Optional display title; defaults to the URL file name.

        Returns:
            A :class:`DownloadResult`.
        """
        fallback = asset_url.rsplit("/", 1)[-1].split("?")[0] or "release_asset"
        result = SearchResult(
            source=self.name,
            title=title or (Path(fallback).stem or "release_asset"),
            url=asset_url,
            download_url=asset_url,
            extra={"kind": "release_asset"},
        )
        return await self.download(result, dest_dir)

    async def read_file(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Read a single file's text content from ``raw.githubusercontent.com``.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Path of the file inside the repository.
            ref: Branch, tag, or commit (defaults to ``main``).

        Returns:
            The decoded text content of the file.
        """
        url = f"{RAW_BASE}/{owner}/{repo}/{ref}/{path}"
        response = await self._http.get(url)
        return response.text
