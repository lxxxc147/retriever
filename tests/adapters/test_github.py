"""Tests for GithubAdapter (B-1/B-2/B-3/B-4). 全部 mock，不打真实 API。"""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from retriever.adapters.github import (
    CODE_SEARCH_URL,
    REPO_SEARCH_URL,
    GithubAdapter,
    _ref_from_blob_url,
)
from retriever.http import HttpClient
from retriever.models import Filters, SearchQuery

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ZIP_CONTENT = b"PK\x03\x04 fake zip bytes"
COMMIT_SHA = "617986f4570d56174cf3759392e02d7af9f58fa6"


@pytest.fixture
def adapter() -> GithubAdapter:
    """GithubAdapter with a never-waiting HttpClient and a fixed token."""
    return GithubAdapter(http=HttpClient(rate_limit_qps=1000.0), token="test-token")


def _fixture(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _repo_query(**overrides: object) -> SearchQuery:
    data: dict = {
        "intent": "repo",
        "raw_query": "rag projects",
        "keywords": ["rag"],
        "filters": Filters(language="python", min_stars=100),
    }
    data.update(overrides)
    return SearchQuery(**data)  # type: ignore[arg-type]


# --------------------------------------------------------------------- B-1
@respx.mock
async def test_search_repos_parses_fixture(adapter: GithubAdapter) -> None:
    """① Repository search maps every required field onto SearchResult."""
    respx.get(REPO_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_fixture("github_repo_search.json"))
    )

    results = await adapter.search(_repo_query())

    assert len(results) == 2
    top = results[0]
    assert top.source == "github"
    assert top.title == "langchain-ai/langchain"
    assert top.url == "https://github.com/langchain-ai/langchain"
    assert top.download_url == ("https://api.github.com/repos/langchain-ai/langchain/zipball/main")
    assert top.authors == ["langchain-ai"]
    assert top.published_at is not None and top.published_at.tzinfo is not None
    assert top.extra["stars"] == 98000
    assert top.extra["language"] == "Python"
    assert top.extra["license"] == "MIT"
    assert top.extra["default_branch"] == "main"
    # /search/repositories does NOT return zipball_url; it must be constructed
    # from full_name + default_branch, otherwise repo results are undownloadable.
    assert results[1].download_url == (
        "https://api.github.com/repos/run-llama/llama_index/zipball/main"
    )


@respx.mock
async def test_search_repos_builds_filter_query(adapter: GithubAdapter) -> None:
    """② language / stars / year filters become GitHub search qualifiers."""
    route = respx.get(REPO_SEARCH_URL).mock(return_value=httpx.Response(200, json={"items": []}))

    await adapter.search(
        _repo_query(filters=Filters(language="python", min_stars=100, year_from=2023))
    )

    query_string = route.calls.last.request.url.params["q"]
    assert "language:python" in query_string
    assert "stars:>=100" in query_string
    assert "pushed:>=2023-01-01" in query_string


@respx.mock
async def test_search_repos_respects_limit(adapter: GithubAdapter) -> None:
    """③ The result list is capped at ``query.limit``."""
    respx.get(REPO_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_fixture("github_repo_search.json"))
    )

    results = await adapter.search(_repo_query(limit=1))

    assert len(results) == 1


@respx.mock
async def test_api_requests_carry_token(adapter: GithubAdapter) -> None:
    """④ Requests to api.github.com carry the bearer token in the header."""
    route = respx.get(REPO_SEARCH_URL).mock(return_value=httpx.Response(200, json={"items": []}))

    await adapter.search(_repo_query())

    assert route.calls.last.request.headers["authorization"] == "Bearer test-token"


# --------------------------------------------------------------------- B-3
@respx.mock
async def test_search_code_returns_file_results(adapter: GithubAdapter) -> None:
    """⑤ intent=code_file routes to /search/code and yields raw download URLs."""
    route = respx.get(CODE_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_fixture("github_code_search.json"))
    )

    results = await adapter.search(_repo_query(intent="code_file", keywords=["def parse_code"]))

    assert route.called
    assert len(results) == 1
    single = results[0]
    assert single.title == "foo/bar/src/retriever.py"
    # The ref comes from html_url (a commit sha here), never from a guessed
    # branch name — the embedded repository object has no default_branch.
    assert single.extra["ref"] == COMMIT_SHA
    assert single.download_url == (
        f"https://raw.githubusercontent.com/foo/bar/{COMMIT_SHA}/src/retriever.py"
    )
    assert single.extra["path"] == "src/retriever.py"
    assert single.extra["kind"] == "code_file"


@respx.mock
async def test_search_code_falls_back_to_default_branch(adapter: GithubAdapter) -> None:
    """⑤b Without a parsable html_url the repository default_branch is used."""
    respx.get(CODE_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "a.py",
                        "path": "a.py",
                        "html_url": "",
                        "repository": {
                            "full_name": "foo/bar",
                            "default_branch": "master",
                            "owner": {"login": "foo"},
                        },
                    }
                ]
            },
        )
    )

    results = await adapter.search(_repo_query(intent="code_file", keywords=["a"]))

    assert results[0].extra["ref"] == "master"
    assert results[0].download_url == "https://raw.githubusercontent.com/foo/bar/master/a.py"


class TestRefFromBlobUrl:
    """Unit tests for the html_url ref parser."""

    def test_pins_the_commit_sha(self) -> None:
        url = f"https://github.com/foo/bar/blob/{COMMIT_SHA}/src/a.py"
        assert _ref_from_blob_url(url, "foo/bar", "src/a.py") == COMMIT_SHA

    def test_handles_a_ref_containing_slashes(self) -> None:
        url = "https://github.com/foo/bar/blob/feature/new-thing/src/a.py"
        assert _ref_from_blob_url(url, "foo/bar", "src/a.py") == "feature/new-thing"

    def test_returns_none_when_not_derivable(self) -> None:
        assert _ref_from_blob_url("", "foo/bar", "a.py") is None
        assert (
            _ref_from_blob_url("https://github.com/other/repo/blob/x/a.py", "foo/bar", "a.py")
            is None
        )


# --------------------------------------------------------------------- B-2
@respx.mock
async def test_download_raw_file_names_and_hashes(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑥ Single-file download keeps a safe name, extension, size and sha256."""
    content = b"print('hello')\n"
    respx.get("https://raw.githubusercontent.com/foo/bar/main/src/retriever.py").mock(
        return_value=httpx.Response(200, content=content)
    )
    result = _raw_result(
        "My retriever.py", "https://raw.githubusercontent.com/foo/bar/main/src/retriever.py"
    )

    download = await adapter.download(result, tmp_path)

    assert download.success
    assert download.size_bytes == len(content)
    assert download.sha256 == hashlib.sha256(content).hexdigest()
    assert download.local_path is not None
    assert download.local_path.name == "My_retriever.py"
    assert download.local_path.read_bytes() == content


@respx.mock
async def test_download_zipball_gets_zip_extension(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑦ A zipball URL is saved with a .zip extension."""
    respx.get("https://api.github.com/repos/foo/bar/zipball/main").mock(
        return_value=httpx.Response(200, content=ZIP_CONTENT)
    )
    result = _raw_result("foo-bar", "https://api.github.com/repos/foo/bar/zipball/main")

    download = await adapter.download(result, tmp_path)

    assert download.success
    assert download.local_path is not None
    assert download.local_path.name == "foo-bar.zip"


@respx.mock
async def test_download_prefers_content_disposition(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑧ Content-Disposition wins over the derived file name."""
    respx.get("https://github.com/foo/bar/releases/download/v1/tool.tar.gz").mock(
        return_value=httpx.Response(
            200,
            content=b"tar",
            headers={"content-disposition": 'attachment; filename="tool-1.2.tar.gz"'},
        )
    )
    result = _raw_result(
        "ignored title", "https://github.com/foo/bar/releases/download/v1/tool.tar.gz"
    )

    download = await adapter.download(result, tmp_path)

    assert download.success
    assert download.local_path is not None
    assert download.local_path.name == "tool-1.2.tar.gz"


@respx.mock
async def test_download_403_returns_failure(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑨ A 403 (rate limit) is reported as a failure, never raised."""
    respx.get("https://raw.githubusercontent.com/foo/bar/main/x.py").mock(
        return_value=httpx.Response(403)
    )
    result = _raw_result("x", "https://raw.githubusercontent.com/foo/bar/main/x.py")

    download = await adapter.download(result, tmp_path)

    assert not download.success
    assert download.error
    assert download.local_path is None


async def test_download_without_url_returns_failure(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑩ A result without download_url cannot be downloaded."""
    download = await adapter.download(_raw_result("nothing", None), tmp_path)

    assert not download.success
    assert download.error


# --------------------------------------------------------------------- B-4
@respx.mock
async def test_download_repo_zip_builds_zipball_url(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑪ download_repo_zip hits the ref-specific zipball endpoint."""
    route = respx.get("https://api.github.com/repos/foo/bar/zipball/v1.0").mock(
        return_value=httpx.Response(200, content=ZIP_CONTENT)
    )

    download = await adapter.download_repo_zip("foo/bar", tmp_path, ref="v1.0")

    assert route.called
    assert download.success
    assert download.local_path is not None
    assert download.local_path.suffix == ".zip"


@respx.mock
async def test_latest_release_assets(adapter: GithubAdapter) -> None:
    """⑫ Release assets are listed with name/url/size."""
    respx.get("https://api.github.com/repos/foo/bar/releases/latest").mock(
        return_value=httpx.Response(
            200,
            json={
                "assets": [
                    {
                        "name": "app.zip",
                        "browser_download_url": "https://github.com/foo/bar/releases/download/v1/app.zip",
                        "size": 4096,
                    }
                ]
            },
        )
    )

    assets = await adapter.latest_release_assets("foo/bar")

    assert assets == [
        {
            "name": "app.zip",
            "url": "https://github.com/foo/bar/releases/download/v1/app.zip",
            "size": 4096,
        }
    ]


@respx.mock
async def test_download_release_asset(adapter: GithubAdapter, tmp_path: Path) -> None:
    """⑬ A release asset downloads by its browser_download_url."""
    url = "https://github.com/foo/bar/releases/download/v1/app.zip"
    respx.get(url).mock(return_value=httpx.Response(200, content=ZIP_CONTENT))

    download = await adapter.download_release_asset(url, tmp_path)

    assert download.success
    assert download.local_path is not None
    assert download.local_path.name == "app.zip"


@respx.mock
async def test_read_file_returns_text(adapter: GithubAdapter) -> None:
    """⑭ read_file fetches raw text content."""
    respx.get("https://raw.githubusercontent.com/foo/bar/main/README.md").mock(
        return_value=httpx.Response(200, text="# hello\n")
    )

    text = await adapter.read_file("foo", "bar", "README.md")

    assert text == "# hello\n"


def _raw_result(title: str, download_url: str | None):
    """Build a minimal GitHub-flavoured SearchResult for download tests."""
    from retriever.models import SearchResult

    return SearchResult(
        source="github",
        title=title,
        url=download_url or "https://github.com/foo/bar",
        download_url=download_url,
    )
