"""Tests for the data-source registry (PRD §2.2 F2.3).

No network: adapters are the bundled ExampleAdapter or local stubs, and every
registry file is written into ``tmp_path``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from retriever.adapters.base import SearchAdapter
from retriever.adapters.example import ExampleAdapter
from retriever.intent import ParsedIntent
from retriever.models import DownloadResult, Filters, SearchQuery, SearchResult
from retriever.router import Router
from retriever.sources import (
    FALLBACK_INTENTS,
    SourceRegistryError,
    build_adapter,
    load_adapters,
    load_sources,
)

EXAMPLE = "retriever.adapters.example:ExampleAdapter"


class _HttpKwargAdapter(SearchAdapter):
    """Adapter using the ``http=`` constructor convention (arXiv/GitHub style)."""

    name = "stub-http"
    rate_limit_qps = 1.0

    def __init__(self, http: object = None) -> None:
        self.http = http

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return []

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        return DownloadResult(success=False, error="stub")


class _HttpClientKwargAdapter(SearchAdapter):
    """Adapter using the ``http_client=`` convention (Scholar/Web/PDF style)."""

    name = "stub-http-client"
    rate_limit_qps = 1.0

    def __init__(self, http_client: object = None) -> None:
        self.http_client = http_client

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        return []

    async def download(self, result: SearchResult, dest_dir: Path) -> DownloadResult:
        return DownloadResult(success=False, error="stub")


@pytest.fixture
def registry(tmp_path: Path):
    """Write a registry file and return its path."""

    def _write(text: str) -> Path:
        path = tmp_path / "sources.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> str:
    """Expose the local stub adapters as an importable module."""
    module = types.ModuleType("retriever_test_stubs")
    module.HttpKwarg = _HttpKwargAdapter
    module.HttpClientKwarg = _HttpClientKwargAdapter
    monkeypatch.setitem(sys.modules, "retriever_test_stubs", module)
    return "retriever_test_stubs"


def _parsed(intent: str) -> ParsedIntent:
    """Build a ParsedIntent for routing tests."""
    return ParsedIntent(  # type: ignore[arg-type]
        intent=intent, keywords=["x"], raw_query="x", filters=Filters()
    )


def _query(intent: str) -> SearchQuery:
    """Build a SearchQuery for routing tests."""
    return SearchQuery(intent=intent, raw_query="x", keywords=["x"])  # type: ignore[arg-type]


# ------------------------------------------------------------------ load_sources
def test_shipped_registry_is_valid_and_ordered() -> None:
    """The repository's own config/sources.yaml parses and keeps file order."""
    specs = load_sources()

    assert [spec.name for spec in specs][:2] == ["arxiv", "github"]
    arxiv = specs[0]
    assert arxiv.adapter == "retriever.adapters.arxiv:ArxivAdapter"
    assert arxiv.enabled is True
    assert arxiv.intents == ["paper", "dataset", "web"]


def test_phase2_sources_are_registered_and_enabled() -> None:
    """Scholar/Web/PDF are registered and enabled; turning one off needs no code."""
    by_name = {spec.name: spec for spec in load_sources()}

    for name in ("scholar", "web", "pdf"):
        assert by_name[name].enabled is True
        assert ":" in by_name[name].adapter


def test_default_router_registers_every_enabled_source() -> None:
    """``Router()`` loads all five enabled sources straight from the registry."""
    router = Router()

    assert [adapter.name for adapter in router.adapters] == [
        "arxiv",
        "github",
        "scholar",
        "web",
        "pdf",
    ]


def test_registry_intents_decide_routing() -> None:
    """The registry table routes intents: paper → arxiv + scholar, web → four sources."""
    router = Router()

    assert [adapter.name for adapter in router.route(_parsed("paper"))] == ["arxiv", "scholar"]
    assert [adapter.name for adapter in router.route(_parsed("web"))] == [
        "arxiv",
        "github",
        "web",
        "pdf",
    ]
    assert [adapter.name for adapter in router.route(_parsed("repo"))] == ["github"]


def test_missing_registry_is_not_an_error(tmp_path: Path) -> None:
    """No registry file simply means "nothing configured"."""
    assert load_sources(tmp_path / "absent.yaml") == []


def test_non_mapping_registry_is_rejected(registry) -> None:
    """A YAML list instead of a mapping is a configuration error."""
    with pytest.raises(SourceRegistryError, match="mapping"):
        load_sources(registry("- arxiv\n- github\n"))


def test_entry_without_adapter_is_rejected(registry) -> None:
    """An entry that cannot name its adapter is rejected at load time."""
    with pytest.raises(SourceRegistryError, match="invalid"):
        load_sources(registry("arxiv:\n  enabled: true\n"))


def test_broken_yaml_is_rejected(registry) -> None:
    """Unparsable YAML surfaces as a registry error."""
    with pytest.raises(SourceRegistryError):
        load_sources(registry("arxiv: [unclosed\n"))


# ----------------------------------------------------------------- build_adapter
def test_build_adapter_imports_and_applies_the_rate_limit() -> None:
    """A real adapter is constructed with a client and the registry's qps."""
    adapter = build_adapter(load_sources()[0])  # arxiv

    assert isinstance(adapter, SearchAdapter)
    assert adapter.name == "arxiv"
    assert adapter.rate_limit_qps == 1.0


def test_build_adapter_supports_both_http_conventions(registry, fake_modules: str) -> None:
    """``http=`` and ``http_client=`` adapters both receive a registry client."""
    path = registry(
        f"a:\n  adapter: {fake_modules}:HttpKwarg\n"
        f"b:\n  adapter: {fake_modules}:HttpClientKwarg\n"
    )
    first, second = (build_adapter(spec) for spec in load_sources(path))

    assert first.http is not None
    assert second.http_client is not None


def test_build_adapter_rejects_unknown_module(registry) -> None:
    """A source that cannot be imported names the broken entry."""
    spec = load_sources(registry("ghost:\n  adapter: retriever.adapters.ghost:Ghost\n"))[0]

    with pytest.raises(SourceRegistryError, match="cannot import"):
        build_adapter(spec)


def test_build_adapter_rejects_non_adapter_class(registry) -> None:
    """Something that is not a SearchAdapter is a registry error, not a crash later."""
    spec = load_sources(registry("bad:\n  adapter: retriever.models:SearchQuery\n"))[0]

    with pytest.raises(SourceRegistryError, match="not a SearchAdapter"):
        build_adapter(spec)


def test_build_adapter_rejects_malformed_pointer(registry) -> None:
    """``module:Class`` is required."""
    spec = load_sources(registry("bad:\n  adapter: retriever.models\n"))[0]

    with pytest.raises(SourceRegistryError, match="module:Class"):
        build_adapter(spec)


# ---------------------------------------------------------------- load_adapters
def test_disabled_sources_are_skipped(registry) -> None:
    """``enabled: false`` keeps an entry documented but out of the router."""
    path = registry(f"example:\n  enabled: false\n  adapter: {EXAMPLE}\n")

    assert load_adapters(path) == ([], {})


def test_one_broken_source_does_not_break_the_rest(registry) -> None:
    """A not-yet-merged adapter is skipped; working sources still load."""
    path = registry(
        f"ghost:\n  adapter: retriever.adapters.ghost:Ghost\nexample:\n  adapter: {EXAMPLE}\n"
    )
    adapters, _ = load_adapters(path)

    assert [adapter.name for adapter in adapters] == [ExampleAdapter.name]


def test_intent_map_comes_from_the_registry(registry) -> None:
    """Declared ``intents`` become the routing table."""
    path = registry(f"example:\n  adapter: {EXAMPLE}\n  intents: [paper, dataset]\n")

    assert load_adapters(path)[1][ExampleAdapter.name] == ["paper", "dataset"]


def test_intent_map_falls_back_to_the_builtin_table(registry) -> None:
    """Without ``intents`` the built-in default table for that source is used."""
    path = registry("arxiv:\n  adapter: retriever.adapters.arxiv:ArxivAdapter\n")

    assert load_adapters(path)[1]["arxiv"] == list(FALLBACK_INTENTS["arxiv"])


# ---------------------------------------------------------------------- router
def test_router_routes_by_the_registry(registry) -> None:
    """The router consults the registry mapping before its built-in table."""
    path = registry(f"example:\n  adapter: {EXAMPLE}\n  intents: [web]\n")
    router = Router(sources_path=path)

    assert [adapter.name for adapter in router.route(_parsed("web"))] == [ExampleAdapter.name]
    assert router.route(_parsed("paper")) == []  # not declared for paper


def test_router_with_an_empty_registry_keeps_builtin_defaults(registry) -> None:
    """An all-disabled registry falls back to arXiv + GitHub rather than nothing."""
    path = registry("arxiv:\n  enabled: false\n  adapter: retriever.adapters.arxiv:ArxivAdapter\n")
    router = Router(sources_path=path)

    assert [adapter.name for adapter in router.adapters] == ["arxiv", "github"]


def test_router_with_a_broken_registry_keeps_builtin_defaults(registry) -> None:
    """A malformed registry degrades to the defaults instead of crashing."""
    router = Router(sources_path=registry("- not a mapping\n"))

    assert [adapter.name for adapter in router.adapters] == ["arxiv", "github"]


def test_router_searches_through_a_registered_source(registry, fake_modules: str) -> None:
    """A registry-loaded adapter is really used by ``search_all``."""
    path = registry(f"stub:\n  adapter: {fake_modules}:HttpKwarg\n  intents: [web]\n")
    router = Router(sources_path=path)
    adapter = router.adapters[0]

    assert router.search_all(_query("web"), adapters=[adapter]) == []
    assert router.route(_parsed("web")) == [adapter]


def test_explicit_adapters_bypass_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing adapters keeps the old behaviour and never reads the file."""

    def boom(path: Path | None = None) -> None:
        raise AssertionError("registry must not be read when adapters are passed")

    monkeypatch.setattr("retriever.router.load_adapters", boom)
    adapter = ExampleAdapter()
    router = Router(adapters=[adapter])

    assert router.adapters == [adapter]
