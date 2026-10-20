"""Data-source registry (PRD §2.2 F2.3 / §8.3): add a source without touching code.

``config/sources.yaml`` describes every data source:

.. code-block:: yaml

    arxiv:
      enabled: true
      rate_limit_qps: 1
      adapter: retriever.adapters.arxiv:ArxivAdapter
      intents: [paper, dataset, web]

Adding a source therefore means "implement :class:`SearchAdapter` + add a YAML
entry" — the framework is untouched (PRD G5: a new adapter within one person-day).

The registry is deliberately forgiving at *runtime* and strict at *load* time:

- a malformed file raises :class:`SourceRegistryError` (the router then falls back
  to the built-in arXiv + GitHub defaults, so the CLI still works);
- a single source that cannot be imported (module not merged yet, typo, missing
  optional dependency) is skipped with a warning, and the others keep working.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError

from .adapters.base import SearchAdapter
from .http import HttpClient
from .models import Intent

DEFAULT_SOURCES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "sources.yaml"

# Used when a source does not declare ``intents`` in the registry.
FALLBACK_INTENTS: dict[str, tuple[Intent, ...]] = {
    "arxiv": ("paper", "dataset", "web"),
    "github": ("repo", "code_file", "dataset", "web"),
    "scholar": ("paper",),
    "pdf": ("paper", "web"),
    "web": ("web",),
}


class SourceRegistryError(RuntimeError):
    """Raised when ``config/sources.yaml`` is missing a usable mapping."""


class SourceSpec(BaseModel):
    """One entry of the data-source registry."""

    name: str
    adapter: str  # "package.module:ClassName"
    enabled: bool = True
    rate_limit_qps: float | None = None
    intents: list[Intent] | None = None


def load_sources(path: Path | None = None) -> list[SourceSpec]:
    """Read and validate the registry file.

    Args:
        path: Override for the registry location (tests, alternate profiles).

    Returns:
        The declared sources in file order; an empty list when the file does not
        exist (nothing configured).

    Raises:
        SourceRegistryError: When the file exists but is not a mapping of source
            entries, or an entry is missing ``adapter``.
    """
    registry_path = path or DEFAULT_SOURCES_PATH
    if not registry_path.exists():
        logger.debug("no source registry at {}, nothing configured", registry_path)
        return []

    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SourceRegistryError(f"{registry_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SourceRegistryError(f"{registry_path}: expected a mapping of source entries")

    specs: list[SourceSpec] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise SourceRegistryError(f"{registry_path}: source {name!r} must be a mapping")
        try:
            specs.append(SourceSpec(name=str(name), **entry))
        except ValidationError as exc:
            raise SourceRegistryError(f"{registry_path}: source {name!r} is invalid: {exc}") from exc
    return specs


def _http_for(spec: SourceSpec) -> HttpClient:
    """Build the HTTP client a source should use (rate limit from the registry)."""
    qps = spec.rate_limit_qps if spec.rate_limit_qps and spec.rate_limit_qps > 0 else 2.0
    return HttpClient(rate_limit_qps=qps)


def build_adapter(spec: SourceSpec) -> SearchAdapter:
    """Import, instantiate and configure one adapter from its registry entry.

    Adapters in this repo accept their HTTP client either as ``http`` (arXiv,
    GitHub) or ``http_client`` (Scholar, Web, PDF); the signature is inspected so
    both conventions work without the adapters changing.

    Args:
        spec: The registry entry.

    Returns:
        A ready-to-use adapter instance.

    Raises:
        SourceRegistryError: When the class cannot be imported, is not a
            :class:`SearchAdapter`, or cannot be constructed.
    """
    module_name, _, class_name = spec.adapter.partition(":")
    if not module_name or not class_name:
        raise SourceRegistryError(f"source {spec.name!r}: expected 'module:Class', got {spec.adapter!r}")

    try:
        module = importlib.import_module(module_name)
        adapter_class = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise SourceRegistryError(
            f"source {spec.name!r}: cannot import {spec.adapter!r} ({exc})"
        ) from exc

    if not (inspect.isclass(adapter_class) and issubclass(adapter_class, SearchAdapter)):
        raise SourceRegistryError(
            f"source {spec.name!r}: {spec.adapter!r} is not a SearchAdapter subclass"
        )

    parameters = inspect.signature(adapter_class.__init__).parameters
    kwargs: dict[str, object] = {}
    if "http" in parameters:
        kwargs["http"] = _http_for(spec)
    elif "http_client" in parameters:
        kwargs["http_client"] = _http_for(spec)

    try:
        adapter = adapter_class(**kwargs)
    except Exception as exc:  # one broken source must not stop the others
        raise SourceRegistryError(f"source {spec.name!r}: cannot instantiate ({exc})") from exc

    if spec.name != getattr(adapter, "name", spec.name):
        logger.warning(
            "source {!r} declares adapter name {!r}; routing uses the adapter's own name",
            spec.name,
            getattr(adapter, "name", None),
        )
    if spec.rate_limit_qps is not None:
        adapter.rate_limit_qps = spec.rate_limit_qps
    return adapter


def load_adapters(path: Path | None = None) -> tuple[list[SearchAdapter], dict[str, list[Intent]]]:
    """Load every enabled source, together with its intent routing table.

    Args:
        path: Override for the registry location.

    Returns:
        ``(adapters, intent_map)`` — adapters in registry order, and a mapping of
        adapter name to the intents it answers (only for sources that declare
        ``intents``; the caller falls back to its built-in table otherwise).

    Raises:
        SourceRegistryError: When the registry file itself is unusable.
    """
    specs = load_sources(path)
    adapters: list[SearchAdapter] = []
    intent_map: dict[str, list[Intent]] = {}
    for spec in specs:
        if not spec.enabled:
            logger.debug("source {} disabled in the registry, skipped", spec.name)
            continue
        try:
            adapter = build_adapter(spec)
        except SourceRegistryError as exc:
            logger.warning("skipping source: {}", exc)
            continue
        adapters.append(adapter)
        intents = spec.intents or list(FALLBACK_INTENTS.get(adapter.name, ()))
        if intents:
            intent_map[adapter.name] = intents
    return adapters, intent_map
