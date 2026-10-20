"""Configuration loading for the retriever project.

Two layers of configuration:

1. Environment variables / ``.env`` file via pydantic-settings (secrets such
   as ``GITHUB_TOKEN`` — never hard-code them, see PRD §3 NF7).
2. ``config/settings.yaml`` via PyYAML (download paths, concurrency, retries).

Access the merged settings through :func:`get_settings` (a lazily-built
singleton). All fields are type-annotated.
"""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


class EnvSettings(BaseSettings):
    """Secrets and environment-level settings loaded from ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_token: str = ""
    retriever_proxy: str = ""
    # LLM credentials (Phase 2). ``LLM_API_KEY`` is the canonical name;
    # ``DEEPSEEK_API_KEY`` is accepted so an existing DeepSeek key works as-is.
    llm_api_key: str = ""
    deepseek_api_key: str = ""
    # Environment overrides for the YAML ``llm`` section (env wins).
    llm_base_url: str = ""
    llm_model: str = ""


class NetworkConfig(BaseModel):
    """``network`` section of ``config/settings.yaml``.

    ``proxy`` is the default outbound proxy for every request, e.g.
    ``http://127.0.0.1:7897`` for a local Clash mixed port. Leave it empty to
    make httpx honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` instead.
    ``RETRIEVER_PROXY`` in ``.env`` overrides this value.
    """

    proxy: str = ""


class DownloadConfig(BaseModel):
    """``download`` section of ``config/settings.yaml`` (PRD §10.2)."""

    base_dir: Path = Path("./downloads")
    max_concurrent: int = 3
    max_retries: int = 3
    timeout_seconds: int = 60


class LLMConfig(BaseModel):
    """``llm`` section of ``config/settings.yaml``, merged with env overrides.

    The API key itself never lives here — it stays in ``.env`` (PRD §3 NF7).
    ``LLM_BASE_URL`` / ``LLM_MODEL`` in the environment override these values so
    switching provider needs no edit to a tracked file.
    """

    enabled: bool = True
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 30
    max_retries: int = 2
    rate_limit_qps: float = 2.0


class Settings(BaseModel):
    """Fully merged application settings."""

    env: EnvSettings
    download: DownloadConfig = DownloadConfig()
    network: NetworkConfig = NetworkConfig()
    llm: LLMConfig = LLMConfig()
    ranking: dict = {}


def load_settings_yaml(path: Path | None = None) -> dict:
    """Load the raw YAML mapping from ``config/settings.yaml``.

    Args:
        path: Override for the settings file location (mainly for tests).

    Returns:
        Parsed YAML mapping (empty dict when the file is missing).
    """
    settings_path = path or DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        return {}
    with settings_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_settings(path: Path | None = None) -> Settings:
    """Return the application settings singleton.

    Args:
        path: Optional override for the YAML settings file (mainly for tests).

    Returns:
        A :class:`Settings` instance combining env vars and YAML config.
    """
    return _get_settings_cached(path)


@cache
def _get_settings_cached(path: Path | None = None) -> Settings:
    raw = load_settings_yaml(path)
    return Settings(
        env=EnvSettings(),
        download=DownloadConfig(**(raw.get("download") or {})),
        network=NetworkConfig(**(raw.get("network") or {})),
        llm=_build_llm_config(raw.get("llm") or {}),
        ranking=raw.get("ranking") or {},
    )


def _build_llm_config(raw_llm: dict) -> LLMConfig:
    """Build the LLM config from the YAML mapping, applying env overrides.

    Args:
        raw_llm: The ``llm`` mapping from ``config/settings.yaml``.

    Returns:
        A :class:`LLMConfig` where ``LLM_BASE_URL`` / ``LLM_MODEL`` from the
        environment take precedence over the YAML values (PRD NF4: the model
        endpoint must be configurable without touching tracked files).
    """
    config = LLMConfig(**raw_llm)
    env = EnvSettings()
    overrides: dict = {}
    if env.llm_base_url.strip():
        overrides["base_url"] = env.llm_base_url.strip()
    if env.llm_model.strip():
        overrides["model"] = env.llm_model.strip()
    return config.model_copy(update=overrides) if overrides else config


def resolve_llm_config() -> LLMConfig:
    """Return the effective LLM settings (YAML merged with environment overrides).

    Returns:
        The :class:`LLMConfig` currently in force.
    """
    return get_settings().llm


def resolve_llm_api_key() -> str:
    """Return the LLM API key, or an empty string when none is configured.

    ``LLM_API_KEY`` wins over ``DEEPSEEK_API_KEY`` so a project-specific key can
    override a machine-wide DeepSeek key. Keys are read from the environment or
    ``.env`` only — never from a tracked file (PRD §3 NF7).

    Returns:
        The API key without surrounding whitespace.
    """
    env = get_settings().env
    return (env.llm_api_key or env.deepseek_api_key).strip()


def resolve_proxy() -> str | None:
    """Return the outbound proxy to use, or ``None`` when none is configured.

    Precedence: ``RETRIEVER_PROXY`` (``.env``) > ``network.proxy``
    (``config/settings.yaml``). When both are empty this returns ``None`` so
    httpx falls back to the ``HTTP_PROXY`` / ``HTTPS_PROXY`` environment
    variables.

    Returns:
        A proxy URL such as ``"http://127.0.0.1:7897"``, or ``None``.
    """
    settings = get_settings()
    proxy = settings.env.retriever_proxy or settings.network.proxy
    return proxy.strip() or None
