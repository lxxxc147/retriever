"""Tests for config loading. No real config files or network are touched."""

from pathlib import Path

from retriever.config import DownloadConfig, EnvSettings, get_settings, load_settings_yaml

SETTINGS_YAML = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class TestEnvSettings:
    def test_defaults(self) -> None:
        """Without a .env file the token defaults to empty (never hard-coded)."""
        assert EnvSettings(_env_file=None).github_token == ""


class TestLoadSettingsYaml:
    def test_loads_repo_settings(self) -> None:
        """The shipped config/settings.yaml parses with the PRD §10.2 values."""
        raw = load_settings_yaml(SETTINGS_YAML)
        assert raw["download"]["base_dir"] == "./downloads"
        assert raw["download"]["max_concurrent"] == 3
        assert raw["download"]["max_retries"] == 3
        assert raw["download"]["timeout_seconds"] == 60
        assert "ranking" in raw
        assert raw["sources"]["arxiv"]["rate_limit_qps"] == 1

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """A missing YAML file yields an empty mapping instead of crashing."""
        assert load_settings_yaml(tmp_path / "nope.yaml") == {}


class TestGetSettings:
    def test_defaults(self) -> None:
        """DownloadConfig defaults match the PRD §10.2 skeleton."""
        download = DownloadConfig()
        assert download.base_dir == Path("./downloads")
        assert download.max_concurrent == 3
        assert download.max_retries == 3
        assert download.timeout_seconds == 60

    def test_singleton(self) -> None:
        """get_settings returns the same cached instance."""
        assert get_settings() is get_settings()
