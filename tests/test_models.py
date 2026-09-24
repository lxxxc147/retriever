"""Tests for data models (SearchQuery / SearchResult / Manifest)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from retriever.models import (
    DownloadResult,
    Manifest,
    ManifestFile,
    SearchQuery,
)


class TestSearchQuery:
    def test_defaults(self, sample_query) -> None:
        """limit defaults to 5 and filters to an empty Filters model."""
        query = sample_query()
        assert query.limit == 5
        assert query.filters.authors is None
        assert query.filters.year_from is None
        assert query.filters.year_to is None
        assert query.filters.language is None
        assert query.filters.min_stars is None

    def test_missing_required_fields(self) -> None:
        """Missing required fields must raise ValidationError."""
        with pytest.raises(ValidationError):
            SearchQuery(raw_query="only raw query")

    def test_intent_literal_validation(self, sample_query) -> None:
        """intent must be one of the five allowed literal values."""
        query = sample_query(intent="repo")
        assert query.intent == "repo"
        with pytest.raises(ValidationError):
            sample_query(intent="music")

    def test_filters_optional_fields(self, sample_query) -> None:
        """Filters accepts any subset of its optional fields."""
        query = sample_query(
            intent="repo",
            filters={"authors": ["vaswani"], "language": "python", "min_stars": 100},
        )
        assert query.filters.language == "python"
        assert query.filters.min_stars == 100


class TestSearchResult:
    def test_defaults(self, sample_result) -> None:
        """authors and extra default to empty containers; nullable fields are None."""
        result = sample_result()
        assert result.authors == []
        assert result.extra == {}
        assert result.download_url is None
        assert result.published_at is None

    def test_extra_metadata(self, sample_result) -> None:
        """extra carries source-specific metadata such as citations/stars."""
        result = sample_result(extra={"citations": 120, "stars": 2300})
        assert result.extra["stars"] == 2300


class TestDownloadResult:
    def test_failure_result(self) -> None:
        """A failed download has success=False and an error message."""
        result = DownloadResult(success=False, error="no download_url on result")
        assert result.local_path is None
        assert result.size_bytes == 0
        assert result.sha256 is None


class TestManifest:
    def test_serialization_roundtrip(self) -> None:
        """Manifest serializes to the PRD §2.5 manifest.json shape."""
        manifest = Manifest(
            task_id="20260214-001",
            query="LLM agent survey 2024",
            created_at=datetime(2026, 2, 14, 10, 30, 0, tzinfo=UTC),
            files=[
                ManifestFile(
                    title="A Survey on LLM-based Agents",
                    source="arxiv",
                    url="https://arxiv.org/abs/2401.xxxx",
                    local_path=Path("downloads/2026-02-14/arxiv/1_A_Survey.pdf"),
                    size_bytes=1234567,
                    sha256="abcd" * 16,
                )
            ],
        )
        data = manifest.model_dump(mode="json")
        assert data["task_id"] == "20260214-001"
        # Path serializes with OS-native separators; compare as Path objects.
        assert Path(data["files"][0]["local_path"]) == Path(
            "downloads/2026-02-14/arxiv/1_A_Survey.pdf"
        )
        assert data["files"][0]["sha256"] == "abcd" * 16
        assert Manifest.model_validate(data) == manifest
