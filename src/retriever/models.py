"""Unified data contracts shared across the whole retriever project.

These models are the ONLY way data flows between modules and adapters
(see PRD §4.3, iron rule #1). All fields match the PRD contracts exactly:

- SearchQuery:  PRD §2.1 (M1 output contract)
- SearchResult: PRD §2.3 (M3 unified result contract)
- Manifest:     PRD §2.5 (M5 manifest.json contract)
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["paper", "repo", "code_file", "dataset", "web"]


class Filters(BaseModel):
    """Optional structured filter slots extracted from a user query.

    All fields are optional; adapters apply whatever they can support.
    """

    authors: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    language: str | None = None
    min_stars: int | None = None
    categories: list[str] | None = None  # arXiv category filter, e.g. ["cs.CL"]


class SearchQuery(BaseModel):
    """Structured search task produced by the intent parser (PRD §2.1)."""

    intent: Intent
    raw_query: str
    keywords: list[str]
    filters: Filters = Field(default_factory=Filters)
    limit: int = 5


class SearchResult(BaseModel):
    """Unified search result returned by every adapter (PRD §2.3)."""

    source: str  # "arxiv" | "github" | "scholar" | "web" | ...
    title: str
    abstract: str = ""
    url: str
    download_url: str | None = None
    published_at: datetime | None = None
    authors: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class DownloadResult(BaseModel):
    """Outcome of a single file download."""

    success: bool
    local_path: Path | None = None
    size_bytes: int = 0
    sha256: str | None = None
    error: str | None = None


class ManifestFile(BaseModel):
    """Single entry inside a manifest.json file list (PRD §2.5)."""

    title: str
    source: str
    url: str
    local_path: Path
    size_bytes: int
    sha256: str


class Manifest(BaseModel):
    """Download manifest written after each task (PRD §2.5)."""

    task_id: str
    query: str
    created_at: datetime
    files: list[ManifestFile]
