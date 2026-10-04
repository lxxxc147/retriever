"""Data-source adapters (M3, PRD §2.3 / §4.3)."""

from .base import SearchAdapter
from .example import ExampleAdapter
from .github import GithubAdapter

__all__ = ["ExampleAdapter", "GithubAdapter", "SearchAdapter"]
