"""Data-source adapters (M3, PRD §2.3 / §4.3)."""

from .base import SearchAdapter
from .example import ExampleAdapter

__all__ = ["ExampleAdapter", "SearchAdapter"]
