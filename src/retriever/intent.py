"""M1 Intent Parser — placeholder. Phase 1 implementation, see PRD §2.1."""

from .models import SearchQuery


def parse_intent(raw_query: str) -> SearchQuery:
    """Parse a natural-language query into a structured ``SearchQuery``.

    Args:
        raw_query: The user's original natural-language input.

    Returns:
        A structured :class:`SearchQuery`.

    Raises:
        NotImplementedError: Until Phase 1 (task X-1).
    """
    raise NotImplementedError("Phase 1 实现，见 PRD §2.1")
