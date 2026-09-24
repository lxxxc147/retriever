"""M2 Router — placeholder. Phase 1 implementation, see PRD §2.2."""

from .adapters.base import SearchAdapter
from .models import SearchQuery


async def route(query: SearchQuery) -> list[SearchAdapter]:
    """Select and order the data sources for a given query.

    Args:
        query: Structured search task.

    Returns:
        Enabled adapters matching the query intent.

    Raises:
        NotImplementedError: Until Phase 1.
    """
    raise NotImplementedError("Phase 1 实现，见 PRD §2.2")
