"""M7 Presenter — placeholder. Phase 1 implementation, see PRD §2.7."""

from .models import SearchResult


def render(results: list[SearchResult]) -> str:
    """Render ranked results as the structured CLI report (PRD §2.7).

    Args:
        results: Ranked search results.

    Returns:
        The formatted report text shown to the user.

    Raises:
        NotImplementedError: Until Phase 1 (task X-3).
    """
    raise NotImplementedError("Phase 1 实现，见 PRD §2.7")
