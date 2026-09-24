"""M4 Result Pipeline — placeholder. Phase 2 implementation, see PRD §2.4."""

from .models import SearchResult


def run_pipeline(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate, score, rank, and summarize search results.

    Args:
        results: Raw results aggregated from all adapters.

    Returns:
        Deduplicated, ranked results.

    Raises:
        NotImplementedError: Until Phase 2 (task X-2).
    """
    raise NotImplementedError("Phase 2 实现，见 PRD §2.4")
