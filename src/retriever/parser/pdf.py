"""M6 PDF parser — placeholder. Phase 3 implementation, see PRD §2.6 (F6.1)."""

from pathlib import Path


def parse_pdf(pdf_path: Path) -> dict:
    """Extract plain text (title/body/references sections) from a PDF.

    Args:
        pdf_path: Local path of the downloaded PDF.

    Returns:
        Parsed sections as a mapping.

    Raises:
        NotImplementedError: Until Phase 3 (task A-6).
    """
    raise NotImplementedError("Phase 3 实现，见 PRD §2.6")
