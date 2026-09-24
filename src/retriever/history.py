"""History records — placeholder. Phase 2 implementation, see PRD §2.7 (F7.4)."""

from pathlib import Path


def save_history(record: dict, history_dir: Path) -> Path:
    """Persist a task record under ``history/`` and return its file path.

    Args:
        record: The task record to persist.
        history_dir: The ``./history/`` directory.

    Returns:
        Path of the written record file.

    Raises:
        NotImplementedError: Until Phase 2 (task X-4).
    """
    raise NotImplementedError("Phase 2 实现，见 PRD §2.7")
