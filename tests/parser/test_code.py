"""Tests for the code parser (task B-6)."""

from pathlib import Path

import pytest

from retriever.parser.code import parse_code


def _make_repo(root: Path) -> None:
    """Create a small mixed-language repo with noise directories."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('hi')\nprint('there')\n", encoding="utf-8")
    (root / "src" / "utils.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "view.js").write_text("console.log(1)\n", encoding="utf-8")
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("gitdir\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("// dep\n", encoding="utf-8")
    (root / "src" / "__pycache__").mkdir()
    (root / "src" / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00\x01")


def test_parse_code_reports_languages_tree_and_entries(tmp_path: Path) -> None:
    """① Languages, tree, entry files and manifests are detected; noise is skipped."""
    _make_repo(tmp_path)

    info = parse_code(tmp_path)

    assert info["root"] == tmp_path.name
    assert info["total_files"] == 5
    assert info["languages"]["Python"] == 2
    assert info["languages"]["JavaScript"] == 1
    assert info["primary_language"] == "Python"
    assert info["total_lines"] == 6
    assert info["manifests"] == ["pyproject.toml"]
    assert info["entry_files"] == ["src/main.py"]


def test_parse_code_tree_excludes_noise_dirs(tmp_path: Path) -> None:
    """② VCS, vendored, and cache directories never appear in the file tree."""
    _make_repo(tmp_path)

    tree = parse_code(tmp_path)["file_tree"]

    assert set(tree) == {"README.md", "pyproject.toml", "src"}
    assert set(tree["src"]) == {"main.py", "utils.py", "view.js"}
    assert ".git" not in tree
    assert "node_modules" not in tree
    assert "__pycache__" not in tree["src"]


def test_parse_code_rejects_missing_directory(tmp_path: Path) -> None:
    """③ A non-existent directory raises NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        parse_code(tmp_path / "does-not-exist")


def test_parse_code_empty_dir(tmp_path: Path) -> None:
    """④ An empty directory yields zeroed-out stats instead of crashing."""
    info = parse_code(tmp_path)

    assert info["total_files"] == 0
    assert info["languages"] == {}
    assert info["primary_language"] is None
    assert info["file_tree"] == {}
    assert info["entry_files"] == []
