"""M6 code parser (PRD §2.6 — task B-6).

Analyses a locally downloaded repository and returns a structural summary:

- detected languages with file counts (and a ``primary_language``),
- a nested directory/file tree,
- likely entry files and dependency manifests.

Pure filesystem work: no network, no dependency on the download layer, so it can
also be used on repositories cloned by hand.
"""

from __future__ import annotations

from pathlib import Path

# Directories that never carry meaningful source (VCS, caches, build output,
# vendored dependencies).
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "dist",
    "build",
    "target",
    "out",
    "vendor",
    ".gradle",
}

# Binary / generated artifacts that must not be parsed as text.
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".dylib",
    ".o",
    ".a",
    ".class",
    ".jar",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".pdf",
    ".lock",
    ".min.js",
}

LANGUAGES: dict[str, str] = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".scala": "Scala",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".mm": "Objective-C++",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".sql": "SQL",
    ".r": "R",
    ".jl": "Julia",
    ".lua": "Lua",
    ".dart": "Dart",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".vue": "Vue",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".toml": "TOML",
    ".md": "Markdown",
    ".rst": "reStructuredText",
}

# Conventional program entry points across common ecosystems.
ENTRY_NAMES = {
    "main.py",
    "__main__.py",
    "app.py",
    "manage.py",
    "run.py",
    "cli.py",
    "wsgi.py",
    "asgi.py",
    "main.js",
    "index.js",
    "app.js",
    "server.js",
    "main.ts",
    "index.ts",
    "app.ts",
    "server.ts",
    "main.go",
    "main.rs",
    "lib.rs",
    "Main.java",
    "Program.cs",
    "main.c",
    "main.cpp",
    "index.html",
    "Dockerfile",
    "Makefile",
}

# Dependency / build manifests worth surfacing to the user.
MANIFEST_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "composer.json",
}

_MAX_TEXT_BYTES = 2_000_000  # skip line counting for larger files


def _is_skipped(rel: Path) -> bool:
    """Return ``True`` when a relative path is noise or a binary artifact."""
    if any(part in SKIP_DIRS for part in rel.parts[:-1]):
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    return rel.suffix.lower() in SKIP_SUFFIXES


def _count_lines(path: Path) -> int:
    """Count lines in a text file, tolerating encoding issues (0 on failure)."""
    try:
        if path.stat().st_size > _MAX_TEXT_BYTES:
            return 0
        with path.open(encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _build_tree(files: list[Path]) -> dict:
    """Build a nested ``{dir: {...}, file: None}`` tree from relative paths."""
    tree: dict = {}
    for rel in files:
        node = tree
        for part in rel.parts[:-1]:
            node = node.setdefault(part, {})
        node[rel.name] = None
    return tree


def parse_code(repo_dir: Path) -> dict:
    """Identify languages, build a file tree, and locate entry files (F6.2).

    Args:
        repo_dir: Local directory of the downloaded repository.

    Returns:
        A mapping with ``root``, ``total_files``, ``total_lines``,
        ``languages`` (language -> file count, most frequent first),
        ``primary_language``, ``file_tree``, ``entry_files`` and ``manifests``.

    Raises:
        NotADirectoryError: When ``repo_dir`` is not an existing directory.
    """
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {repo_dir}")

    files: list[Path] = []
    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if _is_skipped(rel):
            continue
        files.append(rel)

    languages: dict[str, int] = {}
    total_lines = 0
    entry_files: list[str] = []
    manifests: list[str] = []
    for rel in files:
        language = LANGUAGES.get(rel.suffix.lower())
        if language:
            languages[language] = languages.get(language, 0) + 1
        if rel.name in ENTRY_NAMES:
            entry_files.append(rel.as_posix())
        if rel.name in MANIFEST_NAMES:
            manifests.append(rel.as_posix())
        total_lines += _count_lines(repo_dir / rel)

    ranked = sorted(languages.items(), key=lambda item: (-item[1], item[0]))
    return {
        "root": repo_dir.name,
        "total_files": len(files),
        "total_lines": total_lines,
        "languages": dict(ranked),
        "primary_language": ranked[0][0] if ranked else None,
        "file_tree": _build_tree(files),
        "entry_files": sorted(entry_files),
        "manifests": sorted(manifests),
    }
