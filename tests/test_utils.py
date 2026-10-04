"""Tests for the shared helpers in ``retriever.utils``."""

import hashlib
from pathlib import Path

from retriever.utils import (
    build_filename,
    ensure_extension,
    sanitize_filename,
    sha256_of,
    sha256_of_file,
    unique_path,
    url_extension,
)


class TestSanitizeFilename:
    def test_replaces_illegal_chars_and_spaces(self) -> None:
        assert sanitize_filename("a b/c:d*e") == "a_b_c_d_e"

    def test_keeps_cjk_and_truncates(self) -> None:
        assert sanitize_filename("中文标题 ok") == "中文标题_ok"
        assert len(sanitize_filename("x" * 200)) == 80

    def test_empty_becomes_untitled(self) -> None:
        assert sanitize_filename("") == "untitled"
        assert sanitize_filename("...") == "untitled"


class TestUrlExtension:
    def test_strips_query_and_fragment(self) -> None:
        assert url_extension("https://x/y/paper.PDF?v=2#frag") == ".pdf"

    def test_returns_empty_when_unknown(self) -> None:
        assert url_extension("https://x/y/file") == ""
        assert url_extension("") == ""
        assert url_extension("https://x/y/archive.tar.gz") == ".gz"
        assert url_extension("https://x/y/weird.verylongextension") == ""


class TestEnsureExtension:
    def test_appends_when_missing(self) -> None:
        assert ensure_extension("main", ".py") == "main.py"

    def test_does_not_double_the_suffix(self) -> None:
        assert ensure_extension("main.py", ".py") == "main.py"
        assert ensure_extension("main.PY", ".py") == "main.PY"

    def test_empty_extension_is_a_noop(self) -> None:
        assert ensure_extension("main", "") == "main"


class TestBuildFilename:
    def test_sequence_is_zero_padded(self) -> None:
        assert build_filename(1, "My Paper", ".pdf") == "01_My_Paper.pdf"

    def test_no_doubled_extension(self) -> None:
        assert build_filename(12, "main.py", ".py") == "12_main.py"


class TestUniquePath:
    def test_appends_incrementing_counter(self, tmp_path: Path) -> None:
        first = unique_path(tmp_path, "a.txt")
        assert first.name == "a.txt"
        first.write_text("x", encoding="utf-8")

        second = unique_path(tmp_path, "a.txt")
        assert second.name == "a_2.txt"
        second.write_text("x", encoding="utf-8")

        assert unique_path(tmp_path, "a.txt").name == "a_3.txt"


class TestHashing:
    def test_sha_helpers_agree(self, tmp_path: Path) -> None:
        data = b"hello world"
        path = tmp_path / "f.bin"
        path.write_bytes(data)

        assert sha256_of(data) == hashlib.sha256(data).hexdigest()
        assert sha256_of_file(path) == sha256_of(data)

    def test_sha256_of_file_streams_large_files(self, tmp_path: Path) -> None:
        path = tmp_path / "big.bin"
        path.write_bytes(b"a" * (1024 * 1024 + 7))

        assert sha256_of_file(path) == hashlib.sha256(b"a" * (1024 * 1024 + 7)).hexdigest()
