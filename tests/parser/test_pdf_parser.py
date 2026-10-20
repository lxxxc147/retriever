from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from retriever.parser.pdf import parse_pdf


@pytest.fixture
def sample_normal_pdf(tmp_path: Path) -> Path:
    """构造标准测试 PDF：包含标题、正文、参考文献完整结构"""
    pdf_path = tmp_path / "test_paper.pdf"
    doc = pymupdf.open()

    # 第一页：标题 + 摘要 + 引言
    page1 = doc.new_page()
    page1.insert_text((100, 100), "Research on Deep Learning", fontsize=20)
    page1.insert_text((100, 150), "Abstract", fontsize=14)
    page1.insert_text(
        (100, 180),
        "This paper studies the application and development of deep learning in various fields.",
        fontsize=12,
    )
    page1.insert_text((100, 220), "1 Introduction", fontsize=14)
    page1.insert_text(
        (100, 250),
        "With the rapid development of AI, deep learning has been widely used in CV and NLP.",
        fontsize=12,
    )

    # 第二页：相关工作 + 参考文献
    page2 = doc.new_page()
    page2.insert_text((100, 100), "2 Related Work", fontsize=14)
    page2.insert_text(
        (100, 130),
        "Existing research focuses on model optimization and application implementation.",
        fontsize=12,
    )
    page2.insert_text((100, 300), "References", fontsize=16)
    page2.insert_text(
        (100, 330),
        "[1] Zhang S. Introduction to Deep Learning. Computer Press, 2022.",
        fontsize=10,
    )
    page2.insert_text(
        (100, 360),
        "[2] Li M. Deep learning review. Nature, 2021.",
        fontsize=10,
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_scan_pdf(tmp_path: Path) -> Path:
    """构造模拟扫描版 PDF：文本极少，模拟无文本层扫描件"""
    pdf_path = tmp_path / "scan_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((100, 100), " ", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_parse_normal_pdf_extract_all_fields(sample_normal_pdf):
    """测试标准 PDF 解析：正确提取标题、正文、参考文献分节"""
    result = parse_pdf(sample_normal_pdf)

    # 元数据校验
    assert result["meta"]["title"] == "Research on Deep Learning"
    assert result["meta"]["page_count"] == 2
    assert result["meta"]["parse_quality"] == "normal"
    assert result["meta"]["word_count"] > 0
    assert result["meta"]["sha256"] == hashlib.sha256(sample_normal_pdf.read_bytes()).hexdigest()
    assert result["meta"]["file_name"] == "test_paper.pdf"

    # 正文校验
    assert "deep learning" in result["content"]["body"]
    assert "Introduction" in result["content"]["body"]
    assert "Related Work" in result["content"]["body"]
    # 参考文献不在正文中
    assert "References" not in result["content"]["body"]

    # 参考文献校验
    assert len(result["content"]["references"]) >= 2
    assert "Zhang S" in result["content"]["references"][0]
    assert "Nature" in result["content"]["references"][1]


def test_parse_scan_pdf_mark_quality(sample_scan_pdf):
    """测试扫描版 PDF 识别：标记 scan 质量，不抛出异常"""
    result = parse_pdf(sample_scan_pdf)

    assert result["meta"]["parse_quality"] == "scan"
    assert result["meta"]["page_count"] == 1


def test_parse_pdf_sha256_traceable(sample_normal_pdf):
    """测试文件哈希校验正确，支持溯源"""
    result = parse_pdf(sample_normal_pdf)
    expected_sha = hashlib.sha256(sample_normal_pdf.read_bytes()).hexdigest()

    assert result["meta"]["sha256"] == expected_sha


def test_parse_nonexistent_pdf_raises():
    """测试不存在的文件抛出明确异常"""
    with pytest.raises(FileNotFoundError):
        parse_pdf(Path("nonexistent_file.pdf"))
