from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pymupdf
from loguru import logger


def parse_pdf(pdf_path: Path) -> dict[str, Any]:
    """解析 PDF 为结构化文本，支持标题、正文、参考文献分节提取，扫描版自动识别标记。

    输出结果可直接序列化为 JSON，包含完整元数据与溯源哈希，扫描版仅标记质量不抛出异常。

    Args:
        pdf_path: 待解析 PDF 文件路径

    Returns:
        结构化解析结果字典
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 计算原始文件哈希用于溯源
    raw_bytes = pdf_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.error("PDF 打开失败: {} - {}", pdf_path, exc)
        raise RuntimeError(f"PDF 文件损坏或格式不支持: {pdf_path}") from exc

    page_count = len(doc)
    total_text_len = 0
    all_pages_text = []
    title_candidates: list[tuple[str, float]] = []

    # 逐页提取文本与标题候选
    for page in doc:
        text = page.get_text()
        all_pages_text.append(text)
        total_text_len += len(text.strip())

        # 提取大字号文本作为标题候选
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text_span = span["text"].strip()
                    if text_span and span["size"] >= 16:
                        title_candidates.append((text_span, span["size"]))

    full_text = "\n".join(all_pages_text)
    word_count = len(full_text.split())

    # 扫描版识别：平均每页文本少于 50 字符判定为扫描版
    avg_text_per_page = total_text_len / max(page_count, 1)
    parse_quality = "normal" if avg_text_per_page > 50 else "scan"

    # 提取标题：优先元数据 > 最大字号文本 > 文件名
    title = doc.metadata.get("title", "").strip()
    if not title and title_candidates:
        title_candidates.sort(key=lambda x: x[1], reverse=True)
        title = title_candidates[0][0]
    if not title:
        title = pdf_path.stem

    # 查找参考文献章节起始位置
    ref_start = _find_references_start(full_text)

    # 拆分正文与参考文献
    if ref_start != -1:
        body_text = full_text[:ref_start].strip()
        ref_section = full_text[ref_start:]
        references = _extract_references(ref_section)
    else:
        body_text = full_text.strip()
        references = []

    result = {
        "meta": {
            "title": title,
            "page_count": page_count,
            "word_count": word_count,
            "sha256": sha256,
            "parse_quality": parse_quality,
            "file_name": pdf_path.name,
        },
        "content": {
            "body": body_text,
            "references": references,
        },
    }

    doc.close()
    logger.debug(
        "PDF 解析完成: {} 页, 质量: {}, 字数: {}",
        page_count,
        parse_quality,
        word_count,
    )
    return result


def _find_references_start(full_text: str) -> int:
    """查找参考文献章节标题的起始位置。"""
    ref_header_patterns = [
        r"(?m)^参考文献\s*$",
        r"(?m)^References\s*$",
        r"(?m)^REFERENCES\s*$",
        r"(?m)^Bibliography\s*$",
        r"(?m)^BIBLIOGRAPHY\s*$",
    ]

    for pattern in ref_header_patterns:
        match = re.search(pattern, full_text)
        if match:
            return match.start()
    return -1


def _extract_references(ref_section: str) -> list[str]:
    """从参考文献章节中拆分条目列表。

    支持 [数字]、数字编号等多种条目格式。
    """
    entry_pattern = r"(?m)^\s*\[?\d+\]?\.?\s+"
    entries = re.split(entry_pattern, ref_section)
    entries = [e.strip() for e in entries if e.strip() and len(e.strip()) > 20]

    # 移除章节标题本身
    if entries and any(
        h in entries[0] for h in ["参考文献", "References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY"]
    ):
        entries.pop(0)

    return entries
