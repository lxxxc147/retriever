"""M8 本地知识库问答 — parsed 文档驱动的 retrieved-QA（Phase 4，PRD §2.9）。

回答"已经下载并解析过的文档里说了什么"这类问题：

1. :func:`load_docs` 扫描 ``parsed/<日期>/<task_id>/*.json``（可限定 task），
   把 M6 的结构化结果 flatten 成 :class:`QADoc`；
2. :func:`retrieve` 用中英混合的词汇重叠打分（拉丁词 + 中文 bigram，带
   长度归一）选出 top-k 相关文档；
3. :func:`answer_question` 在相关文档上作答：注入了 ``ask``（LLM 调用）
   时生成式回答，否则退化为抽取式——按句重叠度挑最相关的原句，绝不
   因为模型不可用而答不了。

本模块零网络依赖：LLM 通过 callable 注入，测试传假函数即可。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

QA_PROMPT = (
    "你是文献问答助手。请只根据下面的资料回答问题；"
    "资料不足时直说不知道，不要编造。用中文回答，控制在 200 字以内。\n\n"
    "资料：\n{context}\n\n问题：{question}"
)

_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+\-]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*|\n+")


@dataclass(frozen=True)
class QADoc:
    """One parsed document flattened for retrieval."""

    task_id: str  # <parsed_root>/<date>/<task_id>/<file>.json → parent dir name
    path: Path  # path of the parsed JSON this doc came from
    title: str
    text: str  # body text (references excluded by M6)


@dataclass(frozen=True)
class Citation:
    """One supporting document and how well it matched."""

    doc: QADoc
    score: float


@dataclass(frozen=True)
class Answer:
    """A QA result: the answer text plus its supporting documents."""

    text: str
    citations: list[Citation]


# ---------------------------------------------------------------------- loading
def load_docs(parsed_root: Path | str, task_id: str | None = None) -> list[QADoc]:
    """Load parsed documents from ``parsed_root`` (M6 output layout).

    Missing directories and malformed JSON files are skipped with a warning —
    the QA session works with whatever parsed artefacts exist.

    Args:
        parsed_root: The ``parsed/`` root (``<root>/<date>/<task_id>/*.json``).
        task_id: Restrict to one task; ``None`` loads every parsed task.

    Returns:
        Documents carrying a non-empty body, sorted by path for determinism.
    """
    root = Path(parsed_root)
    if not root.exists():
        return []
    pattern = f"*/{task_id}/*.json" if task_id else "*/*/*.json"
    docs: list[QADoc] = []
    for path in sorted(root.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("skipping unreadable parsed file {}: {}", path, exc)
            continue
        meta = data.get("meta") or {}
        content = data.get("content") or {}
        body = str(content.get("body") or "").strip()
        if not body:
            continue
        title = str(meta.get("title") or path.stem)
        docs.append(QADoc(task_id=path.parent.name, path=path, title=title, text=body))
    logger.debug("loaded {} parsed doc(s) from {}", len(docs), root)
    return docs


# ------------------------------------------------------------------- retrieval
def _terms(text: str) -> list[str]:
    """Tokenize for overlap scoring: latin words as-is, CJK runs as bigrams."""
    lowered = text.lower()
    terms: list[str] = []
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    terms.extend(_WORD_RE.findall(lowered))
    return terms


def _score(question_terms: list[str], doc_terms: list[str]) -> float:
    """Sublinear tf with sqrt-length normalization; ``0`` when nothing overlaps."""
    if not question_terms or not doc_terms:
        return 0.0
    doc_tf: dict[str, int] = {}
    for term in doc_terms:
        doc_tf[term] = doc_tf.get(term, 0) + 1
    score = 0.0
    for term in set(question_terms):
        tf = doc_tf.get(term, 0)
        if tf:
            score += 1.0 + math.log(tf)
    return score / math.sqrt(len(doc_terms))


def retrieve(question: str, docs: list[QADoc], top_k: int = 3) -> list[Citation]:
    """Rank ``docs`` by lexical overlap with ``question`` (best first).

    Args:
        question: The user's question.
        docs: Candidate documents (already loaded).
        top_k: Maximum number of citations to keep.

    Returns:
        Citations with a positive score, sorted descending; empty when no
        document shares any term with the question.
    """
    question_terms = _terms(question)
    scored = [(_score(question_terms, _terms(f"{doc.title}\n{doc.text}")), doc) for doc in docs]
    matched = [(score, doc) for score, doc in scored if score > 0]
    matched.sort(key=lambda item: item[0], reverse=True)  # stable: keeps load order on ties
    return [Citation(doc=doc, score=score) for score, doc in matched[:top_k]]


# ---------------------------------------------------------------------- answer
def _build_context(citations: list[Citation], max_chars: int) -> str:
    """Assemble the LLM context: each citation's title plus a bounded excerpt."""
    parts: list[str] = []
    budget = max_chars
    for citation in citations:
        head = citation.doc.text[: max(200, budget // max(1, len(citations)))]
        chunk = f"《{citation.doc.title}》\n{head}"
        parts.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    return "\n\n---\n\n".join(parts)


def _extractive(question: str, citations: list[Citation], max_sentences: int = 2) -> str:
    """Pick the sentences covering the most question terms, in document order.

    Ranking favours sentences that match several *distinct* question terms
    (coverage of the question, not raw term frequency), so caption fragments
    like "Attention Attention" cannot beat real explanatory sentences.
    """
    question_terms = set(_terms(question))
    if not question_terms:
        return ""
    min_hits = 1 if len(question_terms) < 2 else 2
    ranked: list[tuple[int, float, int, int, str]] = []  # (hits, coverage, cit, pos, sent)
    for cit_index, citation in enumerate(citations):
        for position, raw in enumerate(_SENT_SPLIT_RE.split(citation.doc.text)):
            sentence = " ".join(raw.split())
            if len(sentence) < 20:
                continue
            matched = {term for term in _terms(sentence) if term in question_terms}
            if len(matched) < min_hits:  # single-term hits only for single-term questions
                continue
            coverage = len(matched) / len(question_terms)
            ranked.append((len(matched), coverage, cit_index, position, sentence))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    picked = sorted(ranked[:max_sentences], key=lambda item: (item[2], item[3]))
    return " ".join(sentence for *_, sentence in picked)


def answer_question(
    question: str,
    docs: list[QADoc],
    *,
    ask: Callable[[str], str] | None = None,
    top_k: int = 3,
    max_context_chars: int = 6000,
) -> Answer:
    """Answer ``question`` over ``docs``, generating when possible else extractive.

    Args:
        question: The user's question.
        docs: The parsed document corpus.
        ask: Optional LLM callable receiving one prompt string and returning the
            answer text. ``None`` (or a callable that fails) uses the extractive
            fallback so a dead model never blocks QA.
        top_k: How many documents to cite at most.
        max_context_chars: Context budget handed to ``ask``.

    Returns:
        The answer with its citations; ``text`` is empty when no document
        matched the question at all.
    """
    citations = retrieve(question, docs, top_k=top_k)
    if not citations:
        return Answer(text="", citations=[])
    if ask is not None:
        prompt = QA_PROMPT.format(
            context=_build_context(citations, max_context_chars), question=question
        )
        try:
            text = ask(prompt).strip()
        except Exception as exc:  # noqa: BLE001 — 抽取式兜底是安全网
            logger.warning("QA LLM call failed, falling back to extractive answer: {}", exc)
            text = ""
        if text:
            return Answer(text=text, citations=citations)
    return Answer(text=_extractive(question, citations), citations=citations)
