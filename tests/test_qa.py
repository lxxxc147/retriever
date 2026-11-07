"""Tests for M8 local QA — loading, retrieval scoring, answering (PRD §2.9).

No network anywhere: the LLM is a fake callable and the corpus lives in
``tmp_path`` fixtures laid out like ``parsed/<date>/<task_id>/*.json``.
"""

import json
from pathlib import Path

import pytest

from retriever.qa import Answer, QADoc, _extractive, _terms, answer_question, load_docs, retrieve


def _write_parsed(
    root: Path,
    date: str,
    task_id: str,
    name: str,
    *,
    title: str,
    body: str,
) -> Path:
    path = root / date / task_id / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": {"title": title}, "content": {"body": body, "references": []}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, list[QADoc]]:
    """Two parsed tasks: one about transformers, one about rust web frameworks."""
    root = tmp_path / "parsed"
    _write_parsed(
        root,
        "2026-11-01",
        "taskaaa11111",
        "attention",
        title="Attention Is All You Need",
        body=(
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks. We propose the Transformer, a model architecture "
            "eschewing recurrence and relying entirely on an attention mechanism. "
            "Experiments on machine translation show the Transformer trains faster and "
            "achieves better quality than recurrent models."
        ),
    )
    _write_parsed(
        root,
        "2026-11-02",
        "taskbbb22222",
        "axum",
        title="Axum web framework notes",
        body=(
            "Axum is a web application framework that focuses on ergonomics and modularity. "
            "It uses tokio and tower for the async runtime and middleware layers."
        ),
    )
    return root, load_docs(root)


# ---------------------------------------------------------------------- loading
def test_load_docs_flattens_parsed_json(corpus: tuple[Path, list[QADoc]]) -> None:
    """Parsed JSON files become QADocs with title and body text."""
    _root, docs = corpus
    assert len(docs) == 2
    titles = {doc.title for doc in docs}
    assert "Attention Is All You Need" in titles
    assert all(doc.task_id and doc.text for doc in docs)


def test_load_docs_filters_by_task(corpus: tuple[Path, list[QADoc]]) -> None:
    """``task_id`` restricts the corpus to one parsed task."""
    root, _docs = corpus
    docs = load_docs(root, task_id="taskbbb22222")
    assert len(docs) == 1
    assert docs[0].title == "Axum web framework notes"


def test_load_docs_skips_malformed_and_missing_root(tmp_path: Path) -> None:
    """Bad JSON and a missing parsed/ root never break QA."""
    root = tmp_path / "parsed"
    _write_parsed(root, "2026-11-01", "t1", "good", title="good", body="some body text")
    bad = root / "2026-11-01" / "t1" / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    empty = root / "2026-11-01" / "t1" / "empty.json"
    empty.write_text(
        json.dumps({"meta": {"title": "e"}, "content": {"body": "", "references": []}}),
        encoding="utf-8",
    )
    docs = load_docs(root)
    assert len(docs) == 1
    assert load_docs(tmp_path / "nonexistent") == []


# ---------------------------------------------------------------------- _terms
def test_terms_mixes_latin_words_and_cjk_bigrams() -> None:
    """English words stay whole; Chinese text becomes overlapping bigrams."""
    assert "transformer" in _terms("Transformer 注意力机制")
    assert "注意" in _terms("注意力机制")
    assert "力机" in _terms("注意力机制")


# -------------------------------------------------------------------- retrieval
def test_retrieve_ranks_relevant_doc_first(corpus: tuple[Path, list[QADoc]]) -> None:
    """A question about attention must surface the transformer paper, not axum."""
    _root, docs = corpus
    citations = retrieve("attention mechanism 是什么", docs)
    assert citations, "expected at least one match"
    assert citations[0].doc.title == "Attention Is All You Need"
    assert all(citation.score > 0 for citation in citations)


def test_retrieve_returns_empty_when_nothing_matches(
    corpus: tuple[Path, list[QADoc]]
) -> None:
    """Terms absent from the whole corpus yield no citations."""
    _root, docs = corpus
    assert retrieve("quantum chromodynamics 色动力学", docs) == []


# ------------------------------------------------------------------------ answer
def test_answer_question_uses_llm_when_asked(corpus: tuple[Path, list[QADoc]]) -> None:
    """A working ``ask`` produces the model answer plus citations."""
    _root, docs = corpus
    prompts: list[str] = []

    def fake_ask(prompt: str) -> str:
        prompts.append(prompt)
        return "Transformer 完全依赖注意力机制。"

    answer = answer_question("attention 是什么", docs, ask=fake_ask)
    assert answer.text == "Transformer 完全依赖注意力机制。"
    assert len(answer.citations) == 1
    assert answer.citations[0].doc.title == "Attention Is All You Need"
    assert prompts and "attention" in prompts[0]


def test_answer_question_falls_back_when_llm_fails(
    corpus: tuple[Path, list[QADoc]]
) -> None:
    """A dead model degrades to extractive answers instead of failing."""

    def broken_ask(_prompt: str) -> str:
        raise RuntimeError("endpoint down")

    answer = answer_question("attention mechanism", corpus[1], ask=broken_ask)
    assert "attention" in answer.text.lower()
    assert answer.citations


def test_answer_question_extractive_without_llm(corpus: tuple[Path, list[QADoc]]) -> None:
    """Without ``ask`` the answer is the most relevant original sentence."""
    _root, docs = corpus
    answer = answer_question("attention mechanism", docs)
    assert "attention" in answer.text.lower()
    assert "Transformer" in answer.text
    assert answer.citations[0].doc.title == "Attention Is All You Need"


def test_answer_question_no_match_returns_empty(corpus: tuple[Path, list[QADoc]]) -> None:
    """No lexical overlap → empty answer, empty citations, exit code handled by CLI."""
    _root, docs = corpus
    answer = answer_question("zzzz量子引力zzzz", docs)
    assert answer == Answer(text="", citations=[])


# ------------------------------------------------------------------- extractive
def test_extractive_prefers_overlapping_sentences() -> None:
    """Sentences sharing question terms win; output keeps document order."""
    doc = QADoc(
        task_id="t",
        path=Path("x.json"),
        title="t",
        text=(
            "Filler sentence one with nothing relevant. "
            "The attention mechanism maps queries keys and values. "
            "Another unrelated filler sentence here."
        ),
    )
    from retriever.qa import Citation

    text = _extractive("attention mechanism", [Citation(doc=doc, score=1.0)])
    assert "attention mechanism" in text
    assert "Filler" not in text
