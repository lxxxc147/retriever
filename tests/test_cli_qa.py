"""CLI tests for ``retrieve --ask`` (M8 local QA, Phase 4).

The parsed corpus lives in ``tmp_path``; the LLM is faked via monkeypatching
``cli.LLMClient`` — no network access anywhere.
"""

import json
from pathlib import Path

import pytest

from retriever import cli

_BODY = (
    "We propose the Transformer, a model architecture relying entirely on an "
    "attention mechanism that maps queries keys and values. "
    "It trains faster than recurrent models on machine translation."
)


@pytest.fixture
def parsed_dir(tmp_path: Path) -> Path:
    """One parsed task whose body mentions attention and transformers."""
    root = tmp_path / "parsed"
    doc = root / "2026-11-01" / "taskqa123456" / "attention.json"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        json.dumps(
            {
                "meta": {"title": "Attention Is All You Need"},
                "content": {"body": _BODY, "references": []},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_ask_answers_extractively_without_llm(
    parsed_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    """No API key → the answer is extracted from the most relevant sentences."""
    code = cli.main(["--ask", "attention mechanism", "--parsed-dir", str(parsed_dir)])

    assert code == 0
    out = capsys.readouterr().out
    assert "抽取自原文" in out
    assert "attention" in out
    assert "Sources:" in out
    assert "Attention Is All You Need" in out
    assert "task taskqa123456" in out


def test_ask_uses_the_llm_when_configured(
    parsed_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """With a key configured the LLM answers and the model name is shown."""
    prompts: list[str] = []

    class _FakeClient:
        configured = True
        model = "fake-qa-model"

        def chat_sync(self, messages: list[dict], **kwargs: object) -> str:
            prompts.extend(m["content"] for m in messages)
            return "注意力机制直接建模序列内任意两位置的关联。"

    monkeypatch.setattr(cli, "LLMClient", _FakeClient)

    code = cli.main(["--ask", "attention 是什么", "--parsed-dir", str(parsed_dir)])

    assert code == 0
    out = capsys.readouterr().out
    assert "fake-qa-model" in out
    assert "注意力机制直接建模" in out
    assert len(prompts) == 1 and "attention" in prompts[0]


def test_ask_task_filter_selects_the_corpus_subset(
    parsed_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    """``--task`` restricts QA to one task; a wrong id finds no documents."""
    assert cli.main(
        ["--ask", "attention", "--task", "taskqa123456", "--parsed-dir", str(parsed_dir)]
    ) == 0
    capsys.readouterr()

    assert cli.main(
        ["--ask", "attention", "--task", "nosuchtask00", "--parsed-dir", str(parsed_dir)]
    ) == 1
    out = capsys.readouterr().out
    assert "没有找到" in out


def test_ask_without_any_parsed_docs_fails_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """An empty corpus exits 1 and tells the user how to produce documents."""
    code = cli.main(["--ask", "anything", "--parsed-dir", str(tmp_path / "empty")])

    assert code == 1
    out = capsys.readouterr().out
    assert "没有找到" in out
    assert "--download --parse" in out
