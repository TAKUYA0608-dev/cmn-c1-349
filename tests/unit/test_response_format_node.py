# CMN-C1-349 — Unit Tests: ResponseFormatNode

import json
import re

from framework.schemas.agent_status import AgentStatus

from src.nodes.response_format_node import ResponseFormatNode, _cite_sentences

_CITATION = re.compile(r"\[[A-Za-z0-9][\w\-:.]*\]")


def _run(docs):
    return ResponseFormatNode().execute({"retrieved_docs": json.dumps(docs)})


def test_every_sentence_is_cited():
    out = _run([
        {"doc_id": "d1", "title": "Sequential", "text": "Agents run in order. Output passes forward."},
        {"doc_id": "d2", "title": "Handoff", "text": "Control transfers to another agent."},
    ])
    answer = out["answer"]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    assert sentences, "expected a non-empty answer"
    for s in sentences:
        assert _CITATION.search(s), f"uncited sentence: {s!r}"
    assert out["status"] == AgentStatus.SUCCESS.value


def test_disclaimer_and_jp_note_separate_from_answer():
    out = _run([{"doc_id": "d1", "title": "T", "text": "Agents run in order."}])
    assert out["disclaimer"]
    assert "Japan" in out["jp_enterprise_note"]
    # the gated answer body does NOT contain the (uncited) disclaimer text
    assert out["disclaimer"] not in out["answer"]


def test_empty_docs_returns_refusal():
    out = _run([])
    assert "outside the Microsoft agent-framework" in out["answer"]
    assert out["jp_enterprise_note"] == ""
    assert out["status"] == AgentStatus.SUCCESS.value


def test_cite_sentences_helper_marks_each():
    sents = _cite_sentences("First claim. Second claim.", "[d1]")
    assert len(sents) == 2
    assert all(s.endswith("[d1].") for s in sents)


def test_self_skips_on_error():
    assert ResponseFormatNode().execute({"error_code": "X"}) == {}
