# CMN-C1-349 — Unit Tests: HybridRetrieveNode

import json

from framework.schemas.agent_status import AgentStatus

from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.services.service import InMemoryKBBackend


def _backend():
    b = InMemoryKBBackend()
    b.add([
        {"doc_id": "d1", "title": "Sequential orchestration", "text": "agents run in fixed order",
         "score": 0.0, "source": "ms-learn", "framework_version": "1.0.0", "topic": "multi-agent-patterns"},
        {"doc_id": "d2", "title": "Handoff pattern", "text": "agent handoff transfers control",
         "score": 0.0, "source": "ms-learn", "framework_version": "1.0.0", "topic": "multi-agent-patterns"},
    ])
    return b


def _run(state, backend=None):
    return HybridRetrieveNode(retrieval_backend=backend).execute(state)


def test_retrieves_and_builds_citations():
    out = _run({"validated_question": "sequential orchestration order", "topic": "multi-agent-patterns"}, _backend())
    docs = json.loads(out["retrieved_docs"])
    cites = json.loads(out["citations"])
    assert out["retrieval_hit_count"] == len(docs) >= 1
    assert all(c["marker"] == f"[{c['doc_id']}]" for c in cites)
    assert {c["doc_id"] for c in cites} == {d["doc_id"] for d in docs}
    assert out["status"] == AgentStatus.SUCCESS.value


def test_out_of_scope_skips_retrieval():
    out = _run({"validated_question": "weather in tokyo", "topic": "out-of-scope"}, _backend())
    assert out["retrieval_hit_count"] == 0
    assert json.loads(out["retrieved_docs"]) == []
    assert json.loads(out["citations"]) == []


def test_missing_backend_binds_seed_kb():
    # Contract change (default retrieval bind): no injected backend now means the
    # illustrative seed KB, not RETRIEVAL_BACKEND_MISSING — a bare Graph() (the
    # Marketplace runner / server.py construction form) must be a working form.
    out = _run({"validated_question": "multi-agent orchestration patterns", "topic": "multi-agent-patterns"}, None)
    assert "error_code" not in out
    assert out["retrieval_hit_count"] >= 1
    assert out["status"] == AgentStatus.SUCCESS.value


def test_backend_exception_degrades_gracefully():
    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("backend down")
    out = _run({"validated_question": "x", "topic": "multi-agent-patterns"}, Boom())
    assert out["error_code"] == "RETRIEVAL_FAILED"


def test_stopwords_do_not_match():
    # a question made of pure stopwords must not match orchestration docs
    out = _run({"validated_question": "what is the in on for", "topic": "multi-agent-patterns"}, _backend())
    assert out["retrieval_hit_count"] == 0


def test_self_skips_on_error():
    assert _run({"error_code": "X"}, _backend()) == {}
