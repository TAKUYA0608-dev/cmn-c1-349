# CMN-C1-349 — Unit Tests: FrameworkVersionInjectNode

import json

from framework.schemas.agent_status import AgentStatus

from src.nodes.framework_version_inject_node import FrameworkVersionInjectNode, _latest_version
from src.services.service import FRAMEWORK


def _run(docs):
    return FrameworkVersionInjectNode().execute({"retrieved_docs": json.dumps(docs)})


def test_picks_most_cited_version():
    out = _run([
        {"doc_id": "d1", "framework_version": "1.0.0"},
        {"doc_id": "d2", "framework_version": "1.0.0"},
        {"doc_id": "d3", "framework_version": "0.9.0"},
    ])
    ctx = json.loads(out["framework_version_context"])
    assert ctx["framework"] == FRAMEWORK
    assert ctx["latest_version"] == "1.0.0"
    assert ctx["channel"] == "official"
    assert out["status"] == AgentStatus.SUCCESS.value


def test_tie_resolves_to_greatest():
    assert _latest_version(["1.0.0", "1.1.0"]) == "1.1.0"


def test_empty_docs_marks_unavailable():
    out = _run([])
    ctx = json.loads(out["framework_version_context"])
    assert ctx["latest_version"] == ""
    assert ctx["channel"] == "unknown"


def test_docs_without_version_marks_unavailable():
    out = _run([{"doc_id": "d1"}, {"doc_id": "d2"}])
    ctx = json.loads(out["framework_version_context"])
    assert ctx["latest_version"] == ""


def test_handles_malformed_retrieved_docs():
    out = FrameworkVersionInjectNode().execute({"retrieved_docs": "not-json"})
    ctx = json.loads(out["framework_version_context"])
    assert ctx["latest_version"] == ""


def test_self_skips_on_error():
    assert FrameworkVersionInjectNode().execute({"error_code": "X"}) == {}
