# CMN-C1-349 — Unit Tests: OrchestrationMainNode (main slot composite)
#
# OrchestrationMainNode composes the 4 reasoning steps (TopicClassify →
# HybridRetrieve → FrameworkVersionInject → ResponseFormat) into the single main
# slot. These tests verify the composite contract: it threads state through the
# sub-nodes, returns accumulated deltas, and always reports SUCCESS so routing
# advances to post_process.

import inspect
import json

from framework.schemas.agent_status import AgentStatus

from src.nodes.main_node import MainNode, OrchestrationMainNode
from src.services.service import InMemoryKBBackend, StubLLMClient


def _backend():
    b = InMemoryKBBackend()
    b.add([{"doc_id": "d1", "title": "Sequential orchestration",
            "text": "agents run in fixed order passing output forward",
            "score": 0.0, "source": "ms-learn", "framework_version": "1.0.0",
            "topic": "multi-agent-patterns"}])
    return b


def _node(backend=None):
    return OrchestrationMainNode(retrieval_backend=backend or _backend(), llm_client=StubLLMClient())


def test_alias_is_orchestration_main_node():
    assert MainNode is OrchestrationMainNode


def test_composite_produces_full_deltas():
    out = _node().execute({"validated_question": "multi-agent orchestration patterns"})
    assert out["topic"] == "multi-agent-patterns"
    assert out["retrieval_hit_count"] >= 1
    assert "framework_version_context" in out
    assert out["answer"]
    assert out["status"] == AgentStatus.SUCCESS.value


def test_always_success_even_on_subnode_error():
    # A failing backend → HybridRetrieve sets error_code, but the composite still
    # reports SUCCESS so the framework routes on to post_process (auditable).
    # (retrieval_backend=None no longer errors — it binds the illustrative seed
    # KB so a bare Graph() is a working deployment form.)
    class _Boom:
        def search(self, *a, **k):
            raise RuntimeError("backend down")

    out = OrchestrationMainNode(retrieval_backend=_Boom(), llm_client=StubLLMClient()).execute(
        {"validated_question": "multi-agent patterns"}
    )
    assert out["status"] == AgentStatus.SUCCESS.value
    assert out["error_code"] == "RETRIEVAL_FAILED"


def test_out_of_scope_yields_refusal():
    out = _node().execute({"validated_question": "what is the weather in tokyo"})
    assert out["topic"] == "out-of-scope"
    assert out["retrieval_hit_count"] == 0
    assert "outside the Microsoft agent-framework" in out["answer"]


def test_citations_match_retrieved_docs():
    out = _node().execute({"validated_question": "sequential orchestration order"})
    cites = json.loads(out["citations"])
    docs = json.loads(out["retrieved_docs"])
    assert {c["doc_id"] for c in cites} == {d["doc_id"] for d in docs}


def test_execute_signature_is_node_contract():
    sig = inspect.signature(OrchestrationMainNode.execute)
    params = list(sig.parameters.keys())
    assert params[1] == "state"
    assert "_invoke_impl" not in OrchestrationMainNode.__dict__
