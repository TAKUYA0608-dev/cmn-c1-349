# CMN-C1-349 — Unit Tests: SecurityGateOutputNode (S-3 citation gate)

import json

from framework.schemas.agent_status import AgentStatus

from src.nodes.security_gate_output_node import SecurityGateOutputNode, _uncited_claims
from src.services.service import StubLLMClient


def _run(state, llm=None):
    return SecurityGateOutputNode(llm_client=llm).execute(state)


def test_fully_cited_answer_passes():
    state = {"answer": "Agents run in fixed order [d1].\nControl transfers forward [d2].", "retrieval_hit_count": 2}
    out = _run(state)
    assert out["citation_status"] == "passed"
    assert out["audit_logged"] is True
    assert json.loads(out["uncited_claims"]) == []
    assert out["status"] == AgentStatus.SUCCESS.value


def test_zero_hit_refusal_passes_through():
    out = _run({"answer": "This is outside the knowledge base and cannot be grounded.", "retrieval_hit_count": 0})
    assert out["citation_status"] == "passed"
    assert out["audit_logged"] is True


def test_empty_answer_passes_through():
    out = _run({"answer": "", "retrieval_hit_count": 2})
    assert out["citation_status"] == "passed"


def test_upstream_error_passes_through():
    out = _run({"answer": "x", "retrieval_hit_count": 2, "error_code": "BOOM"})
    assert out["citation_status"] == "passed"


def test_uncited_claim_regenerated_when_llm_fixes():
    # LLM returns a fully-cited rewrite → status regenerated.
    fixer = StubLLMClient(canned="The framework supports group chat orchestration [d1].")
    state = {"answer": "The framework supports group chat orchestration with no citation here at all.",
             "retrieval_hit_count": 1}
    out = _run(state, fixer)
    assert out["citation_status"] == "regenerated"
    assert "[d1]" in out["answer"]


def test_uncited_claim_rejected_when_llm_fails():
    # LLM returns another uncited answer → status rejected, safe refusal substituted.
    bad = StubLLMClient(canned="Still no citation on this substantive regulatory claim here.")
    state = {"answer": "An uncited substantive claim about orchestration patterns goes here.",
             "retrieval_hit_count": 1}
    out = _run(state, bad)
    assert out["citation_status"] == "rejected"
    assert "withholding" in out["answer"]
    assert out["audit_logged"] is True


def test_uncited_claims_helper():
    uncited = _uncited_claims("Agents run in fixed order with no citation present here at all. Cited claim [d1].")
    assert len(uncited) == 1
    assert "[d1]" not in uncited[0]


def test_short_non_substantive_sentence_not_flagged():
    # a short fragment (<5 words, <16 chars) is not a substantive claim
    assert _uncited_claims("OK. Yes.") == []
