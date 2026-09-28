# CMN-C1-349 — Regression: a bare Graph() (Marketplace runner / server.py
# construction form) must work with the illustrative seed KB instead of
# degrading every run to RETRIEVAL_BACKEND_MISSING.

from framework.schemas.invocation_context import InvocationContext, TrustLevel

from src.graph.graph import Graph
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.services.service import InMemoryKBBackend


def _ctx():
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, session_id="bare")


def _bare_agent():
    agent = Graph()  # no backend, no LLM — exactly what the Marketplace runner constructs
    agent.compile()
    return agent


def test_bare_graph_answers_from_seed_kb():
    out = _bare_agent().invoke("What multi-agent orchestration patterns are available?", ctx=_ctx())
    assert out["status"] == "success"
    assert out.get("error_code") != "RETRIEVAL_BACKEND_MISSING"
    assert out.get("retrieval_hit_count", 0) >= 1
    assert out.get("output"), "bare construction must still publish a non-empty output"


def test_bare_graph_survives_degenerate_input():
    out = _bare_agent().invoke(".", ctx=_ctx())
    assert out["status"] == "success"
    assert out.get("error_code") != "RETRIEVAL_BACKEND_MISSING"
    assert out.get("output"), "a degraded run must still publish a non-empty output"


def test_injected_backend_still_wins():
    """The DI path is unchanged: an injected backend is used, not the seed."""
    backend = InMemoryKBBackend()
    backend.add(
        [
            {
                "doc_id": "custom-1",
                "title": "custom doc",
                "text": "customtoken only lives here",
                "score": 0.0,
                "source": "test",
                "framework_version": "1.0.0",
                "topic": "multi-agent-patterns",
            }
        ]
    )
    node = HybridRetrieveNode(retrieval_backend=backend)
    assert node._backend is backend
    hits = node._backend.search("customtoken")
    assert hits and hits[0]["doc_id"] == "custom-1"


class TestBilingualRetrieval:
    """A bilingual orchestration QA agent must actually retrieve for Japanese
    questions. Before this fix the backend tokenised with ``str.split()``, so a
    Japanese sentence collapsed into a single term and **every** Japanese
    question returned zero records (measured: 4/4 misses) — the same structural
    defect a sibling template hit in production. CJK bigrams + function-word removal fix
    it; the seed records carry EN/JP retrieval vocabulary so both scripts reach
    the same doc. Pinning the mapping keeps a seed edit from silently breaking
    it.
    """

    _CASES = [
        ("マルチエージェントのオーケストレーションパターンは？", "maf-patterns-001"),
        ("multi-agent orchestration patterns like handoff", "maf-patterns-001"),
        ("タスク分解はプランナーでどう設計する？", "maf-decomp-001"),
        ("task decomposition with a planner", "maf-decomp-001"),
        ("エージェント間の共有状態と協調の方法", "maf-coord-001"),
        ("shared state and coordination between agents", "maf-coord-001"),
        ("権限委譲のガバナンス境界と監査証跡", "maf-gov-001"),
        ("governance boundaries for task delegation", "maf-gov-001"),
        ("ランタイムのアーキテクチャと抽象化", "maf-arch-001"),
    ]

    def test_each_question_reaches_its_doc_in_both_scripts(self):
        from src.services.service import seed_orchestration_kb

        kb = seed_orchestration_kb()
        for question, doc_id in self._CASES:
            hits = [h["doc_id"] for h in kb.search(question, top_k=3)]
            assert doc_id in hits, f"{question!r} missed {doc_id}: got {hits}"

    def test_an_out_of_scope_question_retrieves_nothing(self):
        """Function-word removal is what makes this hold: without it a question
        sharing only particles/stopwords scores highly on any record.
        """
        from src.services.service import seed_orchestration_kb

        assert seed_orchestration_kb().search("今日の東京の天気は？", top_k=3) == []
