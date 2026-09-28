# CMN-C1-349 — Integration Tests: graph wiring + 3-slot composition

from src.graph.graph import AgentOrchestrationDesignQAAgent, Graph
from src.schemas.state import OrchestrationQAState


class TestGraphTopology:
    def test_slot_set(self):
        g = AgentOrchestrationDesignQAAgent()
        g.register_nodes()
        assert set(g._nodes.keys()) == {
            "initialize", "pre_process", "main", "post_process", "finalize"
        }

    def test_compiles_ok(self):
        g = AgentOrchestrationDesignQAAgent()
        g.compile()
        assert g._compiled is not None

    def test_state_schema(self):
        assert AgentOrchestrationDesignQAAgent().state_schema is OrchestrationQAState

    def test_name(self):
        assert AgentOrchestrationDesignQAAgent().name == "AgentOrchestrationDesignQAAgent"

    def test_graph_alias(self):
        assert Graph is AgentOrchestrationDesignQAAgent


class TestNodeHistory:
    # node_history records node CLASS names; the 4 main-slot sub-nodes are
    # composed into OrchestrationMainNode, the S-3 gate is SecurityGateOutputNode.
    _CANONICAL = [
        "InitializeNode", "QueryNormalizeNode", "OrchestrationMainNode",
        "SecurityGateOutputNode", "FinalizeNode",
    ]

    def test_all_slot_nodes_run(self, agent, ctx):
        r = agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})
        for n in self._CANONICAL:
            assert n in r["node_history"], f"{n} missing: {r['node_history']}"

    def test_order_is_canonical(self, agent, ctx):
        r = agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})
        history = r["node_history"]
        idx = {n: history.index(n) for n in self._CANONICAL}
        for a, b in zip(self._CANONICAL, self._CANONICAL[1:]):
            assert idx[a] < idx[b], f"{a} must precede {b}: {history}"

    def test_initialize_first_finalize_last(self, agent, ctx):
        r = agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})
        assert r["node_history"][0] == "InitializeNode"
        assert r["node_history"][-1] == "FinalizeNode"
