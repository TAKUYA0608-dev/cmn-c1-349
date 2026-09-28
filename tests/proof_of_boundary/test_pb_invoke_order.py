# PB-6 — Invoke execution order.
#
# Graph-level canonical order (node_history = class names):
#   InitializeNode → QueryNormalizeNode (pre / S-1+S-2) → OrchestrationMainNode (main)
#   → SecurityGateOutputNode (post / S-3) → FinalizeNode
# The S-3 gate (post_process) must always run before finalize so it is never
# bypassed; initialize is always first.

_CANONICAL = [
    "InitializeNode", "QueryNormalizeNode", "OrchestrationMainNode",
    "SecurityGateOutputNode", "FinalizeNode",
]


class TestPBInvokeOrder:
    def test_canonical_order(self, agent, ctx):
        h = agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})["node_history"]
        first = {n: h.index(n) for n in _CANONICAL}
        for a, b in zip(_CANONICAL, _CANONICAL[1:]):
            assert first[a] < first[b], f"{a} must precede {b}: {h}"

    def test_initialize_first(self, agent, ctx):
        h = agent.invoke("x orchestration", ctx=ctx, input_context={})["node_history"]
        assert h[0] == "InitializeNode"

    def test_finalize_last(self, agent, ctx):
        h = agent.invoke("x orchestration", ctx=ctx, input_context={})["node_history"]
        assert h[-1] == "FinalizeNode"

    def test_s3_gate_precedes_finalize(self, agent, ctx):
        h = agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})["node_history"]
        assert h.index("SecurityGateOutputNode") < h.index("FinalizeNode")

    def test_order_preserved_on_refusal(self, agent, ctx):
        h = agent.invoke("weather in tokyo", ctx=ctx, input_context={})["node_history"]
        assert h.index("QueryNormalizeNode") < h.index("SecurityGateOutputNode") < h.index("FinalizeNode")
