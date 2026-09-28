"""CMN-C1-349 AgentOrchestrationDesignQAAgent — graph composition.

Cat 1: L1-direct inheritance from AgentBaseGraph. The 6 domain steps are composed
into the framework's three writable slots; the framework owns initialize/finalize,
edge wiring, and routing.

    pre_process  = QueryNormalizeNode    (S-1/S-2 input boundary)
    main         = OrchestrationMainNode (TopicClassify → HybridRetrieve
                                          → FrameworkVersionInject → ResponseFormat)
    post_process = OrchestrationPostNode (SecurityGateOutputNode — S-3 citation gate + audit)

Retrieval + LLM backends are dependency-injected so the agent is offline-testable
and deployment-agnostic. A bare ``Graph(config=...)`` (the Marketplace runner form)
binds the illustrative seed KB; the LLM resolves as explicit ``llm_client`` >
``config["llm"]`` (the fleet entry point's lazy Azure client, adapted) > none.
Answer synthesis is deterministic (extractive, every sentence cited) so a run
without an LLM still answers; only the S-3 regeneration step needs the LLM, and
without one it rejects with the named reason ``LLM_NOT_CONFIGURED`` — no stub is
ever bound silently.
"""

from __future__ import annotations
from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph

from src.schemas.state import OrchestrationQAState
from src.services.service import KBRetrievalBackend, LLMClient, resolve_llm_client
from src.nodes.query_normalize_node import QueryNormalizeNode
from src.nodes.main_node import OrchestrationMainNode
from src.nodes.post_process_node import OrchestrationPostNode


class AgentOrchestrationDesignQAAgent(AgentBaseGraph):
    """Microsoft agent-framework orchestration & multi-agent workflow design Q&A (Cat 1, flagship)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        retrieval_backend: KBRetrievalBackend | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._retrieval_backend = retrieval_backend
        # explicit kw > config["llm"] > None (see service.resolve_llm_client)
        self._llm_client = resolve_llm_client(llm_client, config)
        super().__init__(config)

    @property
    def name(self) -> str:
        return "AgentOrchestrationDesignQAAgent"

    @property
    def state_schema(self) -> type:
        """Domain State so per-node fields survive node merges (LangGraph drops
        keys not declared in the schema)."""
        return OrchestrationQAState

    def register_nodes(self) -> None:
        super().register_nodes()  # framework injects InitializeNode + FinalizeNode
        self._nodes["pre_process"] = QueryNormalizeNode()
        self._nodes["main"] = OrchestrationMainNode(
            retrieval_backend=self._retrieval_backend,
            llm_client=self._llm_client,
        )
        self._nodes["post_process"] = OrchestrationPostNode(llm_client=self._llm_client)

    def get_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Surface the orchestration-Q&A payload (this agent writes answer/topic/
        citations/etc., not the framework-default formatted_output)."""
        # The Marketplace runner rejects a successful invocation whose output
        # is missing (verified on a deployed Pod), and a degraded run
        # (SUCCESS + error_code) leaves "answer" unset. Report the degradation —
        # this states what happened, it does not invent an answer.
        #
        # Only on SUCCESS: a request refused by the S-2 gate (status ERROR) must
        # keep publishing nothing, or the refusal is undone.
        _output = state.get("answer")
        if not _output and str(state.get("status", "")).lower().endswith("success"):
            _code = state.get("error_code") or "NO_CONTENT"
            _output = (
                "This request could not be completed "
                f"(error_code={_code}). No content was produced; "
                "see error_code and error_log for the degradation cause."
            )
        return {
            "output": _output,
            "answer": state.get("answer"),
            "topic": state.get("topic"),
            "topic_confidence": state.get("topic_confidence"),
            "citations": state.get("citations"),
            "framework_version_context": state.get("framework_version_context"),
            "disclaimer": state.get("disclaimer"),
            "jp_enterprise_note": state.get("jp_enterprise_note"),
            "retrieval_hit_count": state.get("retrieval_hit_count"),
            "citation_status": state.get("citation_status"),
            "uncited_claims": state.get("uncited_claims"),
            "audit_logged": state.get("audit_logged"),
            "status": state.get("status"),
            "error_code": state.get("error_code"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
            "error_log": state.get("error_log", []),
        }


# Backward-compat alias — the scaffold (api/server.py) imports `Graph`.
Graph = AgentOrchestrationDesignQAAgent
