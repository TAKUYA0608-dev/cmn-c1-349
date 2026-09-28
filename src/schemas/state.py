"""CMN-C1-349 Enterprise AI Agent Orchestration & Multi-Agent Workflow Design Q&A — State schema.

Flat ``TypedDict`` per the node and state-safety contracts: all fields are Optional primitives or
JSON-serialised strings (msgpack round-trips cleanly through the LangGraph
checkpoint). No Pydantic, no dataclass, no arbitrary Python objects, and never
JWT / API keys / credentials (checkpoint DB leakage). The caller's question is
normalised at the input boundary; nothing identity-bearing is persisted.

Pipeline (docs/02_design.md §2):
    QueryNormalize → OrchestrationTopicClassify → HybridRetrieve
    → FrameworkVersionInject → ResponseFormat → SecurityGateOutput (S-3)
"""

from __future__ import annotations

from typing import Optional

from framework.schemas.agent_state import AgentState


class OrchestrationQAState(AgentState):
    """State for the Microsoft agent-framework orchestration Q&A pipeline.

    Shared fields (user_input, status, session_id, node_history, error_log,
    caller_trust_level, trace_id, correlation_id, …) are inherited from
    AgentState. Only domain fields are declared here; LangGraph drops keys not in
    this schema, so every field the pipeline writes MUST appear below.
    """

    # ─── Input (caller supplies via user_input + input_context) ───
    question: Optional[str]  # NL orchestration / framework-design question

    # ─── QueryNormalizeNode (S-1/S-2 input boundary) ───
    validated_question: Optional[str]  # NFKC-normalised, injection-screened question

    # ─── OrchestrationTopicClassifyNode ───
    topic: Optional[str]  # one of services.ORCHESTRATION_TOPICS
    topic_confidence: Optional[float]  # 0.0–1.0

    # ─── HybridRetrieveNode (topic-biased RAG over the orchestration KB) ───
    retrieved_docs: Optional[str]  # JSON: [{doc_id,title,text,score,source,framework_version}]
    retrieval_hit_count: Optional[int]  # total KB records retrieved
    citations: Optional[str]  # JSON: [{marker,doc_id,title,source}]

    # ─── FrameworkVersionInjectNode ───
    framework_version_context: Optional[str]  # JSON: {framework,latest_version,channel,as_of,notes}

    # ─── ResponseFormatNode ───
    answer: Optional[str]  # cited answer — every sentence ends with an inline [<doc_id>]
    disclaimer: Optional[str]  # mandatory "design guidance, not a decision" notice
    jp_enterprise_note: Optional[str]  # Japan enterprise context (SAP/ServiceNow, METI DX, AI Reg Bill 2026)

    # ─── SecurityGateOutputNode (S-3 citation gate) ───
    citation_status: Optional[str]  # "passed" | "regenerated" | "rejected"
    uncited_claims: Optional[str]  # JSON list of claims lacking a [<doc_id>] reference
    audit_logged: Optional[bool]  # True once the terminal audit event is emitted

    # ─── Error propagation (any node; downstream nodes self-skip) ───
    error_code: Optional[str]
    error_message: Optional[str]


# Backward-compat alias: the scaffold (graph.py / server.py) references `State`.
State = OrchestrationQAState
