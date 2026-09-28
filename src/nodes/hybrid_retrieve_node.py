"""HybridRetrieveNode (step 3) — main slot, step 2.

Topic-biased RAG over the orchestration KB via the injected backend. Microsoft
agent-framework facts (pattern names, version notes, integration guidance) are
KB *data* (docs/02_design.md §1.2) — this node never hard-codes them. Builds the
``citations`` envelope (one marker per retrieved doc) consumed by ResponseFormat
and verified by the S-3 gate. External-service boundary (PB-3); emits an S-4
audit event (counts only).
"""

from __future__ import annotations

import json
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import KBRetrievalBackend, seed_orchestration_kb
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

_TOP_K = 6


class HybridRetrieveNode(FunctionNode):
    """Retrieve governing KB docs (dependency-injected backend) + build citations."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, retrieval_backend: KBRetrievalBackend | None = None, top_k: int = _TOP_K) -> None:
        super().__init__()
        # Default = illustrative in-memory seed KB (D-form fleet idiom). A bare
        # Graph() is what both server.py and the Marketplace runner construct, so
        # the default must be a working deployment form — with no backend every
        # run degraded to RETRIEVAL_BACKEND_MISSING. Deployments with a real
        # vector store still inject their own backend here (unchanged).
        self._backend: KBRetrievalBackend = retrieval_backend or seed_orchestration_kb()
        self._top_k = top_k

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}
        if self._backend is None:
            return {
                "error_code": "RETRIEVAL_BACKEND_MISSING",
                "error_message": "HybridRetrieveNode: no KBRetrievalBackend injected",
                "status": AgentStatus.ERROR.value,
            }

        query = state.get("validated_question") or state.get("question") or ""
        topic = state.get("topic") or "out-of-scope"

        # An out-of-scope question has no grounding in the orchestration KB — skip
        # retrieval so ResponseFormat returns a grounded refusal rather than an
        # answer stitched from incidental lexical matches.
        if topic == "out-of-scope":
            emit_trace_event("hybrid_retrieve", {"hit_count": 0, "top_k": self._top_k, "topic": topic}, state)
            return {
                "retrieved_docs": json.dumps([], ensure_ascii=False),
                "retrieval_hit_count": 0,
                "citations": json.dumps([], ensure_ascii=False),
                "status": AgentStatus.SUCCESS.value,
            }

        try:
            docs = self._backend.search(query, top_k=self._top_k, topic=topic) or []
        except Exception as exc:  # graceful degradation per design §9
            return {
                "error_code": "RETRIEVAL_FAILED",
                "error_message": f"HybridRetrieveNode: backend error: {exc}",
                "status": AgentStatus.ERROR.value,
            }

        citations = [
            {
                "marker": f"[{d.get('doc_id', '')}]",
                "doc_id": d.get("doc_id", ""),
                "title": d.get("title", ""),
                "source": d.get("source", ""),
            }
            for d in docs
        ]

        emit_trace_event(
            "hybrid_retrieve",
            {"hit_count": len(docs), "top_k": self._top_k, "topic": topic},
            state,
        )

        return {
            "retrieved_docs": json.dumps(docs, ensure_ascii=False),
            "retrieval_hit_count": len(docs),
            "citations": json.dumps(citations, ensure_ascii=False),
            "status": AgentStatus.SUCCESS.value,
        }
