"""OrchestrationMainNode (main slot) — composes the 4 reasoning steps.

The framework exposes three writable slots; the orchestration-Q&A core is four
steps composed into this single ``main`` node in fixed order:

    OrchestrationTopicClassify → HybridRetrieve → FrameworkVersionInject → ResponseFormat

Each sub-node returns only its changed fields and self-skips on ``error_code``.
This composite threads a running state to the sub-nodes but returns ONLY the
accumulated deltas (per the "return changed fields only" node contract).
Retrieval + LLM backends are dependency-injected into the sub-nodes that need them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

from src.services.service import KBRetrievalBackend, LLMClient
from src.nodes.orchestration_topic_classify_node import OrchestrationTopicClassifyNode
from src.nodes.hybrid_retrieve_node import HybridRetrieveNode
from src.nodes.framework_version_inject_node import FrameworkVersionInjectNode
from src.nodes.response_format_node import ResponseFormatNode
from src.utils.audit import emit_trace_event


class OrchestrationMainNode(FunctionNode):
    """main slot — TopicClassify → HybridRetrieve → FrameworkVersionInject → ResponseFormat."""

    # CoE ARCH-0918-R1-01: declare the S-1 trust gate (agent requires VERIFIED_EXTERNAL).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        retrieval_backend: KBRetrievalBackend | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__()
        self._seq: list[FunctionNode] = [
            OrchestrationTopicClassifyNode(),
            HybridRetrieveNode(retrieval_backend=retrieval_backend),
            FrameworkVersionInjectNode(),
            ResponseFormatNode(llm_client=llm_client),
        ]

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        working = dict(state)
        deltas: dict[str, Any] = {}
        for node in self._seq:
            updates = node.execute(working, config) or {}
            working.update(updates)
            deltas.update(updates)
        # Always report SUCCESS — even when a sub-node set error_code — so the
        # framework routes on to post_process instead of aborting. The error_code
        # rides along in deltas and is surfaced by SecurityGateOutput (gate
        # pass-through + terminal audit) at the terminus, keeping every invocation
        # auditable rather than silently short-circuited.
        deltas["status"] = AgentStatus.SUCCESS.value
        # S-4 (central-CI audit-trace gate): record that this
        # boundary node completed. Field NAMES only — never values.
        emit_trace_event("orchestration_main_completed", {}, state)
        return deltas


# Backward-compat: the scaffold test/layout references `MainNode`.
MainNode = OrchestrationMainNode
