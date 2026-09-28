"""OrchestrationTopicClassifyNode (step 2) — main slot, step 1.

Classify the question into exactly one orchestration topic
(services.ORCHESTRATION_TOPICS) via deterministic keyword signals, with a
confidence score. The topic biases retrieval (HybridRetrieve) toward the most
relevant KB slice. ``out-of-scope`` is the safe default when no topic signal
matches — it short-circuits to a grounded refusal downstream rather than
hallucinating an answer outside the Microsoft agent-framework domain.
"""

from __future__ import annotations

import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel

# Confidence when a clear topic signal matched vs an uncertain default.
_CONF_CLEAR = 0.85
_CONF_UNCERTAIN = 0.55

# Topic signals (highest-priority first — first match wins). Each pattern targets
# the vocabulary of one orchestration topic in the Microsoft agent-framework domain.
_TOPIC_SIGNALS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "governance-delegation",
        re.compile(
            r"(?i)(governance|delegation|delegat|ai\s+act|regulation|compliance|audit\s+trail"
            r"|accountab|task\s+delegation|権限委譲|ガバナンス|統制)"
        ),
    ),
    (
        "enterprise-integration",
        re.compile(
            r"(?i)(sap|servicenow|salesforce|erp|enterprise\s+(?:system|integration)|connector"
            r"|legacy\s+system|on[\s-]?prem|mcp\s+server|external\s+api|基幹システム|連携)"
        ),
    ),
    (
        "task-decomposition",
        re.compile(
            r"(?i)(task\s+decompos|decompos|planner|planning|sub[\s-]?task|break\s+down|breakdown"
            r"|workflow\s+design|step\s+sequenc|タスク分解|計画)"
        ),
    ),
    (
        "multi-agent-patterns",
        re.compile(
            r"(?i)(multi[\s-]?agent|group[\s-]?chat|handoff|magentic|sequential|concurrent"
            r"|orchestrat|agent\s+team|swarm|round[\s-]?robin|pattern|マルチエージェント|オーケストレ)"
        ),
    ),
    (
        "agent-coordination",
        re.compile(
            r"(?i)(coordinat|shared\s+state|message\s+pass|termination|consensus|hand[\s-]?off"
            r"|communicat|state\s+management|協調|連携状態)"
        ),
    ),
    (
        "framework-architecture",
        re.compile(
            r"(?i)(architecture|abstraction|runtime|\.net|python\s+sdk|api\s+surface|class\s+hierarchy"
            r"|framework\s+(?:design|internals|component)|アーキテクチャ|構成)"
        ),
    ),
)


def _classify(question: str) -> tuple[str, float]:
    for topic, pattern in _TOPIC_SIGNALS:
        if pattern.search(question):
            return topic, _CONF_CLEAR
    return "out-of-scope", _CONF_UNCERTAIN


class OrchestrationTopicClassifyNode(FunctionNode):
    """Classify the question into one orchestration topic (+ confidence)."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        question = state.get("validated_question") or state.get("question") or ""
        topic, confidence = _classify(question)

        emit_trace_event(
            "orchestration_topic_classify",
            {"topic": topic, "confidence": confidence},
            state,
        )

        return {
            "topic": topic,
            "topic_confidence": confidence,
            "status": AgentStatus.SUCCESS.value,
        }
