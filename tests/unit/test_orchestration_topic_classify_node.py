# CMN-C1-349 — Unit Tests: OrchestrationTopicClassifyNode

from framework.schemas.agent_status import AgentStatus

from src.nodes.orchestration_topic_classify_node import OrchestrationTopicClassifyNode
from src.services.service import ORCHESTRATION_TOPICS


def _topic(q):
    return OrchestrationTopicClassifyNode().execute({"validated_question": q})


def test_multi_agent_patterns():
    out = _topic("Which multi-agent orchestration patterns are supported?")
    assert out["topic"] == "multi-agent-patterns"
    assert out["topic_confidence"] == 0.85
    assert out["status"] == AgentStatus.SUCCESS.value


def test_task_decomposition():
    assert _topic("How do I decompose a task with a planner?")["topic"] == "task-decomposition"


def test_agent_coordination():
    assert _topic("How is shared state coordinated and termination decided?")["topic"] == "agent-coordination"


def test_enterprise_integration():
    assert _topic("How do agents integrate with SAP and ServiceNow?")["topic"] == "enterprise-integration"


def test_governance_delegation():
    assert _topic("What are the AI Act task delegation governance boundaries?")["topic"] == "governance-delegation"


def test_framework_architecture():
    assert _topic("Explain the framework runtime architecture and .NET surface")["topic"] == "framework-architecture"


def test_japanese_query_classifies():
    assert _topic("マルチエージェントのオーケストレーション方法は?")["topic"] == "multi-agent-patterns"


def test_out_of_scope_default():
    out = _topic("What is the weather in Tokyo today?")
    assert out["topic"] == "out-of-scope"
    assert out["topic_confidence"] == 0.55


def test_topic_is_in_taxonomy():
    out = _topic("multi-agent handoff")
    assert out["topic"] in ORCHESTRATION_TOPICS


def test_self_skips_on_error():
    assert OrchestrationTopicClassifyNode().execute({"error_code": "X"}) == {}
