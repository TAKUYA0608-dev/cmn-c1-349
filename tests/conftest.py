# CMN-C1-349 — shared test fixtures: seeded orchestration KB + compiled agent.

import pytest

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import AgentOrchestrationDesignQAAgent
from src.services.service import InMemoryKBBackend, StubLLMClient

# A small but topically-diverse orchestration KB (Microsoft agent-framework).
_SEED = [
    {"doc_id": "ms-af-pat-001", "title": "Sequential orchestration",
     "text": "Sequential orchestration runs agents in a fixed order passing output forward",
     "source": "ms-learn", "framework_version": "1.0.0", "topic": "multi-agent-patterns"},
    {"doc_id": "ms-af-pat-002", "title": "Group chat orchestration",
     "text": "Group chat orchestration coordinates multiple agents through a shared conversation",
     "source": "ms-learn", "framework_version": "1.0.0", "topic": "multi-agent-patterns"},
    {"doc_id": "ms-af-dec-001", "title": "Planner task decomposition",
     "text": "A planner decomposes a goal into ordered sub tasks for delegation",
     "source": "ms-learn", "framework_version": "1.0.0", "topic": "task-decomposition"},
    {"doc_id": "ms-af-int-001", "title": "Enterprise integration connectors",
     "text": "Connectors integrate agents with SAP and ServiceNow enterprise systems",
     "source": "ms-learn", "framework_version": "1.0.0", "topic": "enterprise-integration"},
    {"doc_id": "ms-af-gov-001", "title": "Task delegation governance",
     "text": "Agent task delegation boundaries must be auditable for governance compliance",
     "source": "ms-learn", "framework_version": "0.9.0", "topic": "governance-delegation"},
]


def seeded_backend() -> InMemoryKBBackend:
    b = InMemoryKBBackend()
    b.add([dict(r, score=0.0) for r in _SEED])
    return b


@pytest.fixture
def backend() -> InMemoryKBBackend:
    return seeded_backend()


@pytest.fixture
def agent(backend) -> AgentOrchestrationDesignQAAgent:
    a = AgentOrchestrationDesignQAAgent(retrieval_backend=backend, llm_client=StubLLMClient())
    a.compile()
    return a


@pytest.fixture
def ctx() -> InvocationContext:
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, session_id="test")
