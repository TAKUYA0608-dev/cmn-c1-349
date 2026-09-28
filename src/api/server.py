"""CMN-C1-349 — standalone HTTP entry point (adapter only; no business logic).

For platform-level routing, AgentGateway calls ``agent.invoke()`` directly. The
secrets binding (``shared.secrets`` / ``framework.secrets.context``) is part of
the published SDK surface and is absent under the local SDK-stub shim, so it is
imported defensively: when present the production secret provider is bound; when
absent the server still imports + serves (degraded, for local/offline runs).

``build_agent(...)`` is the single agent factory so tests can bind in-memory
backends without monkeypatching module state.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, cast
from uuid import uuid4

from fastapi import FastAPI, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.services.service import KBRetrievalBackend, LLMClient

try:  # pragma: no cover - present only under the published SDK
    from framework.secrets.context import bound_secrets as _bound_secrets
    from shared.secrets import factory as _secrets_factory
except Exception:  # ModuleNotFoundError under a local SDK stub / offline runs
    _bound_secrets = None
    _secrets_factory = None

_MAX_QUERY_LEN = 4000


def build_agent(
    retrieval_backend: KBRetrievalBackend | None = None,
    llm_client: LLMClient | None = None,
) -> Graph:
    """Construct + compile the agent, binding the production secret provider when available."""
    g = Graph(retrieval_backend=retrieval_backend, llm_client=llm_client)
    g.compile()
    if _secrets_factory is not None:  # pragma: no cover - SDK-only path
        g.provision_secrets(_secrets_factory(namespace="cmn-c1-349", agent_name="AgentOrchestrationDesignQAAgent"))
    return g


@contextmanager
def _secret_scope(agent: Graph) -> Iterator[None]:
    if _bound_secrets is not None:  # pragma: no cover - SDK-only path
        with _bound_secrets(agent._secrets_provider):
            yield
    else:
        yield


app = FastAPI(title="CMN-C1-349 AgentOrchestrationDesignQAAgent")
agent = build_agent()


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""


@app.post("/invoke", response_model=None)
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    with _secret_scope(agent):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=getattr(request.state, "trust_level", TrustLevel.VERIFIED_EXTERNAL),
            caller_id=getattr(request.state, "caller_id", ""),
        )
        return cast(dict[str, Any], agent.invoke(req.input, ctx=ctx))


@app.get("/health", response_model=None)
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "AgentOrchestrationDesignQAAgent"}
