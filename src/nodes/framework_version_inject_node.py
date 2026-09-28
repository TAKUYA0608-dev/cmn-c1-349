"""FrameworkVersionInjectNode (step 4) — main slot, step 3.

Microsoft agent-framework is a NEW, fast-moving official framework (Python/.NET),
so a stale answer is a correctness risk. This node derives a *framework version
context* from the version metadata carried by the retrieved KB docs (never
hard-coded — version facts are KB data, docs/02_design.md §1.2) and threads it to
ResponseFormat so the answer is anchored to the framework version the cited docs
describe. When retrieval is empty (out-of-scope), it injects an explicit
"version context unavailable" marker rather than guessing.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import FRAMEWORK
from src.utils.audit import emit_trace_event
from framework.schemas.trust_level import TrustLevel


def _latest_version(versions: list[str]) -> str:
    """Pick the most-cited version string; ties resolve to the lexicographically
    greatest (a deterministic, KB-data-only heuristic — no hard-coded version)."""
    present = [v for v in versions if v]
    if not present:
        return ""
    counts = Counter(present)
    top = max(counts.values())
    return sorted([v for v, c in counts.items() if c == top])[-1]


class FrameworkVersionInjectNode(FunctionNode):
    """Inject the framework version context derived from retrieved docs."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        try:
            docs = json.loads(state.get("retrieved_docs") or "[]")
        except (json.JSONDecodeError, TypeError):
            docs = []

        versions = [str(d.get("framework_version", "")) for d in docs]
        latest = _latest_version(versions)

        if latest:
            context = {
                "framework": FRAMEWORK,
                "latest_version": latest,
                "channel": "official",
                "as_of": "from cited KB docs",
                "notes": f"Answer anchored to {FRAMEWORK} {latest} per the cited documentation.",
            }
        else:
            context = {
                "framework": FRAMEWORK,
                "latest_version": "",
                "channel": "unknown",
                "as_of": "n/a",
                "notes": "No version-tagged source retrieved; version context unavailable.",
            }

        emit_trace_event(
            "framework_version_inject",
            {"latest_version": latest, "doc_count": len(docs)},
            state,
        )

        return {
            "framework_version_context": json.dumps(context, ensure_ascii=False),
            "status": AgentStatus.SUCCESS.value,
        }
