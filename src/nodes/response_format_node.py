"""ResponseFormatNode (step 5) — main slot, step 4.

Compose the final answer from the retrieved KB docs, with an inline ``[<doc_id>]``
citation on EVERY substantive sentence (so the S-3 citation gate passes). The
disclaimer and Japan-enterprise context note are surfaced as SEPARATE state
fields (not folded into the gated answer body), mirroring the §2 design: the
``answer`` is the grounded, fully-cited core; meta-statements ride alongside.

When retrieval is empty (out-of-scope), it returns a grounded refusal — never an
ungrounded answer. Synthesis is deterministic (extractive, per-sentence cited) and
does not call the LLM; the optional ``llm_client`` is accepted for slot symmetry
only. No stub is bound by default.
"""

from __future__ import annotations

import json
import re
from typing import Any

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus

from src.services.service import LLMClient
from framework.schemas.trust_level import TrustLevel
from src.utils.audit import emit_trace_event

# Sentence splitter (ASCII enders with trailing whitespace + JP enders).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")

_DISCLAIMER = (
    "This is framework design guidance based on the cited documentation, not a "
    "production architecture decision — validate against your environment and "
    "Japan AI Regulation Bill 2026 task-delegation governance before adoption."
)
_JP_ENTERPRISE_NOTE = (
    "Japan enterprise context: confirm SAP/ServiceNow connector availability and "
    "METI DX alignment; agent task-delegation boundaries must be auditable per the "
    "Japan AI Regulation Bill 2026."
)
_REFUSAL = (
    "This question falls outside the Microsoft agent-framework orchestration "
    "knowledge base, so I can't ground an answer in the documentation. Please ask "
    "about multi-agent patterns, task decomposition, agent coordination, enterprise "
    "integration, governance/delegation, or framework architecture."
)


def _cite_sentences(text: str, marker: str) -> list[str]:
    """Split ``text`` into sentences and append ``marker`` inside each (before the
    final period) so every emitted sentence carries an inline citation."""
    out: list[str] = []
    for s in _SENT_SPLIT.split(text or ""):
        s = s.strip().rstrip(".!?。！？").strip()
        if s:
            out.append(f"{s} {marker}.")
    return out


class ResponseFormatNode(FunctionNode):
    """Compose the per-sentence-cited answer (or a grounded refusal when empty)."""

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__()
        # Not used for synthesis (deterministic extractive answer); None is fine.
        self._llm: LLMClient | None = llm_client

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        try:
            docs = json.loads(state.get("retrieved_docs") or "[]")
        except (json.JSONDecodeError, TypeError):
            docs = []

        if not docs:
            return {
                "answer": _REFUSAL,
                "disclaimer": _DISCLAIMER,
                "jp_enterprise_note": "",
                "status": AgentStatus.SUCCESS.value,
            }

        sentences: list[str] = []
        for d in docs:
            doc_id = d.get("doc_id", "")
            marker = f"[{doc_id}]"
            title = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            content = f"{title}. {body}" if title else body
            sentences.extend(_cite_sentences(content, marker))

        answer = "\n".join(sentences)
        # S-4 (central-CI audit-trace gate): record that this
        # boundary node completed. Field NAMES only — never values.
        emit_trace_event(
            "response_format_completed", {"fields": ["answer", "disclaimer", "jp_enterprise_note", "status"]}, state
        )
        return {
            "answer": answer,
            "disclaimer": _DISCLAIMER,
            "jp_enterprise_note": _JP_ENTERPRISE_NOTE,
            "status": AgentStatus.SUCCESS.value,
        }
