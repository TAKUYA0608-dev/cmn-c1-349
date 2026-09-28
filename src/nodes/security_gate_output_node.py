"""SecurityGateOutputNode (step 6) — post_process slot. **The S-3 citation gate.**

The quality core: every substantive sentence in the answer MUST cite a KB doc
(``[<doc_id>]``). Uncited sentences trigger one regeneration with a "cite every
claim" instruction; if claims are still uncited, the answer is rejected (replaced
with a safe refusal). With NO LLM bound (``llm_client`` / ``config["llm"]``) the
regeneration cannot run: the answer is rejected the same way and the run carries
``error_code=LLM_NOT_CONFIGURED`` — a named reason, never a stub rewrite. This is domain validation (LLM regeneration) inside
``execute()`` — distinct from the framework ``@final`` credential gate.

Also emits the terminal S-4 audit event (counts only) and sets ``audit_logged``.

citation_status: passed | regenerated | rejected.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel

from src.services.service import LLMClient
from src.utils.audit import emit_trace_event

# A KB citation: a bracketed doc-id-ish token (letters/digits/-/_/:/.).
_CITATION = re.compile(r"\[[A-Za-z0-9][\w\-:.]*\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*")
_MIN_CLAIM_WORDS = 5
# Japanese sentences don't split on whitespace, so a claim is also "substantive"
# once it reaches this character length (mirrors a fix made in a sibling template).
_MIN_CLAIM_CHARS = 16

_REJECTED = (
    "I can't ground every claim in a framework document, so I'm withholding this "
    "answer to avoid unsupported assertions about the Microsoft agent-framework. "
    "Please rephrase toward a documented orchestration topic."
)


def _uncited_claims(answer: str) -> list[str]:
    """Return substantive answer sentences that lack a [<doc_id>] citation."""
    if not answer:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT.split(answer) if s.strip()]
    out: list[str] = []
    for s in sentences:
        substantive = len(s.split()) >= _MIN_CLAIM_WORDS or len(s) >= _MIN_CLAIM_CHARS
        if substantive and not _CITATION.search(s):
            out.append(s)
    return out


class SecurityGateOutputNode(FunctionNode):
    """S-3: enforce that every answer claim cites a KB doc; emit terminal audit."""

    # CoE ARCH-0918-R1-01: declare the S-1 trust gate (agent requires VERIFIED_EXTERNAL).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        super().__init__()
        # None = no LLM bound. Resolved by the graph (explicit kw > config["llm"]);
        # a bare node stays unbound and rejects instead of rewriting from a stub.
        self._llm: LLMClient | None = llm_client

    def _audit(self, state: dict[str, Any], status: str, uncited: int) -> None:
        emit_trace_event(
            "security_gate_output",
            {"citation_status": status, "uncited_count": uncited},
            state,
        )

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        answer = state.get("answer") or ""
        # Nothing to gate (upstream error, empty answer, or a 0-hit out-of-scope
        # refusal — a refusal is a meta-statement, not a grounded claim) → pass.
        if not answer or state.get("error_code") or (state.get("retrieval_hit_count") or 0) <= 0:
            self._audit(state, "passed", 0)
            return {
                "citation_status": "passed",
                "uncited_claims": json.dumps([], ensure_ascii=False),
                "audit_logged": True,
                "status": AgentStatus.SUCCESS.value,
            }

        uncited = _uncited_claims(answer)
        if not uncited:
            self._audit(state, "passed", 0)
            return {
                "citation_status": "passed",
                "uncited_claims": json.dumps([], ensure_ascii=False),
                "audit_logged": True,
                "status": AgentStatus.SUCCESS.value,
            }

        if self._llm is None:
            # No LLM to regenerate with: withhold the answer (same safe refusal as a
            # failed regeneration) and name the cause. S-4: field names + counts only.
            emit_trace_event("llm_not_configured", {"uncited_count": len(uncited)}, state)
            self._audit(state, "rejected", len(uncited))
            return {
                "answer": _REJECTED,
                "citation_status": "rejected",
                "uncited_claims": json.dumps(uncited, ensure_ascii=False),
                "error_code": "LLM_NOT_CONFIGURED",
                "error_message": (
                    "SecurityGateOutputNode: uncited claims found but no LLM bound "
                    "(llm_client / config['llm']); regeneration skipped, answer withheld"
                ),
                "audit_logged": True,
                "status": AgentStatus.SUCCESS.value,
            }

        # Regenerate once, demanding a [<doc_id>] citation on every claim.
        correction = (
            "Your previous answer contained claims with no KB citation. Rewrite it so "
            "EVERY sentence that states a fact ends with an inline [<doc_id>] citation "
            "drawn only from the retrieved documents. Do not add unsupported claims.\n\n"
            "=== Previous answer ===\n" + answer
        )
        regenerated = self._llm.generate(correction)
        residual = _uncited_claims(regenerated)

        if residual:
            self._audit(state, "rejected", len(residual))
            return {
                "answer": _REJECTED,
                "citation_status": "rejected",
                "uncited_claims": json.dumps(uncited, ensure_ascii=False),
                "audit_logged": True,
                "status": AgentStatus.SUCCESS.value,
            }

        self._audit(state, "regenerated", len(uncited))
        return {
            "answer": regenerated,
            "citation_status": "regenerated",
            "uncited_claims": json.dumps(uncited, ensure_ascii=False),
            "audit_logged": True,
            "status": AgentStatus.SUCCESS.value,
        }
