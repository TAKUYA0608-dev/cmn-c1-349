"""QueryNormalizeNode (step 1) — pre_process slot / S-1+S-2 input boundary.

Bridge the framework I/O contract (``user_input`` + ``input_context``) to the
domain ``question`` field, NFKC-normalise it, enforce a length cap, and reject
prompt-injection before any LLM sees the input.

Node contract: extend FunctionNode; override
``execute(self, state, config=None) -> dict``; return ONLY changed fields + an
AgentStatus enum value; read ``input_context`` read-only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event

# Max question length (docs/02_design.md §5). Mirrors api/server.py.
_MAX_LEN = 4000

# Prompt-injection / jailbreak signatures (reject, never forward to the LLM).
_INJECTION = re.compile(
    r"(?i)(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"
    r"|disregard\s+(?:the\s+)?(?:system|previous)\s+(?:prompt|instructions)"
    r"|reveal\s+(?:your\s+)?system\s+prompt"
    r"|you\s+are\s+now\s+(?:a|an|in)\b"
    r"|<\s*script\b|</\s*script\s*>)"
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class QueryNormalizeNode(FunctionNode):
    """S-1/S-2: bridge I/O → question, NFKC-normalise, cap length, reject injection."""

    # CoE ARCH-0918-R1-01: declare the S-1 trust gate (agent requires VERIFIED_EXTERNAL).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        if state.get("error_code"):
            return {}

        ic = state.get("input_context") or {}
        raw = state.get("question") or state.get("user_input") or ic.get("question") or ""

        if not raw or not str(raw).strip():
            return {
                "error_code": "INPUT_EMPTY",
                "error_message": "QueryNormalizeNode: question is empty or missing",
                "status": AgentStatus.ERROR.value,
            }
        if len(str(raw)) > _MAX_LEN:
            return {
                "error_code": "INPUT_TOO_LONG",
                "error_message": f"QueryNormalizeNode: question exceeds {_MAX_LEN} chars",
                "status": AgentStatus.ERROR.value,
            }
        if _INJECTION.search(str(raw)):
            return {
                "error_code": "INJECTION_DETECTED",
                "error_message": "QueryNormalizeNode: prompt-injection pattern rejected",
                "status": AgentStatus.ERROR.value,
            }

        validated = unicodedata.normalize("NFKC", str(raw))
        validated = _CONTROL.sub("", validated).strip()
        validated = re.sub(r"\s+", " ", validated)

        # S-4 (central-CI audit-trace gate): record that this
        # boundary node completed. Field NAMES only — never values.
        emit_trace_event("query_normalize_completed", {"fields": ["question", "status", "validated_question"]}, state)
        return {
            "question": validated,
            "validated_question": validated,
            "status": AgentStatus.SUCCESS.value,
        }
