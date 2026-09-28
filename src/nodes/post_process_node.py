"""OrchestrationPostNode (post_process slot) — the S-3 citation gate + terminal audit.

For this Cat 1 Q&A agent the post-process slot IS the S-3 security gate: it
verifies that every answer claim cites a KB doc (regenerate-once → reject) and
emits the terminal S-4 audit event. The implementation lives in
``SecurityGateOutputNode``; this module is the scaffold's canonical post_process
entry point and re-exports it as ``OrchestrationPostNode`` (aliased, not
subclassed) so node_history records the domain class name and the slot wiring +
MANIFEST layout stay aligned without duplicating the gate logic.
"""

from __future__ import annotations

from src.nodes.security_gate_output_node import SecurityGateOutputNode

# The post_process slot node (S-3 gate). Aliased so node_history records the
# domain class name ``SecurityGateOutputNode``.
OrchestrationPostNode = SecurityGateOutputNode
PostProcessNode = SecurityGateOutputNode  # scaffold-layout backward-compat

__all__ = ["OrchestrationPostNode", "PostProcessNode"]
