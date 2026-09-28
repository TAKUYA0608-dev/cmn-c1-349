# PB-1 — BaseNode → AuditLogger boundary.
#
# Every domain side-effect operation must emit a trace event (no silent failures).
# ``emit_trace_event`` delegates to the platform AuditSink under the published SDK
# and to a stderr fallback under a local SDK stub, so asserting on stderr is environment-
# dependent. Instead we SPY on the underlying ``_platform_emit`` binding, which
# makes the assertion identical in both environments.

import pytest


@pytest.fixture
def trace_sink(monkeypatch):
    """Capture every emit_trace_event(event_type, payload, state) call."""
    events: list[dict] = []

    def _rec(event_type, payload, state):
        events.append({"event_type": event_type, "payload": payload or {}, "state": state or {}})

    monkeypatch.setattr("src.utils.audit._platform_emit", _rec)
    return events


class TestPBTraceEvents:
    def test_domain_events_emitted_on_happy_path(self, agent, ctx, trace_sink):
        agent.invoke("Which multi-agent orchestration patterns are supported?", ctx=ctx, input_context={})
        types = {e["event_type"] for e in trace_sink}
        assert {"orchestration_topic_classify", "hybrid_retrieve",
                "framework_version_inject", "security_gate_output"}.issubset(types)

    def test_events_carry_trace_id(self, agent, ctx, trace_sink):
        agent.invoke("sequential orchestration", ctx=ctx, input_context={})
        retr = [e for e in trace_sink if e["event_type"] == "hybrid_retrieve"]
        assert retr, "expected a hybrid_retrieve event"
        assert retr[0]["state"].get("trace_id")

    def test_security_gate_event_records_status(self, agent, ctx, trace_sink):
        agent.invoke("multi-agent orchestration patterns", ctx=ctx, input_context={})
        gate = [e for e in trace_sink if e["event_type"] == "security_gate_output"]
        assert gate and gate[-1]["payload"]["citation_status"] == "passed"

    def test_no_silent_failure_on_refusal(self, agent, ctx, trace_sink):
        # Even an out-of-scope refusal emits the terminal gate audit event.
        agent.invoke("weather in tokyo", ctx=ctx, input_context={})
        assert "security_gate_output" in {e["event_type"] for e in trace_sink}
