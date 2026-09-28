# CMN-C1-349 — Test Specification

## Test Strategy
- Coverage target: **80%** (enforced via `--cov-fail-under=80` in CI); **actual ≈ 99%**.
- Test types: Unit (per-node) / Integration (agent.invoke e2e + HTTP) / Proof-of-Boundary.
- Backends are dependency-injected: deterministic `InMemoryKBBackend` + `StubLLMClient`
  keep every path offline-testable; production binds the real vector store + LLM.

## Framework Compliance Tests (`tests/unit/test_framework_compliance.py`)

| TC-ID | Test | Expected Result | Result |
|-------|------|----------------|--------|
| TC-01 | State contract: flat TypedDict | `__required_keys__` present, no Pydantic/dataclass | PASS |
| TC-02 | Input boundary rejects injection | `error_code=INJECTION_DETECTED`; `SecurityViolationError` importable | PASS |
| TC-03 | No JWT/credential fields in State | 0 banned field names | PASS |
| TC-04 | InvocationContext not in State | no `InvocationContext` in type hints | PASS |
| TC-05 | S-4 `emit_trace_event()` callable | event written with `template_id` + `trace_id` | PASS |
| TC-06/07 | `_security_gate_input/output()` not overridden | domain nodes use `execute()`; gates `@final` (SDK) | PASS |
| TC-08 | `required_trust_level` enforced (S-1) | ANONYMOUS → refused; INTERNAL → runs | PASS |

## Proof-of-Boundary Tests (`tests/proof_of_boundary/`)

| PB-ID | Boundary | Expected Result | Result |
|-------|----------|----------------|--------|
| PB-1 | BaseNode → AuditLogger (`test_pb_trace_events.py`) | classify/retrieve/version/gate events emitted; no silent failure | PASS |
| PB-2 | State serialization (`test_state_safety.py`) | primitives only, no Pydantic/dataclass | PASS |
| PB-4 | Import isolation (`test_import_isolation.py`) | AST scan: 0 Level-0 imports | PASS |
| PB-6 | Invoke execution order (`test_pb_invoke_order.py`) | Initialize → QueryNormalize → Main → SecurityGateOutput → Finalize | PASS |

> PB-3 (real external service) / PB-5 (checkpoint inspection) are exercised under the
> published SDK in STG; locally the DI in-memory backend stands in (PB-3) and the flat
> JSON-string State guarantees checkpoint safety (PB-5, covered structurally by PB-2).

## Business Logic Tests (`tests/integration/test_business_logic.py`)

| BL-ID | Input | Expected Result | Result |
|-------|-------|----------------|--------|
| BL-01 | multi-agent patterns question | `topic=multi-agent-patterns`, hits≥1, `citation_status=passed`, audited | PASS |
| BL-02 | task-decomposition question | `topic=task-decomposition`, hits≥1 | PASS |
| BL-03 | enterprise-integration question | `topic=enterprise-integration`, SAP/ServiceNow doc cited | PASS |
| BL-04 | governance-delegation question | `topic=governance-delegation`, hits≥1 | PASS |
| BL-05 | any in-scope answer | every answer sentence carries a `[<doc_id>]` citation | PASS |
| BL-06 | framework version surfaced | `framework_version_context.latest_version` derived from docs | PASS |
| BL-07 | out-of-scope ("weather in Tokyo") | `topic=out-of-scope`, hits=0, grounded refusal | PASS |
| BL-08 | injection input | `error_code=INJECTION_DETECTED` | PASS |
| BL-09 | empty input | `error_code=INPUT_EMPTY`, still reaches `FinalizeNode` | PASS |
| BL-10 | HTTP `/invoke` + `/health` (`test_api_endpoint.py`) | 200 + structured payload; health reports agent id | PASS |

Graph wiring (`tests/integration/test_graph_wiring.py`) verifies the 5-slot topology,
state schema, `Graph` alias, and node_history (class names + canonical order).

## Test Execution Summary

| Metric | Value |
|--------|-------|
| Execution date | 2026-06-17 |
| Total tests | 126 |
| Pass / Fail / Skip | 126 / 0 / 0 |
| Coverage (enforced) | 80% |
| Coverage (actual) | ≈ 99% |
| Composition | Unit (per-node + framework compliance) + Integration (e2e + HTTP) + PB |
