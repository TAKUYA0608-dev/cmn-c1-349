# CMN-C1-349 — Template Design Specification

Enterprise AI Agent Orchestration & Multi-Agent Workflow Design Q&A
(`AgentOrchestrationDesignQAAgent`) — a Cat 1 knowledge Q&A agent over the
Microsoft agent-framework (Python/.NET) orchestration knowledge base, with Japan
enterprise context (SAP/ServiceNow integration, METI DX, Japan AI Regulation Bill
2026 agent task-delegation governance).

## §1 Position in AgentCore Architecture

- **Agent Class**: `AgentOrchestrationDesignQAAgent`
- **L1 Base**: **`AgentBaseGraph`** (Cat 1 — single technical capability: grounded
  framework-design Q&A; use-case-agnostic across enterprise architects / SIers).
- **Three-Layer Separation**:
  - **State**: flat `TypedDict` (`OrchestrationQAState`) — primitives + JSON strings only.
  - **Node**: L1 inheritance (Template Method `execute(self, state, config=None) -> dict`).
  - **Graph**: composition — 6 domain steps registered into the 3 writable slots.

### §1.1 Domain

The agent answers questions about Microsoft agent-framework orchestration:
multi-agent patterns (sequential / concurrent / group-chat / handoff / magentic),
task decomposition, agent coordination, enterprise integration, and Japan AI Act
2026 task-delegation governance. It is a **grounded RAG Q&A** — every answer claim
cites a KB document, and out-of-scope questions are refused, not hallucinated.

### §1.2 Facts are KB data, not code

Microsoft agent-framework facts (pattern names, framework versions, integration
guidance) live in the retrieval KB. The template hard-codes **no** framework facts
or version numbers — `FrameworkVersionInjectNode` derives the version context from
the retrieved docs' metadata, so the agent stays current as the (fast-moving) KB
updates.

## §2 Architecture Overview — 6-node Q&A pipeline

| # | Node | Slot | Responsibility | Input | Output |
|---|------|------|----------------|-------|--------|
| 1 | `QueryNormalizeNode` | pre_process | S-1/S-2: I/O bridge → `question`, NFKC, length cap (≤4000), injection reject | `user_input`/`input_context` | `validated_question` |
| 2 | `OrchestrationTopicClassifyNode` | main | classify into one orchestration topic (+confidence) via keyword signals | `validated_question` | `topic`, `topic_confidence` |
| 3 | `HybridRetrieveNode` | main | topic-biased RAG over the orchestration KB; build citations envelope | `validated_question`, `topic` | `retrieved_docs`, `retrieval_hit_count`, `citations` |
| 4 | `FrameworkVersionInjectNode` | main | derive framework version context from retrieved docs | `retrieved_docs` | `framework_version_context` |
| 5 | `ResponseFormatNode` | main | per-sentence-cited answer; refuse when 0 hits; disclaimer + JP note as separate fields | `retrieved_docs`, `framework_version_context` | `answer`, `disclaimer`, `jp_enterprise_note` |
| 6 | `SecurityGateOutputNode` | post_process | **S-3 citation gate**: every claim cites a `[<doc_id>]` (regenerate-once → reject); terminal S-4 audit | `answer`, `retrieval_hit_count` | `citation_status`, `uncited_claims`, `audit_logged` |

Steps 2–5 are composed into the single `main` slot (`OrchestrationMainNode`); the
framework owns initialize / finalize, edge wiring, and routing.

### §3 Data Flow

```
START → initialize → QueryNormalizeNode (pre_process: S-1/S-2)
                   → OrchestrationMainNode (main):
                        TopicClassify → HybridRetrieve → FrameworkVersionInject → ResponseFormat
                   → SecurityGateOutputNode (post_process: S-3 citation gate + audit)
                   → finalize → END
```

Slot mapping: `pre_process = QueryNormalizeNode`, `main = OrchestrationMainNode`
(4-step composite), `post_process = OrchestrationPostNode` (= `SecurityGateOutputNode`).

### §4 State Definition (flat `TypedDict`)

| Field | Type | Purpose |
|-------|------|---------|
| `question` | `str` | NL orchestration / framework-design question (input) |
| `validated_question` | `str` | NFKC-normalised, injection-screened (after QueryNormalize) |
| `topic` | `str` | one of the 7 orchestration topics (after Classify) |
| `topic_confidence` | `float` | 0.0–1.0 |
| `retrieved_docs` | `str` (JSON) | `[{doc_id,title,text,score,source,framework_version}]` |
| `retrieval_hit_count` | `int` | KB records retrieved (0 ⇒ refusal) |
| `citations` | `str` (JSON) | `[{marker,doc_id,title,source}]` |
| `framework_version_context` | `str` (JSON) | `{framework,latest_version,channel,as_of,notes}` |
| `answer` | `str` | per-sentence-cited answer (inline `[<doc_id>]`) |
| `disclaimer` / `jp_enterprise_note` | `str` | meta-statements surfaced alongside the gated answer |
| `citation_status` | `str` | `passed` / `regenerated` / `rejected` |
| `uncited_claims` | `str` (JSON) | claims lacking a citation |
| `audit_logged` | `bool` | terminal audit emitted |
| `error_code` / `error_message` | `str` | error propagation (downstream nodes self-skip) |

**Constraints (mandatory):** flat TypedDict, primitives + JSON strings only; no
JWT/API-key/credential fields; no Pydantic/dataclass; `InvocationContext` only via
`config["configurable"]`, never State.

## §5 Framework Utilization

- **S-1 Trust Gate**: framework-enforced in `BaseNode.__call__()`; the agent is for
  `VERIFIED_EXTERNAL` callers (config/agent.yaml `required_trust_level`).
- **S-2 Input boundary**: `QueryNormalizeNode.execute()` — NFKC + length cap (≤4000)
  + prompt-injection rejection. The framework `_security_gate_input()` (`@final`) runs
  the standard PII scan under the published SDK; domain validation lives in `execute()`.
- **S-3 Output gate**: `SecurityGateOutputNode.execute()` — citation completeness
  (every claim cites a `[<doc_id>]`; regenerate-once → reject). The framework
  `_security_gate_output()` (`@final`) runs the credential scan under the SDK.
- **S-4 Audit**: `emit_trace_event()` (from `src.utils.audit`, platform-binding with a
  stderr JSON-Lines fallback) is called for every side-effect step (classify, retrieve,
  version-inject, gate). `node_start`/`complete`/`error` are framework-emitted — not duplicated.
- **S-5 Credential scan / dep pinning**: no credentials in `src/`; deps exactly `==` pinned.

### §5.1 Composition Pattern

| Aspect | Decision |
|--------|----------|
| Pattern | **Standalone** (no `GraphNode` subgraph; no `RemoteAgentNode`). Cat 1 — the 6 steps compose into the fixed 3 slots. |
| Error propagation | **handle**: domain errors set `error_code`; downstream nodes self-skip; `OrchestrationMainNode` forces `status=SUCCESS` so the graph always reaches `post_process` (S-3 gate + audit) and `finalize` — a domain error never short-circuits the audit. |
| Backends | Retrieval KB + LLM are dependency-injected (`KBRetrievalBackend` / `LLMClient` Protocols). Production binds the real vector store + platform LLM via `ctx.secrets`; tests bind in-memory / stub. |

## §6 Retrieval & grounding strategy

- The topic biases retrieval toward the relevant KB slice; an `out-of-scope` topic
  **skips retrieval** so the answer is a grounded refusal, not an answer stitched from
  incidental lexical matches.
- Every answer sentence carries an inline `[<doc_id>]` citation; the S-3 gate rejects
  any answer with uncited substantive claims (after one regeneration attempt).

## §7 Import Isolation Confirmation

- [x] No Level 0 (`agenticstar`) import; imports limited to `framework.*` + `src.*` + stdlib.
- [x] Service layer (`src/services/service.py`) is pure domain logic — no `framework.*` dependency.

## §8 Design Decision Record

| Decision | Chosen | Rationale |
|----------|--------|-----------|
| L1 base type | `AgentBaseGraph` | Cat 1 single capability; no autonomous loop needed |
| Composition | 6 nodes → 3 slots (pre / main-composite / post) | mirrors the flagship Cat 1 Q&A pattern |
| Out-of-scope handling | classify → skip retrieve → grounded refusal | prevents hallucination outside the framework domain |
| Framework version | derived from KB doc metadata, never hard-coded | keeps a fast-moving framework's answers current (§1.2) |
| Disclaimer/JP-note | separate state fields, not in gated answer | keeps the S-3 gate focused on grounded claims |
| LLM binding (ADR-7) | explicit `llm_client` > `config["llm"]` (adapted) > none; **no silent stub** | the Marketplace runner constructs `Graph(config=...)`; a stub bound by default would run on the Marketplace even with Azure keys registered |

### ADR-7 — `config["llm"]` seam, deterministic answer path kept (2026-09-03)

- **Context.** The Marketplace runner constructs the agent as `agent_cls(config=<config.yaml>)`
  — no `llm_client` keyword — and the fleet entry point can only place a lazily-resolved
  Azure client under `config["llm"]` (an object answering `invoke(prompt)` / `complete(prompt)`).
  Before this ADR the graph read only the keyword, and `ResponseFormatNode` /
  `SecurityGateOutputNode` bound a `StubLLMClient` when nothing was injected, so the
  Marketplace deployment ran on the stub regardless of registered keys.
- **Decision.** `service.resolve_llm_client(explicit, config)`: explicit `llm_client` kw >
  `config["llm"]` (adapted by `service.ConfigLLMAdapter` to `generate(prompt)`; an unusable
  object raises at construction) > none. The stub is never bound by default (tests inject it).
- **Deterministic answer path KEPT.** Answer synthesis (`ResponseFormatNode`) is extractive and
  per-sentence cited — it never calls the LLM — so a bare run still answers with citations.
  The only LLM use is the S-3 regeneration step: with no LLM bound and uncited claims present,
  `SecurityGateOutputNode` withholds the answer (same safe refusal as a failed regeneration),
  sets `error_code=LLM_NOT_CONFIGURED` (SUCCESS + non-empty output) and emits the S-4 event
  `llm_not_configured` — a named reason, never a stub rewrite.
- **Consequences.** `tests/integration/test_full_path_invoke.py` pins the bare-construction
  path on the real SDK, the seam, the explicit-kw precedence, and the named rejection.
