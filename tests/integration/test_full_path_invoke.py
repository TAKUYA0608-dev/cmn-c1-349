# CMN-C1-349 — Integration: the production path, on the real SDK.
#
# Every test builds the agent the way the Marketplace runner does —
# `Graph(config=<config/config.yaml dict>)`, no keyword backends, no llm_client — or
# adds one dependency at a time to prove a specific seam, then goes through the
# framework's own `invoke()`. Pinned here: the bare run answers deterministically
# (extractive, per-sentence cited — no LLM needed), `config["llm"]` reaches the node
# the graph wires, an explicit `llm_client=` wins over it, and with no LLM the S-3
# regeneration step rejects with the named reason instead of a silent stub rewrite.

import json
import pathlib

import pytest
import yaml

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.nodes.security_gate_output_node import SecurityGateOutputNode
from src.services.service import ConfigLLMAdapter, StubLLMClient, resolve_llm_client

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_QUESTION = "What multi-agent orchestration patterns are available?"
_UNCITED_ANSWER = "Group chat orchestration coordinates several agents through one shared conversation."


def _runner_config() -> dict:
    """The dict the runner passes: config/config.yaml as loaded, nothing added."""
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg, dict) and cfg, "config/config.yaml must load to a non-empty dict"
    return cfg


def _ctx() -> InvocationContext:
    return InvocationContext(caller_id="marketplace-user", caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)


def _invoke(agent, message: str = _QUESTION) -> dict:
    agent.compile()
    return agent.invoke(message, ctx=_ctx(), input_context={"conversation_history": []})


def _is_success(out: dict) -> bool:
    return str(out.get("status", "")).lower().endswith("success")


def _gate_state(hits: int = 1) -> dict:
    """A post_process-shaped state whose answer has one uncited substantive claim."""
    return {"answer": _UNCITED_ANSWER, "retrieval_hit_count": hits, "trace_id": "t", "node_history": []}


class _ScriptedLLM:
    """A config["llm"]-shaped client: `invoke(prompt) -> str`, no `generate`."""

    def __init__(self, reply: str = "SCRIPTED: group chat coordinates agents [ms-af-pat-002].") -> None:
        self.prompts: list[str] = []
        self.reply = reply

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _RaisingLLM:
    def invoke(self, prompt: str) -> str:
        raise RuntimeError("upstream LLM failure")


class TestBareRunnerConstruction:
    """`Graph(config=...)` alone — exactly what the Marketplace runner does."""

    def test_answers_deterministically_without_any_llm(self):
        agent = Graph(config=_runner_config())
        assert agent._llm_client is None, "a bare graph must not bind a stub LLM silently"
        out = _invoke(agent)
        assert _is_success(out), out.get("status")
        assert out.get("output"), "a runner-shaped invocation produced no output"
        assert out.get("error_code") is None, out.get("error_code")
        assert (out.get("retrieval_hit_count") or 0) >= 1
        assert out.get("citation_status") == "passed"
        # The extractive answer is grounded: every sentence carries a [doc_id].
        assert "[maf-" in out["output"]

    def test_wired_gate_has_no_llm_and_rejects_uncited_claims_with_named_reason(self):
        agent = Graph(config=_runner_config())
        agent.compile()
        gate = agent._nodes["post_process"]
        assert isinstance(gate, SecurityGateOutputNode) and gate._llm is None
        out = gate.execute(_gate_state())
        assert out["status"] == "success"
        assert out["error_code"] == "LLM_NOT_CONFIGURED"
        assert out["citation_status"] == "rejected"
        assert out["answer"] and "withholding" in out["answer"], "rejection must publish a non-empty safe refusal"
        assert _UNCITED_ANSWER not in out["answer"], "the uncited claim leaked through"
        assert json.loads(out["uncited_claims"]) == [_UNCITED_ANSWER]


class TestConfigLlmSeam:
    """`config["llm"]` reaches the gate node the graph wires; explicit kw wins over it."""

    def test_full_path_with_config_llm_completes_without_needing_it(self):
        llm = _ScriptedLLM()
        out = _invoke(Graph(config={**_runner_config(), "llm": llm}))
        assert _is_success(out) and out.get("error_code") is None, out
        assert out.get("citation_status") == "passed"
        # Deterministic synthesis: a fully cited answer never triggers regeneration.
        assert llm.prompts == []

    def test_scripted_config_llm_shapes_the_regenerated_answer(self):
        llm = _ScriptedLLM()
        agent = Graph(config={**_runner_config(), "llm": llm})
        agent.compile()
        gate = agent._nodes["post_process"]
        assert isinstance(gate._llm, ConfigLLMAdapter) and gate._llm._client is llm
        out = gate.execute(_gate_state())
        assert len(llm.prompts) == 1, "the config['llm'] client was not called exactly once"
        assert _UNCITED_ANSWER in llm.prompts[0], "the previous answer did not reach the prompt"
        assert out["answer"] == llm.reply, "the answer does not derive from the client's reply"
        assert out["citation_status"] == "regenerated"
        assert out.get("error_code") is None

    def test_explicit_llm_client_wins_over_config_llm(self):
        config_llm = _ScriptedLLM(reply="FROM CONFIG [d1].")
        agent = Graph(
            config={**_runner_config(), "llm": config_llm}, llm_client=StubLLMClient(canned="FROM EXPLICIT KW [d1].")
        )
        agent.compile()
        out = agent._nodes["post_process"].execute(_gate_state())
        assert out["answer"] == "FROM EXPLICIT KW [d1]."
        assert config_llm.prompts == [], "config['llm'] was called although an explicit client was given"

    def test_raising_config_llm_is_not_swallowed(self):
        agent = Graph(config={**_runner_config(), "llm": _RaisingLLM()})
        agent.compile()
        with pytest.raises(RuntimeError, match="upstream LLM failure"):
            agent._nodes["post_process"].execute(_gate_state())

    def test_adapter_prefers_invoke_then_complete_and_coerces_message_content(self):
        class _Msg:
            content = "reply text"

        class _CompleteOnly:
            def complete(self, prompt, **kw):
                return _Msg()

        assert ConfigLLMAdapter(_ScriptedLLM(reply="x")).generate("p") == "x"
        assert ConfigLLMAdapter(_CompleteOnly()).generate("p") == "reply text"
        with pytest.raises(TypeError):
            ConfigLLMAdapter(object())
        with pytest.raises(TypeError):
            Graph(config={**_runner_config(), "llm": object()})
        assert resolve_llm_client(None, {"llm": None}) is None
        assert resolve_llm_client(None, None) is None
        stub = StubLLMClient()
        assert resolve_llm_client(None, {"llm": stub}) is stub  # already speaks generate()
        assert resolve_llm_client(stub, {"llm": _ScriptedLLM()}) is stub
