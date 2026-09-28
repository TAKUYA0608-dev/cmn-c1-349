"""CMN-C1-349 — service layer: orchestration KB retrieval + LLM boundaries (DI).

The two external services (the orchestration knowledge-base retriever, and the
LLM) are injected so the template is offline-testable and deployment-agnostic.
Production binds the real vector store + platform LLM client (via ctx.secrets);
tests bind the in-memory / stub implementations defined here.

Microsoft agent-framework facts (orchestration patterns, version notes, Japan
enterprise integration guidance) are KB *data*, never hard-coded here — see
docs/02_design.md §1.2. No ``agenticstar`` (Level 0) imports; no ``framework.*``
dependency — pure domain logic.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

# The framework this template provides Q&A over (docs/02_design.md §1.1).
FRAMEWORK: str = "microsoft/agent-framework"

# Orchestration topic taxonomy (docs/02_design.md §1.2). The classifier maps a
# question to exactly one of these; "out-of-scope" is the safe default when no
# topic signal matches (0-hit refusal downstream).
ORCHESTRATION_TOPICS: tuple[str, ...] = (
    "multi-agent-patterns",  # sequential / concurrent / group-chat / handoff / magentic
    "task-decomposition",  # planner / sub-task breakdown
    "agent-coordination",  # shared state, message passing, termination
    "enterprise-integration",  # SAP / ServiceNow / Japan enterprise systems
    "governance-delegation",  # Japan AI Regulation Bill 2026 agent task-delegation boundaries
    "framework-architecture",  # framework runtime, abstractions, Python/.NET surface
    "out-of-scope",  # default: no orchestration topic matched
)

# Confidence below which the topic is treated as uncertain (still classified, but
# the value is surfaced so callers can gauge reliability).
CONFIDENCE_THRESHOLD: float = 0.6


# ★ 日本語 KB × 語彙検索の構造欠陥 (本番 Pod で実証):
# str.split() は空白区切りなので日本語の文が 1 トークンに潰れ、日本語質問は
# 何も引けない (349 の実測: 日本語 4 問すべて 0 hit)。CJK 文字 bigram +
# 機能語除去で解決する。実装は 328 の service.lexical_terms を移植。
_CJK_RE = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟ]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")

# Function words carry no topical signal but are dense in prose, so they let an
# out-of-scope question score highly on any chunk. Measured before removal:
# "What is the weather in Tokyo today?" scored 0.429 on a citations chunk
# (matching only "the"/"is"/"in") — higher than a correct Japanese retry
# question at 0.172, so no score threshold could separate them.
_STOPWORDS = frozenset(
    """
a an the and or but if then than that this these those of in on at to for from by with
without about into over under as is are was were be been being do does did doing have
has had having i you he she it we they me my your his her its our their what which who
whom when where why how should would could can may might must will shall not no yes so
such only own same too very just also there here up down out off again further once
""".split()
)


# Japanese function-word bigrams. Bigram indexing makes a short, function-word
# query dangerous rather than merely imprecise: "ある" yields the single term
# {ある}, and any chunk containing it scores 1.0 — no threshold can separate
# that from a real hit, and the weak match is then cited. Dropping these before
# scoring makes such a query yield no terms at all, which is the correct answer
# (the query carries no topical content). Segmenting properly would need a
# morphological analyser; this list covers the high-frequency verb, copula,
# demonstrative, formal-noun and particle forms that reach 1.0 on their own.
_JA_FUNCTION_BIGRAMS = frozenset(
    """
する すれ すな すべ して した しな しま しょ され せる れる られ でき きる こう
ある あり あっ ない なく なし いる いた いま なる なっ なり なら れば
これ それ あれ どれ この その あの どの どう そう ああ いう
れは れを れが れに れで はこ はそ はど はな はで はと
こと もの ため とき よう ところ ばあ あい
です ます ませ まし だっ であ でし でも ても とも
から まで より ので のに には では との への とし につ いて ついて ため
がで がい がな をど をす をし にす にお おけ ける
うす うか うし うも いか いき かた たら
るこ るの るか ると るが るを るに るで るは るも るま
たこ たの たか たと たが たを たに
のこ のか のと のが のを のに のは のも
なの なか なと なが なに ので うな あな
""".split()
)


def lexical_terms(text: str) -> set[str]:
    """Topical lexical terms: content words plus CJK character bigrams.

    Punctuation is stripped so "retrieval?" and "retrieval" are the same term,
    and function words are dropped in both scripts (`_STOPWORDS`,
    `_JA_FUNCTION_BIGRAMS`). A query left with no terms retrieves nothing,
    which is the intended outcome for a query with no topical content.
    """
    terms = {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS}
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            terms.add(run)
        else:
            terms |= {run[i : i + 2] for i in range(len(run) - 1)} - _JA_FUNCTION_BIGRAMS
    return terms


@runtime_checkable
class KBRetrievalBackend(Protocol):
    """Orchestration KB retriever boundary.

    ``search`` returns a ranked list of KB record dicts, each shaped:
        {"doc_id": str, "title": str, "text": str, "score": float,
         "source": str, "framework_version": str}
    ``topic``, when given, filters / biases retrieval to that orchestration topic.
    """

    def search(self, query: str, top_k: int = 6, topic: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class LLMClient(Protocol):
    """LLM boundary — ``generate(prompt) -> str``."""

    def generate(self, prompt: str) -> str: ...


class InMemoryKBBackend:
    """In-memory KBRetrievalBackend for tests / local runs.

    Seed with ``add([record, ...])``; ``search`` ranks by naive lexical overlap
    and, when a ``topic`` is given (and not "out-of-scope"), biases to records
    tagged with that topic (via the record ``source`` / a ``topic`` field).
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def add(self, records: list[dict[str, Any]]) -> None:
        self._records.extend(records)

    # Tokenisation / stopword removal is delegated to module-level
    # ``lexical_terms`` (content words + CJK bigrams + function-word removal in
    # both scripts) so a question never matches a doc purely on filler words —
    # and so Japanese questions actually tokenise (the real vector store does
    # this via embeddings; the in-memory backend approximates it).
    def search(self, query: str, top_k: int = 6, topic: str | None = None) -> list[dict[str, Any]]:
        terms = lexical_terms(query)

        def score(r: dict[str, Any]) -> float:
            words = lexical_terms(str(r.get("text", ""))) | lexical_terms(str(r.get("title", "")))
            base = (len(terms & words) / len(terms)) if terms and words else 0.0
            # Topic bias is a re-rank BOOST for docs that already overlap the query —
            # it must not turn a zero-overlap doc into a hit (else a content-free
            # question would match purely on its topic tag).
            if base > 0.0 and topic and topic not in ("out-of-scope", None) and r.get("topic") == topic:
                base += 0.15
            return round(base, 4)

        ranked = sorted(
            ({**r, "score": score(r)} for r in self._records),
            key=lambda r: r["score"],
            reverse=True,
        )
        return [r for r in ranked if r["score"] > 0.0][:top_k]


class StubLLMClient:
    """Deterministic stub LLM — returns a templated, citation-bearing answer.

    Tests / local runs only: it is NEVER bound by default. The graph resolves the
    LLM as explicit ``llm_client`` > ``config["llm"]`` > none (``resolve_llm_client``);
    with none, the citation gate cannot regenerate and rejects with a named reason.
    """

    def __init__(self, canned: str | None = None) -> None:
        self._canned = canned

    def generate(self, prompt: str) -> str:
        if self._canned is not None:
            return self._canned
        tail = prompt.strip().splitlines()[-1][:200] if prompt.strip() else ""
        return f"Based on the cited framework docs: {tail} [ms-af-orchestration-001]"


# ── LLM seam: config["llm"] → LLMClient ───────────────────────────────────────
def _as_text(result: Any) -> str:
    """Coerce a client reply to text: str as-is, message-like objects via ``.content``."""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return "" if result is None else str(result)


class ConfigLLMAdapter:
    """Adapt a ``config["llm"]`` object to this template's ``LLMClient`` Protocol.

    The fleet entry point places a lazily-resolved chat client under ``config["llm"]``
    that answers ``invoke(prompt) -> str`` (and ``complete(prompt, **kw) -> str``);
    this template's nodes speak ``generate(prompt) -> str``. The adapter forwards to
    ``invoke`` first, then ``complete``, and never swallows the client's exceptions —
    a configured but failing LLM must surface, not silently degrade.
    """

    def __init__(self, client: Any) -> None:
        call = getattr(client, "invoke", None)
        if not callable(call):
            call = getattr(client, "complete", None)
        if not callable(call):
            raise TypeError(
                "config['llm'] must expose generate(prompt), invoke(prompt) or "
                f"complete(prompt); got {type(client).__name__}"
            )
        self._client = client
        self._call = call

    def generate(self, prompt: str) -> str:
        return _as_text(self._call(prompt))


def resolve_llm_client(explicit: LLMClient | None, config: Any) -> LLMClient | None:
    """Precedence: explicit ``llm_client`` kw > ``config["llm"]`` > None.

    None means "no LLM bound". Answer synthesis in this template is deterministic
    (extractive, per-sentence cited — it needs no LLM), so a bare run still answers;
    only the S-3 regeneration step is affected: with no LLM it rejects with
    ``LLM_NOT_CONFIGURED`` instead of rewriting. An object under ``config["llm"]``
    that answers none of generate/invoke/complete raises at construction
    (misconfiguration is not a reason to degrade quietly).
    """
    if explicit is not None:
        return explicit
    candidate = config.get("llm") if isinstance(config, dict) else None
    if candidate is None:
        return None
    if isinstance(candidate, LLMClient):
        return candidate
    return ConfigLLMAdapter(candidate)


def seed_orchestration_kb() -> InMemoryKBBackend:
    """Return an InMemoryKBBackend seeded with representative orchestration docs.

    A compact, illustrative slice (EN + JP keyword tags) — enough for a bare
    ``Graph()`` (the form both server.py and the Marketplace runner construct) to
    answer instead of degrading every run to RETRIEVAL_BACKEND_MISSING.
    Framework facts remain KB *data* (docs/02_design.md §1.2): production
    replaces this with the full curated/indexed KB; the record shape is identical.
    """
    backend = InMemoryKBBackend()
    backend.add(
        [
            {
                "doc_id": "maf-patterns-001",
                "title": "Multi-agent orchestration patterns",
                "text": "multi-agent orchestration patterns sequential concurrent group-chat handoff magentic "
                "マルチエージェント オーケストレーション パターン",
                "score": 0.0,
                "source": "seed-kb",
                "framework_version": "1.0.0",
                "topic": "multi-agent-patterns",
            },
            {
                "doc_id": "maf-decomp-001",
                "title": "Task decomposition with a planner",
                "text": "task decomposition planner sub-task breakdown workflow design step sequencing "
                "タスク分解 プランナー ワークフロー設計",
                "score": 0.0,
                "source": "seed-kb",
                "framework_version": "1.0.0",
                "topic": "task-decomposition",
            },
            {
                "doc_id": "maf-coord-001",
                "title": "Agent coordination and shared state",
                "text": "agent coordination shared state message passing termination condition consensus "
                "協調 共有状態 メッセージパッシング 終了条件",
                "score": 0.0,
                "source": "seed-kb",
                "framework_version": "1.0.0",
                "topic": "agent-coordination",
            },
            {
                "doc_id": "maf-arch-001",
                "title": "Framework runtime and abstractions",
                "text": "framework architecture runtime abstractions python sdk dotnet api surface "
                "アーキテクチャ ランタイム 抽象化",
                "score": 0.0,
                "source": "seed-kb",
                "framework_version": "1.0.0",
                "topic": "framework-architecture",
            },
            {
                "doc_id": "maf-gov-001",
                "title": "Governance boundaries for agent task delegation",
                "text": "governance delegation boundaries japan ai regulation compliance audit trail accountability "
                "ガバナンス 権限委譲 統制 監査証跡",
                "score": 0.0,
                "source": "seed-kb",
                "framework_version": "1.0.0",
                "topic": "governance-delegation",
            },
        ]
    )
    return backend
