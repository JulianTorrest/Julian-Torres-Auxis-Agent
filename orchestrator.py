import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from pii_guard import redact_pii
from prompt_guard import is_injection
from governance import check_policy
from guardian_kit import InputGuardian, output_guard
from llm_clients import MultiLLM
from observability_manager import ObservabilityManager
from semantic_cache import SemanticCache
from context_manager import ContextManager
from memory import MemoryManager
from multi_store import Embedder

class AgentState(TypedDict):
    query: str
    user: str
    start_time: float
    mode: str
    selected_provider: str
    max_retries: int
    redacted: str
    pii_found: List[str]
    injection: bool
    policy_ok: bool
    violations: List[str]
    input_guard: dict
    output_guard: List[str]
    context: List[str]
    answer: str
    answer_provider: str
    latency_ms: float
    trace_id: str
    tokens_input: int
    tokens_output: int
    cache_hit: str
    tool_used: str
    moderation_flags: List[str]
    rejected: bool
    rejection_reason: str
    topic_changed: bool

def create_workflow(rag, multi_llm: MultiLLM, tracer: ObservabilityManager = None,
                    token_budget: int = 2000, session_id: str = "default"):
    tracer = tracer or ObservabilityManager()
    cache = SemanticCache()
    ctx = ContextManager(llm=multi_llm, token_budget=token_budget)
    mem = MemoryManager()
    embedder = Embedder()
    if rag.nodes:
        embedder.fit([n["text"] for n in rag.nodes])

    def sanitize(state: AgentState):
        redacted, found = redact_pii(state["query"])
        return {"redacted": redacted, "pii_found": found}

    def guardrails(state: AgentState):
        injection = is_injection(state["redacted"])
        ok, violations = check_policy(state["redacted"], bool(state["pii_found"]), injection)
        guardian = InputGuardian(max_requests=1000, window_seconds=60)
        guard = guardian.check(state["user"], state["redacted"])
        topic_changed = ctx.topic_changed(state["redacted"])
        all_ok = ok and guard["allowed"] and not topic_changed
        all_violations = violations + guard["reasons"]
        if topic_changed:
            all_violations.append("cambio de tema detectado")
        return {
            "injection": injection,
            "policy_ok": all_ok,
            "violations": all_violations,
            "input_guard": guard,
            "topic_changed": topic_changed,
            "rejected": not all_ok,
            "rejection_reason": "; ".join(all_violations) if not all_ok else "",
            "moderation_flags": guard.get("moderation", []),
            "tool_used": "InputGuardian",
        }

    def retrieve(state: AgentState):
        if not state["policy_ok"]:
            return {"context": []}
        context = rag.retrieve(state["redacted"])
        ctx.add("user", state["redacted"], tokens=len(state["redacted"].split()))
        return {"context": context}

    def generate(state: AgentState):
        if not state["policy_ok"]:
            reason = state.get("rejection_reason", "motivo no especificado")
            return {
                "answer": f"Solicitud bloqueada por violaciones de gobernanza: {reason}.",
                "answer_provider": "guardian",
                "tokens_input": 0,
                "tokens_output": 0,
                "output_guard": [],
                "cache_hit": "n/a",
            }
        cached = cache.get_exact(state["redacted"], cache_type="general") or cache.get_semantic(
            state["redacted"], embedder.embed, cache_type="general", threshold=0.92)
        if cached:
            return {
                "answer": cached["answer"],
                "answer_provider": cached["provider"],
                "tokens_input": 0,
                "tokens_output": 0,
                "cache_hit": cached["hit"],
                "output_guard": [],
            }
        prompt = ctx.build_prompt(state["redacted"], state["context"])
        provider, answer = multi_llm.generate(
            prompt,
            mode=state["mode"],
            router_provider=state.get("selected_provider"),
            max_retries=state.get("max_retries", 1),
        )
        out_guard = output_guard(answer)
        tokens_in = len(prompt.split())
        tokens_out = len(answer.split())
        cache.put(state["redacted"], answer, provider, embedder.embed, cache_type="general", ttl=300, alert=bool(out_guard))
        mem.save("session", "last", {"query": state["redacted"], "answer": answer, "provider": provider}, session_id=session_id)
        ctx.add("assistant", answer, tokens=tokens_out)
        return {
            "answer": answer,
            "answer_provider": provider,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "cache_hit": "miss",
            "output_guard": out_guard,
            "moderation_flags": list(set(state.get("moderation_flags", []) + out_guard)),
            "rejected": bool(out_guard),
            "rejection_reason": f"output guard: {', '.join(out_guard)}" if out_guard else "",
        }

    def trace(state: AgentState):
        latency = (time.time() - state["start_time"]) * 1000
        record = tracer.log(
            state,
            state["user"],
            latency,
            tokens_input=state.get("tokens_input", 0),
            tokens_output=state.get("tokens_output", 0),
            rejected=state.get("rejected", False),
            rejection_reason=state.get("rejection_reason", ""),
            tool_used=state.get("tool_used", ""),
            moderation_flags=state.get("moderation_flags", []),
        )
        return {"latency_ms": latency, "trace_id": record["id"]}

    builder = StateGraph(AgentState)
    builder.add_node("sanitize", sanitize)
    builder.add_node("guardrails", guardrails)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("trace", trace)
    builder.set_entry_point("sanitize")
    builder.add_edge("sanitize", "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        lambda s: "retrieve" if s["policy_ok"] else "generate",
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "trace")
    builder.add_edge("trace", END)
    return builder.compile()
