from __future__ import annotations
from typing import Any, Callable

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    VECTORSTORE_DIR,
    RETRIEVAL_CONFIG, EMBEDDING_CONFIG, LLM_CONFIG, PROMPT_CONFIG,
    DEBUG_CONFIG
)
from src.rag.embeddings import build_embeddings
from src.rag.llms import build_llm
from src.rag.prompts import RAG_PROMPT
from src.rag.formatting import format_docs_for_prompt, normalize_answer_for_cli
from src.rag.gating import GateConfig
from src.rag.retriever import build_retrieve_and_gate_l2
from src.rag.policy import refuse_if_no_docs, REFUSAL_TEXT
from src.rag.ambiguity import AmbiguityConfig
from src.rag.ood import OODConfig, ood_gate
from src.rag.coverage import CoverageConfig, coverage_gate
from src.schemas import RetrievalState
from src.utils.diagnostics import build_debug_logger

_dbg = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.chain",
    key="print_refusal_reason"
)

def _calculate_safe_fetch_k(
    *,
    fetch_k: int,
    final_k: int,
    max_options: int,
    buffer: int=2
) -> int:
    if final_k < 1:
        raise ValueError(f"final_k must be at least 1, got {final_k}")
    
    safe_fetch_k = (final_k + 2*max_options) + buffer
    return max(safe_fetch_k, fetch_k)

def _answer_or_refuse(
    state: dict[str, Any],
    *, llm: Runnable
) -> str:
    if state.get("skip_llm", False):
        _dbg(f"refusal_reason = {state.get('refusal_reason')}")
        return normalize_answer_for_cli(state.get("answer", REFUSAL_TEXT))

    try:
        raw = (RAG_PROMPT | llm | StrOutputParser()).invoke({
            "input": state.get("input", ""),
            "context": state.get("context", "")
        })
        return normalize_answer_for_cli(str(raw))
    except Exception as e:
        return f"LLM backend error. {type(e).__name__}."

def _format_final_output(state: dict[str, Any]) -> dict[str, Any]:
    is_refusal = state.get("skip_llm", False)

    return {
        "input": state.get("input", ""),
        "source_documents": [] if is_refusal else state.get("docs", []),
        "answer": state.get("answer", ""),
        "status": state.get("status", None),
        "refusal_reason": state.get("refusal_reason", None),
        "options": state.get("options", None),
        "selected_option": state.get("selected_option", None)
    }

def _guard_retrieval(
    state: dict[str, Any],
    *, retrieve_and_gate: Callable[[RetrievalState], RetrievalState]
) -> dict[str, Any]:
    if state.get("skip_llm", False):
        return state

    return retrieve_and_gate(state)

def _normalize_state(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        text = state.get("input") or state.get("query") or state.get("question")
        return {
            **state,
            "input": str(text) if text is not None else ""
        }
    return {"input": str(state)}

def build_rag_chain() -> Runnable:
    max_chars = int(PROMPT_CONFIG["max_chars_per_chunk"])
    gate_cfg = GateConfig.from_dict(RETRIEVAL_CONFIG)
    ood_cfg = OODConfig.from_dict(RETRIEVAL_CONFIG.get("ood"))
    cvg_cfg = CoverageConfig.from_dict(RETRIEVAL_CONFIG.get("coverage"))
    max_options=int(RETRIEVAL_CONFIG["max_options"])
    
    fetch_k = _calculate_safe_fetch_k(
        fetch_k=int(RETRIEVAL_CONFIG["fetch_k"]),
        final_k=gate_cfg.final_k,
        max_options=max_options
    )
    
    embeddings = build_embeddings(EMBEDDING_CONFIG)

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    ambiguity_cfg = AmbiguityConfig.from_dict(
        RETRIEVAL_CONFIG,
        embed_docs=embeddings.embed_documents
    )

    retrieve_and_gate = build_retrieve_and_gate_l2(
        gate_cfg=gate_cfg,
        vectorstore=vectorstore,
        fetch_k=fetch_k,
        ambiguity_cfg=ambiguity_cfg
    )

    llm = build_llm(LLM_CONFIG)

    chain = (
        # Guardrail for dict-shape input
        RunnableLambda(_normalize_state)
        # OOD gate before retrieval
        | RunnableLambda(lambda s: ood_gate(s, cfg=ood_cfg))
        # Retrieve docs or short-circuits on refusal
        | RunnableLambda(lambda s: _guard_retrieval(s, retrieve_and_gate=retrieve_and_gate))
        # Coverage gate
        | RunnableLambda(lambda s: coverage_gate(s, cfg=cvg_cfg))
        # Decide refuse early
        | RunnableLambda(refuse_if_no_docs)        
        # Format docs into context string
        | RunnablePassthrough.assign(
            context=lambda x: format_docs_for_prompt(
                x.get("docs", []),
                max_chars_per_chunk=max_chars
            )
        )
        # Compute answer
        | RunnablePassthrough.assign(
            answer=RunnableLambda(lambda x: _answer_or_refuse(x, llm=llm))
        )
        # Final putput
        | RunnableLambda(_format_final_output)
    )
    
    return chain
