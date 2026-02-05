from __future__ import annotations
from typing import Any, Callable

from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    VECTORSTORE_DIR, CI_VECTORSTORE_DIR,
    RETRIEVAL_CONFIG, AMBIGUITY_CONFIG,
    EMBEDDING_CONFIG, LLM_CONFIG, PROMPT_CONFIG,
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
from src.rag.entity_extract import EntityExtractor
from src.rag.ood import OODConfig, ood_gate
from src.rag.coverage import CoverageConfig, coverage_gate
from src.schemas import RetrievalState, RetrievalStatusEnum
from src.utils.diagnostics import build_debug_logger

_dbg = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.chain",
    key="print_refusal_reason"
)

def _is_status(
    state: RetrievalState,
    value: str
)-> bool:
    return str(state.get("status") or "").lower() == str(value).lower()

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

def _format_ambiguous_answer(
    state: RetrievalState,
    *, preview_n_sources: int=3
) -> str:
    options = state.get("options") or []
    if not options:
        return "Ambiguous, but no options were produced."

    lines: list[str] = []
    lines.append("Ambiguous Options:")

    for option in options:
        opt_id = getattr(option, "option_id", None)
        sources = getattr(option, "sources", None) or []
        best_l2 = getattr(option, "best_l2", None)

        preview_sources = sources[:preview_n_sources]
        src_txt = ", ".join(
            f"{getattr(src, 'filename', 'unknown')}:{getattr(src, 'page', '?')}"
            for src in preview_sources
        )
        more = f" (+{len(sources) - len(preview_sources)} more)" if len(sources) > len(preview_sources) else ""
        l2_txt = f" best_l2={best_l2:.4f}" if isinstance(best_l2, (int, float)) else ""

        lines.append(f"- option_id={opt_id}{l2_txt}: {src_txt}{more}")

    return "\n".join(lines)

def _format_final_output(state: RetrievalState) -> dict[str, Any]:
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

def _answer_or_refuse(
    state: RetrievalState,
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

def _guard_retrieval(
    state: RetrievalState,
    *, retrieve_and_gate: Callable[[RetrievalState], RetrievalState]
) -> RetrievalState:
    if state.get("skip_llm", False):
        return state

    return retrieve_and_gate(state)

def _guard_coverage_gate(
    state: RetrievalState,
    *, cfg: CoverageConfig
) -> RetrievalState:
    if state.get("skip_llm", False):
        return state

    if not _is_status(state, RetrievalStatusEnum.OK.value):
        return state

    return coverage_gate(state, cfg=cfg)

def _guard_refuse_if_no_docs(state: RetrievalState) -> RetrievalState:
    if state.get("skip_llm", False):
        return state

    if _is_status(state, RetrievalStatusEnum.AMBIGUOUS.value):
        return state

    if _is_status(state, RetrievalStatusEnum.OK.value):
        return refuse_if_no_docs(state)

    return state

def _short_circuit_ambiguous(
    state: RetrievalState,
    *, preview_n_sources: int=3
) -> RetrievalState:
    if state.get("skip_llm", False):
        return state

    if _is_status(state, RetrievalStatusEnum.AMBIGUOUS.value):
        return {
            **state,
            "docs": [],
            "skip_llm": True,
            "refusal_reason": None,
            "answer": _format_ambiguous_answer(state, preview_n_sources=preview_n_sources),
        }

    return state

def _normalize_state(state: Any) -> RetrievalState:
    if isinstance(state, dict):
        text = state.get("input") or state.get("query") or state.get("question")
        return {
            **state,
            "input": str(text) if text is not None else ""
        }
    return {"input": str(state)}

def build_rag_chain(is_ci: bool=False) -> Runnable:
    max_chars = int(PROMPT_CONFIG["max_chars_per_chunk"])
    gate_cfg = GateConfig.from_dict(RETRIEVAL_CONFIG)
    ood_cfg = OODConfig.from_dict(RETRIEVAL_CONFIG.get("ood"))
    cvg_cfg = CoverageConfig.from_dict(RETRIEVAL_CONFIG.get("coverage"))
    entity_extractor = EntityExtractor(entity_patterns=cvg_cfg.entity_patterns)
    max_options=int(RETRIEVAL_CONFIG["max_options"])
    
    fetch_k = _calculate_safe_fetch_k(
        fetch_k=int(RETRIEVAL_CONFIG["fetch_k"]),
        final_k=gate_cfg.final_k,
        max_options=max_options
    )
    
    embeddings = build_embeddings(EMBEDDING_CONFIG)

    vs_dir = str(VECTORSTORE_DIR) if not is_ci else str(CI_VECTORSTORE_DIR)
    _dbg(f"Loading vectorstore from: {vs_dir}")
    vectorstore = FAISS.load_local(
        vs_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    ambiguity_cfg = AmbiguityConfig.from_dict(
        RETRIEVAL_CONFIG,
        AMBIGUITY_CONFIG,
        embed_docs=embeddings.embed_documents,
        entity_extractor=entity_extractor
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
        # Check if short-circuits to ambiguous-state
        | RunnableLambda(lambda s: _short_circuit_ambiguous(s, preview_n_sources=3))
        # Coverage gate for ok-state only
        | RunnableLambda(lambda s: _guard_coverage_gate(s, cfg=cvg_cfg))
        # Decide refuse early for ok-state only
        | RunnableLambda(_guard_refuse_if_no_docs)        
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
