from __future__ import annotations
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import DEBUG_CONFIG
from src.schemas import ScoredDocument, RetrievalState, RetrievalOption, RetrievalStatusEnum
from src.rag.formatting import collect_sources, normalize_page
from src.rag.gating import GateConfig, gate_scored_docs_l2
from src.utils.diagnostics import build_debug_logger

_dbg = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.retriever",
    key="print_ambiguous_options"
)

def _ok_state(
    state: RetrievalState,
    *, docs: list[Document]
) -> RetrievalState:
    return {
        **state,
        "docs": docs,
        "status": RetrievalStatusEnum.OK.value,
        "options": [],
        "selected_option": None,
        "refusal_reason": None
    }

def _refuse_state(
    state: RetrievalState,
    *, reason: str
) -> RetrievalState:
    return {
        **state,
        "docs": [],
        "status": RetrievalStatusEnum.REFUSE.value,
        "options": [],
        "selected_option": None,
        "refusal_reason": reason
    }
    
def _ambiguous_state(
    state: RetrievalState,
    *, options: list[RetrievalOption]
) -> RetrievalState:
    return {
        **state,
        "docs": [],
        "status": RetrievalStatusEnum.AMBIGUOUS.value,
        "options": options,
        "selected_option": None,
        "refusal_reason": None
    }

def _is_from_same_file(
    doc_a: Document,
    doc_b: Document
) -> bool:
    source_a = (doc_a.metadata or {}).get("source")
    source_b = (doc_b.metadata or {}).get("source")
    
    if not source_a or not source_b:
        return False
    
    return Path(source_a).name == Path(source_b).name

def _option_signature(option: RetrievalOption) -> tuple[tuple[str, str], ...]:
    signature = sorted((src.filename, str(src.page)) for src in option.sources)
    
    return tuple(signature)

def _deduplicate_options(options: list[RetrievalOption]) -> list[RetrievalOption]:
    seen: set[tuple[tuple[str , str], ...]] = set()
    outs: list[RetrievalOption] = []
    
    for option in options:
        signature = _option_signature(option)
        if signature in seen:
            continue
        seen.add(signature)
        outs.append(option)

    return [
        RetrievalOption(
            option_id=i+1,
            docs=opt.docs,
            sources=opt.sources,
            best_l2=opt.best_l2
        )
        for i, opt in enumerate(outs)
    ]

def _get_source_info(doc: Document) -> tuple[str, str]:
    meta = doc.metadata or {}
    filename = Path(meta.get("source", "unknown")).name
    page = normalize_page(meta.get("page"))  
    
    return filename, str(page)

def _doc_signature(doc: Document) -> tuple[str, str]:
    filename, page = _get_source_info(doc)
    return filename, page

def _prioritize_documents_for_anchor(
    *,
    anchor: Document,
    scored: list[ScoredDocument]
) -> list[Document]:
    """
    Form a document list with below order:
    1. same file
    2. other files
    """
    same_file_docs: list[Document] = []
    other_docs: list[Document] = []
    
    for sd in scored:
        doc = sd.doc
        if doc is anchor:
            continue
        # Fill into same_file_docs or other_docs
        if _is_from_same_file(anchor, doc):
            same_file_docs.append(doc)
        else:
            other_docs.append(doc)
    
    return same_file_docs + other_docs

def _select_distinct_docs(
    *,
    anchor: Document,
    candidates: list[Document],
    need: int
) -> list[Document]:
    if need <= 0:
        return []
    
    picked: list[Document] = []
    filename, page = _get_source_info(anchor)
    seen_files: set[str] = {filename}
    seen_pages: set[str] = {page}
    seen_signatures: set[tuple[str, str]] = {_doc_signature(anchor)}

    phases = 3
    for phase in range(phases):
        for candidate in candidates:
            if len(picked) >= need:
                return picked
            sig = _doc_signature(candidate)
            if sig in seen_signatures:
                continue
            
            filename, page = _get_source_info(candidate)
            should_pick = False
            
            match phase:
                case 0: should_pick = (page not in seen_pages)
                case 1: should_pick = (filename not in seen_files)
                case 2: should_pick = True
                
            if should_pick:
                picked.append(candidate)
                seen_signatures.add(sig)
                seen_files.add(filename)
                seen_pages.add(page)
                
    return picked

def _build_ambiguous_options(
    scored: list[ScoredDocument],
    *,
    final_k: int,
    max_options: int=3
) -> list[RetrievalOption]:
    if not scored:
        return []
    
    effective_k = max(1, final_k)       # final_k >= 1
    safe_k = max(0, effective_k - 1)    # Control docs qty included
    options: list[RetrievalOption] = []
    
    n = min(max_options, len(scored))
    
    for rank in range(n):
        anchor = scored[rank].doc
        candidates = _prioritize_documents_for_anchor(
            anchor=anchor,
            scored=scored
        )
        
        picked = _select_distinct_docs(
            anchor=anchor,
            candidates=candidates,
            need=safe_k
        )

        docs = [anchor] + picked
        sources = collect_sources(docs)
        
        options.append(
            RetrievalOption(
                option_id=rank + 1,
                docs=docs,
                sources=sources,
                best_l2=float(scored[rank].score)
            )
        )
    
    return options

def fetch_scored_docs_l2(
    *,
    vectorstore: FAISS,
    query: str,
    fetch_k: int
) -> list[ScoredDocument]:
    """
    Fetch docs with L2 distance scores from FAISS with quantity limited by fetch_k.
    Returns list sorted by ascending score (best first).
    """
    pairs: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        query=query,
        k=fetch_k
    )
    scored = [ScoredDocument(doc=d, score=float(s)) for (d, s) in pairs]
    scored.sort(key=lambda x: x.score)
    
    return scored

def build_retrieve_and_gate_l2(
    *,
    gate_cfg: GateConfig,
    vectorstore: FAISS,
    fetch_k: int,
    max_options: int=3
) -> Callable[[RetrievalState], RetrievalState]:
    def _step(state: RetrievalState) -> RetrievalState:
        selected = state.get("selected_option")
        options = state.get("options")
        
        if selected is not None and options:
            chosen = next((option for option in options if option.option_id == selected), None)
            if chosen is None:
                return _refuse_state(
                    state,
                    reason=f"Invalid selection: {selected}"
                )
            
            return _ok_state(
                state,
                docs=chosen.docs
            )
     
        query = str(state.get("input", "")).strip()
        scored = fetch_scored_docs_l2(
            vectorstore=vectorstore,
            query=query,
            fetch_k=fetch_k,
        )
        docs, status = gate_scored_docs_l2(
            scored,
            cfg=gate_cfg
        )

        match status:
            case RetrievalStatusEnum.AMBIGUOUS.value:
                raw_options = _build_ambiguous_options(
                    scored,
                    final_k=gate_cfg.final_k,
                    max_options=max_options
                )
                options = _deduplicate_options(raw_options)
                _dbg(f"Ambiguous: Raw options={len(raw_options)}, Deduplicated options={len(options)}")
                if len(options) == 1:
                    chosen = options[0]
                    return _ok_state(
                        state,
                        docs=chosen.docs
                    )
                
                if len(options) == 0:
                    return _refuse_state(
                        state,
                        reason="Ambiguous gate produced no valid options"
                    )
                
                return _ambiguous_state(
                    state,
                    options=options
                )
            
            case RetrievalStatusEnum.REFUSE.value:
                return _refuse_state(
                    state,
                    reason="No relevant documents found"
                )
            
            case RetrievalStatusEnum.OK.value:
                if not docs:
                    return _refuse_state(
                        state,
                        reason="OK status but empty documents (unexpected)"
                    )
       
                return _ok_state(
                    state,
                    docs=docs
                )
            
            case _:
                return _refuse_state(
                    state,
                    reason="Unknown status (unexpected)"
                )                

    return _step
