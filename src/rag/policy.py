from __future__ import annotations
from collections.abc import Sequence

from langchain_core.documents import Document

from src.schemas import RetrievalState, RetrievalStatusEnum

REFUSAL_TEXT = "I don't know based on the provided documents."

def refuse_if_no_docs(state: RetrievalState) -> RetrievalState:
    """
    Ensure the state has valid docs; otherwise short-circuit the chain.
    """
    docs = state.get("docs", [])
    status = state.get("status")
    options = state.get("options")

    refuse_state = {
        **state,
        "docs": [],
        "context": "",
        "answer": REFUSAL_TEXT,
        "skip_llm": True,
        "refusal_reason": "No relevant documents found"
    }

    match status:
        case RetrievalStatusEnum.OK.value:
            if not isinstance(docs, Sequence):
                reason =  "Invalid document format"
            elif len(docs) == 0:
                reason = "No relevant documents found"
            elif not all(isinstance(d, Document) for d in docs):
                reason = "Document integrity error"
            else:
                return {**state, "skip_llm": False, "refusal_reason": None}
            
            return {**refuse_state, "refusal_reason": reason}
        case RetrievalStatusEnum.REFUSE.value:
            return refuse_state
        
        case RetrievalStatusEnum.AMBIGUOUS.value:
            if options:
                return {
                    **state,
                    "skip_llm": True,
                    "answer": "Ambiguous retrieval. Please choose an option number (1..N) to continue.",
                    "refusal_reason": "User selection required"
                }
            
            return refuse_state
        
        case _:
            return {**refuse_state, "refusal_reason": "Unknown retrieval state"}
