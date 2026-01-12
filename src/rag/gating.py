from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import DEBUG_CONFIG
from src.schemas import ScoredDocument, RetrievalStatus, RetrievalStatusEnum
from src.utils.diagnostics import build_debug_logger

_dbg_abs = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.gating",
    key="print_absolute"
)

_dbg_gap = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.gating",
    key="print_gap"
)

_dbg_den = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.gating",
    key="print_density"
)

@dataclass(frozen=True)
class GateConfig:
    final_k: int
    max_l2: float
    min_keep: int
    min_gap: float | None = None

def _validate_absolute_gate(
    scored: list[ScoredDocument],
    *, max_l2: float
) -> bool:
    """
    Validate that the best match is within the allowed distance threshold.
    Returns:
        True: If the best match's score is <= max_l2.
        False: If the list is empty or the best match is too distant (irrelevant).
    Remarks:
        Expects 'scored' to be pre-sorted (Achieved in fetch_scored_docs_l2).
    """
    if not scored:
        _dbg_abs("Blocked: No scored documents")
        return False
    
    validate = scored[0].score <= max_l2
    
    validate_str = "Passed" if validate else "Blocked"
    _dbg_abs(f"{validate_str}: score={scored[0].score}, max_l2={max_l2}")
    
    return validate

def _validate_confidence_gap_gate(
    scored: list[ScoredDocument],
    *, min_gap: float | None=None
) -> bool:
    """
    Validates that the best match is significantly better than the runner-up (2nd).
    Returns:
        True: If the gap is sufficient (>= min_gap) or disabled (None)
        False: If the gap is too small (ambiguous) or < 2 documents exist.
    Remarks:
        Expects 'scored' to be pre-sorted (Achieved in fetch_scored_docs_l2).
    """
    if min_gap is None:
        _dbg_gap("Skipped: min_gap is disabled")
        return True
    
    if not scored:
        _dbg_gap("Blocked: No scored documents")
        return False        
    
    if len(scored) == 1:
        _dbg_gap("Passed: Only one document found (no ambiguity)")
        return True
    
    best_score = scored[0].score
    second_score = scored[1].score
    gap = abs(second_score - best_score)    # Worse (higer score) - Best (lower score)

    best_src = scored[0].doc.metadata.get("source")
    second_src = scored[1].doc.metadata.get("source")

    best_page = scored[0].doc.metadata.get("page")
    second_page = scored[1].doc.metadata.get("page")

    same_file = (best_src == second_src)
    pages_close = (
        isinstance(best_page, int)
        and isinstance(second_page, int)
        and abs(best_page - second_page) <= 2
    )

    _dbg_gap(f"gap={gap}, min_gap={min_gap}, same_file={same_file}, pages_close={pages_close}")

    if gap < min_gap and not (same_file and pages_close):
        return False

    return True

def _validate_density_gate(
    scored: list[ScoredDocument],
    *,
    max_l2: float,
    min_keep: int
) -> list[ScoredDocument] | None:
    """
    Validate that a sufficient number of relevant documents were retrieved.
    Returns:
        list[Document]: The list of documents with score <= max_l2, provided the count meets or exceeds min_keep.
        None: If the number of relevant documents is below min_keep.
    Remarks:
        Expects 'scored' to be pre-sorted (Achieved in fetch_scored_docs_l2).   
    """
    relevant_scored = [sd for sd in scored if sd.score <= max_l2]
    relevant_count = len(relevant_scored)
    is_sufficient = relevant_count >= min_keep
    
    validate = "Passed" if is_sufficient else "Blocked"
    _dbg_den(f"{validate}: relevant_count={relevant_count}, min_keep={min_keep}, max_l2={max_l2}")
    
    return relevant_scored if is_sufficient else None

def gate_scored_docs_l2(
    scored: list[ScoredDocument],
    *, cfg: GateConfig
) -> tuple[list[Document], RetrievalStatus]:
    """
    Apply production-grade gating logic to filter and trim retrieved documents.
    Returns:
        list[Document]: A filtered list of Document objects.
        Empty list: if any gate fails or if minimum density is not met.
    Remarks:
        Expects 'scored' to be pre-sorted (Achieved in fetch_scored_docs_l2). 
    """
    status: RetrievalStatus = RetrievalStatusEnum.OK.value
    
    if not _validate_absolute_gate(scored, max_l2=cfg.max_l2):
        status = RetrievalStatusEnum.REFUSE.value
        _dbg_abs(f"Absolute_gate = {status}")
        return [], status

    relevant_scored = _validate_density_gate(
        scored,
        max_l2=cfg.max_l2,
        min_keep=cfg.min_keep
    )
    
    if relevant_scored is None:
        status = RetrievalStatusEnum.REFUSE.value
        _dbg_den(f"Density gate = {status}")
        return [], status

    final_scored = relevant_scored[: cfg.final_k]

    if not _validate_confidence_gap_gate(final_scored, min_gap=cfg.min_gap):
        status = RetrievalStatusEnum.AMBIGUOUS.value
        _dbg_gap(f"Confidence gap gate = {status}")
        return [], status

    return [sd.doc for sd in final_scored], status

def build_gate_config(cfg: dict[str, Any]) -> GateConfig:
    return GateConfig(
        final_k=int(cfg["final_k"]),
        max_l2=float(cfg["max_l2"]),
        min_keep=int(cfg["min_keep"]),
        min_gap=float(cfg["min_gap"]) if cfg.get("min_gap") is not None else None
    )
