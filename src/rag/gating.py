from __future__ import annotations
from typing import Any
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import DEBUG_CONFIG
from src.schemas import ScoredDocument, RetrievalStatus, RetrievalStatusEnum
from src.utils.diagnostics import build_debug_logger
from src.rag.catalog import tag_signature

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
    soft_max_l2: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "GateConfig":
        return cls(
            final_k=int(cfg.get("final_k", 4)),
            max_l2=float(cfg.get("max_l2", 0.8)),
            min_keep=int(cfg.get("min_keep", 1)),
            min_gap=float(cfg.get("min_gap")) if cfg.get("min_gap") is not None else None,
            soft_max_l2=float(cfg.get("soft_max_l2")) if cfg.get("soft_max_l2") is not None else None,
        )

def _validate_absolute_gate(
    scored: list[ScoredDocument],
    *,
    max_l2: float
) -> bool:
    if not scored:
        _dbg_abs("Blocked: No scored documents")
        return False

    best = float(scored[0].score)
    passed = best <= float(max_l2)

    _dbg_abs(f"Hard threshold {'passed' if passed else 'failed'}: best={best:.4f}, max_l2={max_l2:.4f}")
    return passed

def _select_l2_threshold(
    scored: list[ScoredDocument],
    *,
    hard_max_l2: float,
    soft_max_l2: float | None,
) -> float | None:
    """
    Select effective L2 threshold only.
    No density or relevance counting here.
    """
    if not scored:
        return None

    if _validate_absolute_gate(scored, max_l2=hard_max_l2):
        _dbg_abs(f"Threshold selected: hard_max_l2={hard_max_l2}")
        return float(hard_max_l2)

    if soft_max_l2 is None:
        _dbg_abs("Hard failed and soft_max_l2=None")
        return None

    best = float(scored[0].score)
    if best > float(soft_max_l2):
        _dbg_abs(f"Blocked: best({best:.4f}) > soft_max_l2({soft_max_l2:.4f})")
        return None

    _dbg_abs(f"Threshold selected: soft_max_l2={soft_max_l2:.4f}")
    return float(soft_max_l2)

def _filter_scored_by_threshold(
    scored: list[ScoredDocument],
    *,
    max_l2: float
) -> list[ScoredDocument]:
    threshold = float(max_l2)
    out: list[ScoredDocument] = []

    for sd in scored:
        if float(sd.score) <= threshold:
            out.append(sd)
        else:
            break

    return out

def _validate_density_gate(
    scored_filtered: list[ScoredDocument],
    *,
    min_keep: int
) -> list[ScoredDocument] | None:
    count = len(scored_filtered)
    passed = count >= min_keep

    _dbg_den(f"{'Passed' if passed else 'Blocked'}: relevant_count={count}, min_keep={min_keep}")

    return scored_filtered if passed else None

def _validate_confidence_gap_gate(
    scored: list[ScoredDocument],
    *,
    min_gap: float | None
) -> bool:
    if min_gap is None:
        _dbg_gap("Skipped: min_gap disabled")
        return True

    if not scored:
        return False

    if len(scored) == 1:
        _dbg_gap("Passed: single document")
        return True

    best = scored[0]
    second = scored[1]

    best_score = float(best.score)
    second_score = float(second.score)
    gap = abs(second_score - best_score)

    best_meta = best.doc.metadata or {}
    second_meta = second.doc.metadata or {}

    same_file = best_meta.get("source") == second_meta.get("source")

    best_page = best_meta.get("page")
    second_page = second_meta.get("page")
    pages_close = (
        isinstance(best_page, int)
        and isinstance(second_page, int)
        and abs(best_page - second_page) <= 2
    )

    best_sig = tag_signature(best_meta, strict=False)
    second_sig = tag_signature(second_meta, strict=False)
    same_sig = (best_sig == second_sig) and any(v is not None for v in best_sig)

    _dbg_gap(
        f"gap={gap:.4f}, min_gap={min_gap:.4f}, "
        f"same_file={same_file}, pages_close={pages_close}, same_sig={same_sig}"
    )

    if gap < min_gap and not (same_file and pages_close) and not same_sig:
        return False

    return True

def gate_scored_docs_l2(
    scored: list[ScoredDocument],
    *,
    cfg: GateConfig
) -> tuple[list[Document], RetrievalStatus]:
    if not scored:
        return [], RetrievalStatusEnum.REFUSE.value

    threshold = _select_l2_threshold(
        scored,
        hard_max_l2=cfg.max_l2,
        soft_max_l2=cfg.soft_max_l2,
    )

    if threshold is None:
        return [], RetrievalStatusEnum.REFUSE.value

    scored_filtered = _filter_scored_by_threshold(
        scored,
        max_l2=threshold
    )

    scored_dense = _validate_density_gate(
        scored_filtered,
        min_keep=cfg.min_keep
    )

    if scored_dense is None:
        return [], RetrievalStatusEnum.REFUSE.value

    if not _validate_confidence_gap_gate(
        scored_dense,
        min_gap=cfg.min_gap
    ):
        return [], RetrievalStatusEnum.AMBIGUOUS.value

    docs = [sd.doc for sd in scored_dense[: cfg.final_k]]
    return docs, RetrievalStatusEnum.OK.value
