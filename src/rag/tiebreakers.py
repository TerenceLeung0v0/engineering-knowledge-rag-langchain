# src/rag/tiebreakers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeAlias
from src.config import DEBUG_CONFIG
from src.utils.diagnostics import build_debug_logger

import math

Signature: TypeAlias = tuple[str | None, ...]

_dbg_score = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.tiebreakers",
    key="print_score"
)

@dataclass(frozen=True)
class SigPickResult:
    best_sig: Signature
    best_sim: float
    second_sim: float
    sims: list[tuple[Signature, float]]  # sorted

@dataclass(frozen=True)
class AnchorPickResult:
    best_idx: int
    best_sim: float
    second_sim: float
    sims: list[tuple[int, float]]   # sorted

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _l2_norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))

def _embed_text_cached(
    *,
    embed_docs: Callable[[list[str]], list[list[float]]],
    text: str,
) -> list[float]:
    """
    Manual cache - lru_cache cannot accept embed_docs callable safely.
    Cache key includes embedder identity to avoid cross-model contamination.
    """
    cache: dict[tuple[int, str], list[float]] = getattr(_embed_text_cached, "_cache", {})
    key = (id(embed_docs), text)
    if key in cache:
        return cache[key]      # Already been embedded

    vector = embed_docs([text])[0]
    cache[key] = vector
    setattr(_embed_text_cached, "_cache", cache)
    
    return vector

def _render_signature_text(sig: Signature) -> str:
    """
    Core signature: (domain, doc_type, product)
    Strict signature: (domain, doc_type, product, vendor, version)
    Fallback signature may be ("__file__:xxx.pdf", None, None, ...)
    """
    parts: list[str] = []
    for idx, v in enumerate(sig):
        if v is None:
            continue
        
        match idx:
            case 0:
                parts.append(f"domain: {v}")
            case 1:
                parts.append(f"doc_type: {v}")
            case 2:
                parts.append(f"product: {v}")
            case 3:
                parts.append(f"vendor: {v}")
            case 4:
                parts.append(f"version: {v}")
            case _:
                parts.append(str(v))

    if not parts:
        return "signature: unknown"

    return "; ".join(parts)

def _clip_text(
    text: str,
    *, max_chars: int=800
) -> str:
    t = (text or "").strip()
    
    return t[:max_chars] if len(t) > max_chars else t

def cosine_sim(a: list[float], b: list[float]) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)

def pick_group_by_query_embedding(
    *,
    query: str,
    group_sigs: list[Signature],
    embed_docs: Callable[[list[str]], list[list[float]]],
    min_sig_sim: float | None,
    min_sig_sim_gap: float | None,
) -> SigPickResult | None:
    """
    Return best signature if confident enough, else None.
    Uses cosine similarity between query embedding and signature-text embeddings.
    """
    if not group_sigs:
        return None

    q_vector = embed_docs([query])[0]

    sims: list[tuple[Signature, float]] = []
    for sig in group_sigs:
        sig_text = _render_signature_text(sig)
        sig_vector = _embed_text_cached(embed_docs=embed_docs, text=sig_text)
        sim = cosine_sim(q_vector, sig_vector)
        sims.append((sig, float(sim)))

    sims.sort(key=lambda x: x[1], reverse=True)     # sort for similarity score
    sims_len = len(sims)
    
    best_sig, best_sim = sims[0]
    second_sim = sims[1][1] if sims_len >= 2 else -1.0

    _dbg_score(
        f"min_sig_sim={min_sig_sim}, sims_len={sims_len}, "
        f"best_sim={best_sim}, second_sim={second_sim}, min_sig_sim_gap={min_sig_sim_gap}"
    )
    
    if min_sig_sim is not None and best_sim < float(min_sig_sim):
        _dbg_score(f"best_sim({best_sim}) < min_sig_sim({min_sig_sim})")
        return None
    
    if min_sig_sim_gap is not None and sims_len >= 2:
        if (best_sim - second_sim) < float(min_sig_sim_gap):
            _dbg_score(f"best_sim - second_sim({best_sim - second_sim}) < min_sig_sim_gap({min_sig_sim_gap})")
            return None

    return SigPickResult(
        best_sig=best_sig,
        best_sim=best_sim,
        second_sim=second_sim,
        sims=sims,
    )

def pick_group_by_anchor_content(
    *,
    query: str,
    anchors_text: list[str],
    embed_docs: Callable[[list[str]], list[list[float]]],
    min_anchor_sim: float | None,
    min_anchor_sim_gap: float | None
) -> AnchorPickResult | None:
    """
    Return best index if confident enough, else None.
    Uses cosine similarity between query embedding and anchor-text embeddings.
    """
    if not anchors_text:
        return None
    
    inputs = [query] + [_clip_text(t) for t in anchors_text]
    vectors = embed_docs(inputs)
    
    if not vectors or len(vectors) != len(inputs):
        return None
    
    q_vector = vectors[0]
    sims: list[tuple[int, float]] = []
    for i, anchor_vector in enumerate(vectors[1:]):
        sim = cosine_sim(q_vector, anchor_vector)
        sims.append((i, float(sim)))    
    
    sims.sort(key=lambda x: x[1], reverse=True)
    sims_len = len(sims)
    
    best_idx, best_sim = sims[0]
    second_sim = sims[1][1] if sims_len >= 2 else -1.0

    _dbg_score(
        f"min_anchor_sim={min_anchor_sim}, sims_len={sims_len}, "
        f"best_sim={best_sim}, second_sim={second_sim}, min_anchor_sim_gap={min_anchor_sim_gap}"
    )
    
    if min_anchor_sim is not None and best_sim < float(min_anchor_sim):
        _dbg_score(f"best_sim({best_sim}) < min_anchor_sim({min_anchor_sim})")
        return None
    
    if min_anchor_sim_gap is not None and sims_len >= 2:
        if (best_sim - second_sim) < float(min_anchor_sim_gap):
            _dbg_score(f"best_sim - second_sim({best_sim - second_sim}) < min_anchor_sim_gap({min_anchor_sim_gap})")
            return None
    
    return AnchorPickResult(
        best_idx=best_idx,
        best_sim=best_sim,
        second_sim=second_sim,
        sims=sims
    )
    