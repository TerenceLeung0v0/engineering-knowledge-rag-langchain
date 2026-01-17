from __future__ import annotations
from pathlib import Path
from typing import Callable, DefaultDict
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import DEBUG_CONFIG
from src.schemas import ScoredDocument, RetrievalState, RetrievalOption, RetrievalStatusEnum
from src.rag.formatting import collect_sources, normalize_page
from src.rag.gating import GateConfig, gate_scored_docs_l2
from src.rag.catalog import tag_signature
from src.rag.ambiguity import AmbiguityConfig
from src.rag.tiebreakers import pick_group_by_query_embedding, pick_group_by_anchor_content
from src.utils.diagnostics import build_debug_logger

_dbg_ambiguous = build_debug_logger(
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

def _get_source_info(doc: Document) -> tuple[str, str]:
    meta = doc.metadata or {}
    filename = Path(meta.get("source", "unknown")).name
    page = normalize_page(meta.get("page"))

    return filename, str(page)

def _doc_signature(doc: Document) -> tuple[str, str]:
    filename, page = _get_source_info(doc)

    return filename, page

def _option_signature(option: RetrievalOption) -> tuple[tuple[str, str], ...]:
    signature = sorted((src.filename, str(src.page)) for src in option.sources)

    return tuple(signature)

def _deduplicate_options(options: list[RetrievalOption]) -> list[RetrievalOption]:
    """
    Helper function for _prepare_retrieval_options.
    """
    seen: set[tuple[tuple[str, str], ...]] = set()
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

def _prioritize_documents_for_anchor(
    *,
    anchor: Document,
    scored: list[ScoredDocument]
) -> list[Document]:
    """
    Helper function for _prepare_retrieval_options.
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
    """
    Helper function for _prepare_retrieval_options.
    """
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
                case 0:
                    should_pick = (page not in seen_pages)
                case 1:
                    should_pick = (filename not in seen_files)
                case 2:
                    should_pick = True

            if should_pick:
                picked.append(candidate)
                seen_signatures.add(sig)
                seen_files.add(filename)
                seen_pages.add(page)

    return picked

def _safe_tag_signature(
    doc: Document,
    *, strict: bool
) -> tuple[str | None, ...]:
    sig = tag_signature(doc.metadata or {}, strict=strict)

    if any(v is not None for v in sig):
        return sig

    filename = Path((doc.metadata or {}).get("source", "unknown")).name
    num = 5 if strict else 3

    return (f"__file__:{filename}",) + (None,) * (num - 1)

def _group_scored_by_tag_signature(
    scored: list[ScoredDocument],
    *, strict: bool = False
) -> list[list[ScoredDocument]]:
    """
    Helper function for _resolve_tag_ambiguity.
    """
    buckets: DefaultDict[tuple[str | None, ...], list[ScoredDocument]] = defaultdict(list)

    for sd in scored:
        sig = _safe_tag_signature(sd.doc, strict=strict)
        buckets[sig].append(sd)

    groups: list[list[ScoredDocument]] = []
    for _, items in buckets.items():
        items.sort(key=lambda x: float(x.score))
        groups.append(items)

    groups.sort(key=lambda g: float(g[0].score) if g else 1e18)
    return groups

def _resolve_by_groups_score_gap(
    groups: list[list[ScoredDocument]],
    *, cfg: AmbiguityConfig
) -> list[Document] | None:
    if cfg.min_group_gap is not None and len(groups) >= 2:
        best_gp0 = float(groups[0][0].score)
        best_gp1 = float(groups[1][0].score)
        gap = best_gp1 - best_gp0

        _dbg_ambiguous(
            f"Tag-group gap: best_gp0={best_gp0:.4f}, best_gp1={best_gp1:.4f}, "
            f"gap={gap:.4f}, min_group_gap={cfg.min_group_gap}"
        )

        if gap >= float(cfg.min_group_gap):
            return [sd.doc for sd in groups[0]]

    return None

def _tiebreak_signature_embedding(
    groups: list[list[ScoredDocument]],
    *,
    query: str,
    cfg: AmbiguityConfig
) -> list[Document] | None:
    """
    Helper function for _tiebreak_groups_by_query_aware.
    Expects groups are non-empty and sorted.
    """
    if not cfg.enable_sig_tiebreak:
        _dbg_ambiguous("Tie-breaking[Sig] is disaled")
        return None

    if cfg.embed_docs is None:
        _dbg_ambiguous("Tie-breaking[Sig] is skipped: embed_docs=None")
        return None

    group_sigs: list[tuple[str | None, ...]] = []
    sig_to_idx: dict[tuple[str | None, ...], int] = {}

    for i, g in enumerate(groups):
        sig = _safe_tag_signature(g[0].doc, strict=cfg.strict_sig)
        group_sigs.append(sig)
        sig_to_idx.setdefault(sig, i)

    picked = pick_group_by_query_embedding(
        query=query,
        group_sigs=group_sigs,
        embed_docs=cfg.embed_docs,
        min_sig_sim=cfg.min_sig_sim,
        min_sig_sim_gap=cfg.min_sig_sim_gap
    )

    if picked is None:
        _dbg_ambiguous(
            "Tie-breaking[Sig] no-pick: "
            f"min_sig_sim={cfg.min_sig_sim}, min_sig_sim_gap={cfg.min_sig_sim_gap}"
        )
        return None

    _dbg_ambiguous(
        f"sig_tiebreak pick: best_sig={picked.best_sig}, "
        f"best_sim={picked.best_sim:.4f}, second_sim={picked.second_sim:.4f}, "
        f"sims={[(s, round(v, 4)) for (s, v) in picked.sims]}"
    )

    winner_idx = sig_to_idx.get(picked.best_sig)
    if winner_idx is None or winner_idx < 0 or winner_idx >= len(groups) or not groups[winner_idx]:
        return None

    return [sd.doc for sd in groups[winner_idx]]

def _tiebreak_anchor_embedding(
    groups: list[list[ScoredDocument]],
    *,
    query: str,
    cfg: AmbiguityConfig
) -> list[Document] | None:
    """
    Helper function for _tiebreak_groups_by_query_aware.
    Expects groups are non-empty and sorted.
    """
    if not cfg.enable_anchor_tiebreak:
        _dbg_ambiguous("Tie-breaking[Anchor] is disaled")
        return None

    if len(groups) < 2:
        _dbg_ambiguous("Tie-breaking[Anchor] is skipped: groups < 2")
        return None

    if cfg.embed_docs is None:
        _dbg_ambiguous("Tie-breaking[Anchor] is skipped: embed_docs=None")
        return None

    anchors_text = [str(g[0].doc.page_content or "") for g in groups]
    picked = pick_group_by_anchor_content(
        query=query,
        anchors_text=anchors_text,
        embed_docs=cfg.embed_docs,
        min_anchor_sim=cfg.min_anchor_sim,
        min_anchor_sim_gap=cfg.min_anchor_sim_gap
    )

    if picked is None:
        _dbg_ambiguous(
            "Tie-breaking[Anchor] no-pick: "
            f"min_anchor_sim={cfg.min_anchor_sim}, min_anchor_sim_gap={cfg.min_anchor_sim_gap}"
        )
        return None

    _dbg_ambiguous(
        f"anchor_tiebreak pick: best_idx={picked.best_idx}, "
        f"best_sim={picked.best_sim:.4f}, second_sim={picked.second_sim:.4f}, "
        f"sims={[(i, round(v, 4)) for (i, v) in picked.sims]}"
    )

    winner_idx = picked.best_idx
    if winner_idx < 0 or winner_idx >= len(groups) or not groups[winner_idx]:
        return None

    return [sd.doc for sd in groups[winner_idx]]

def _tiebreak_groups_by_query_aware(
    groups: list[list[ScoredDocument]],
    *,
    query: str,
    cfg: AmbiguityConfig
) -> list[Document] | None:
    if cfg.embed_docs is None:
        _dbg_ambiguous("Tie-breaking is skipped: embed_docs=None")
        return None

    non_empty_groups = [g for g in groups if g]
    if len(non_empty_groups) < 2:
        _dbg_ambiguous("Tie-breaking is skipped: non_empty_groups < 2")
        return None

    sig_docs = _tiebreak_signature_embedding(
        groups=non_empty_groups,
        query=query,
        cfg=cfg
    )
    if sig_docs is not None:
        return sig_docs

    return _tiebreak_anchor_embedding(
        groups=non_empty_groups,
        query=query,
        cfg=cfg
    )

def _prepare_retrieval_options(
    groups: list[list[ScoredDocument]],
    *,
    cfg: AmbiguityConfig,
    effective_k: int
) -> list[RetrievalOption]:
    options: list[RetrievalOption] = []
    safe_k = max(0, effective_k - 1)

    for opt_id, group in enumerate(groups[: cfg.max_options], start=1):
        anchor = group[0].doc

        candidates = _prioritize_documents_for_anchor(
            anchor=anchor,
            scored=group
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
                option_id=opt_id,
                docs=docs,
                sources=sources,
                best_l2=float(group[0].score)
            )
        )

    return _deduplicate_options(options)

def _resolve_tag_ambiguity(
    scored: list[ScoredDocument],
    *,
    query: str,
    final_k: int,
    cfg: AmbiguityConfig
) -> tuple[list[RetrievalOption], bool, list[Document]]:
    """
    Resolve tag ambiguity.
    Return: (options, auto_resolved, resolved_docs)
    """
    if not scored:
        return [], False, []

    groups = _group_scored_by_tag_signature(scored, strict=cfg.strict_sig)
    effective_k = max(1, final_k)

    # 1) only one group
    if len(groups) == 1:
        resolved_docs = [sd.doc for sd in groups[0][: effective_k]]
        return [], True, resolved_docs

    # 2) resolve by group score gap
    resolved_docs = _resolve_by_groups_score_gap(groups, cfg=cfg)
    if resolved_docs is not None:
        return [], True, resolved_docs[: effective_k]

    # 3) query-aware tie-break
    winner_docs = _tiebreak_groups_by_query_aware(
        groups=groups,
        query=query,
        cfg=cfg
    )
    if winner_docs is not None:
        return [], True, winner_docs[: effective_k]

    # 4) user options
    options = _prepare_retrieval_options(
        groups=groups,
        cfg=cfg,
        effective_k=effective_k
    )

    return options, False, []

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
    ambiguity_cfg: AmbiguityConfig
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
                raw_options, auto_resolved, resolved_docs = _resolve_tag_ambiguity(
                    scored,
                    query=query,
                    final_k=gate_cfg.final_k,
                    cfg=ambiguity_cfg
                )

                _dbg_ambiguous(f"Ambiguous: Raw options={len(raw_options)}, auto_resolved={auto_resolved}")

                if auto_resolved:
                    return _ok_state(
                        state,
                        docs=resolved_docs
                    )

                if len(raw_options) == 1:
                    return _ok_state(
                        state,
                        docs=raw_options[0].docs
                    )

                if len(raw_options) == 0:
                    return _refuse_state(
                        state,
                        reason="Ambiguous gate produced no valid options"
                    )

                return _ambiguous_state(
                    state,
                    options=raw_options
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
