from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Pattern

from langchain_core.documents import Document

from src.config import DEBUG_CONFIG
from src.schemas import RetrievalState, RetrievalStatusEnum
from src.utils.diagnostics import build_debug_logger

import re

_dbg = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.coverage",
    key="print_coverage"
)

@dataclass(frozen=True)
class CoverageConfig:
    enabled: bool = True

    compare_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    generic_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    entity_patterns: dict[str, tuple[Pattern[str], ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "CoverageConfig":
        if not cfg:
            return cls()
        
        enabled = bool(cfg.get("enabled", True))
        
        def _compile(patterns: Iterable[str]) -> tuple[Pattern[str], ...]:
            out: list[Pattern[str]] = []
            for p in patterns or []:
                if not isinstance(p, str) or not p.strip():
                    continue
                try:
                    out.append(re.compile(p, re.IGNORECASE))
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern: {p!r}. Error={e}") from e
            return tuple(out)     

        compare = _compile(cfg.get("compare_markers", []))
        generic = _compile(cfg.get("generic_markers", []))

        entity_patterns: dict[str, tuple[Pattern[str], ...]] = {}
        raw_aliases = cfg.get("entity_aliases", {}) or {}
        for entity, aliases in raw_aliases.items():
            entity_patterns[str(entity)] = _compile(list(aliases) if aliases else [])
     
        return cls(
            enabled=enabled,
            compare_patterns=compare,
            generic_patterns=generic,
            entity_patterns=entity_patterns
        )

def _refuse(
    state: RetrievalState,
    *, reason: str
) -> RetrievalState:
    return {
        **state,
        "skip_llm": True,
        "status": RetrievalStatusEnum.REFUSE.value,
        "docs": [],
        "context": "",
        "answer": "I don't know based on the provided documents.",
        "refusal_reason": reason,
    }

def _any_match(
    patterns: Iterable[Pattern[str]],
    text:str
) -> bool:
    for pattern in patterns:
        match = pattern.search(text)
        if match and match.group(0):
            return True
    return False

def _first_match(
    patterns: Iterable[Pattern[str]],
    text: str
) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match and match.group(0):
            return f"{pattern.pattern!r} -> {match.group(0)!r}"
    return None

def _extract_unique_entities(docs: Iterable[Document]) -> set[str]:
    outs: set[str] = set()
    for doc in docs:
        meta = doc.metadata or {}
        entities = meta.get("entities", [])
        
        if isinstance(entities, (list, tuple, set)):
            for entity in entities:
                if isinstance(entity, str) and entity.strip():
                    outs.add(entity.strip())
    
    return outs

def coverage_gate(
    state: RetrievalState,
    *, cfg: CoverageConfig
) -> RetrievalState:
    if not cfg.enabled:
        _dbg("Coverage gate is disabled")
        return state

    if state.get("skip_llm", False):
        _dbg("skip_llm = True")
        return state

    query = state.get("input", "")
    if not isinstance(query, str) or not query.strip():
        _dbg("Reject: Empty/invalid query")
        return _refuse(
            state,
            reason="Empty or invalid query"
        )

    docs = state.get("docs", [])
    if not docs:
        _dbg("No documents")
        return state  # handle by downstream: refuse_if_no_docs

    is_compare = _any_match(cfg.compare_patterns, query)
    is_generic = _any_match(cfg.generic_patterns, query)

    entities_in_query: list[str] = []
    for entity, aliases in cfg.entity_patterns.items():
        if _any_match(aliases, query):
            entities_in_query.append(entity)

    doc_entities = _extract_unique_entities(docs)
    _dbg(f"is_compare={is_compare}, is_generic={is_generic}, entities_in_query={entities_in_query}")
    _dbg(f"doc_entities = {sorted(doc_entities)}")

    # Debug: show which patterns hit query & docs
    for entity in entities_in_query:
        q_hit = _first_match(cfg.entity_patterns.get(entity, ()), query)
        d_hit = "metadata.entities -> True" if entity in doc_entities else None      
        _dbg(f"entity={entity!r}, query_hit={q_hit}, doc_hit={d_hit}")

    missing_entities = [e for e in entities_in_query if e not in doc_entities]
    if is_compare and len(entities_in_query) >= 2 and missing_entities:
        _dbg(f"Refuse: compare missing -> {missing_entities}")
        return _refuse(
            state,
            reason=f"Missing document coverage for: {', '.join(missing_entities)}"
        )

    if is_generic and entities_in_query and missing_entities:
        _dbg(f"Refuse: generic missing -> {missing_entities}")
        return _refuse(
            state,
            reason=f"Missing document coverage for: {', '.join(missing_entities)}"
        )

    _dbg("Coverage gate is passed")
    return state
