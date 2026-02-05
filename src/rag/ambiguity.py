from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeAlias, Any, Pattern, Iterable

from src.rag.entity_extract import EntityExtractor

import re

EmbeddingsDocs: TypeAlias = Callable[[list[str]], list[list[float]]]

@dataclass(frozen=True)
class AmbiguityConfig:
    max_options: int = 3
    min_group_gap: float | None = None

    # embedding tie-breaker
    strict_sig: bool = False
    embed_docs: EmbeddingsDocs | None=None

    # query-aware signature embedding tie-breaker
    enable_sig_tiebreak: bool = False
    min_sig_sim: float | None = None
    min_sig_sim_gap: float | None = None
    
    # anchor-content embedding tie-breaker
    enable_anchor_tiebreak: bool = False
    min_anchor_sim: float | None = None
    min_anchor_sim_gap: float | None = None

    # entity-aware
    enable_entity_resolve: bool = False
    require_full_entity_coverage: bool = False
    entity_extractor: EntityExtractor | None = None

    # ambiguous handling for generic queries
    keep_ambiguous_for_generic_queries: bool = False
    generic_query_patterns: tuple[Pattern[str], ...] = ()
    facet_query_patterns: tuple[Pattern[str], ...] = ()

    @classmethod
    def from_dict(
        cls,
        retrieval_cfg: dict[str, Any],
        ambiguity_cfg: dict[str, Any],
        *, embed_docs: EmbeddingsDocs | None=None,
        entity_extractor: EntityExtractor | None=None
    ) -> "AmbiguityConfig":
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
        
        ambiguous_keep = bool(ambiguity_cfg.get("keep_ambiguous_for_generic_queries", False))
        sig_enabled = bool(retrieval_cfg.get("enable_sig_tiebreak", False))
        anchor_enabled = bool(retrieval_cfg.get("enable_anchor_tiebreak", False))

        generic_query_patterns = _compile(ambiguity_cfg.get("generic_query_patterns", []))
        facet_query_patterns =  _compile(ambiguity_cfg.get("facet_query_patterns", []))
                         
        return cls(
            max_options=int(retrieval_cfg.get("max_options", 3)),
            strict_sig=bool(retrieval_cfg.get("strict_sig", False)),
            min_group_gap=float(retrieval_cfg.get("min_group_gap")) if retrieval_cfg.get("min_group_gap") is not None else None,
            
            enable_sig_tiebreak=sig_enabled,
            min_sig_sim=float(retrieval_cfg.get("min_sig_sim")) if retrieval_cfg.get("min_sig_sim") is not None else None,
            min_sig_sim_gap=float(retrieval_cfg.get("min_sig_sim_gap")) if retrieval_cfg.get("min_sig_sim_gap") is not None else None,
            
            enable_anchor_tiebreak=anchor_enabled,
            min_anchor_sim=float(retrieval_cfg.get("min_anchor_sim")) if retrieval_cfg.get("min_anchor_sim") is not None else None,
            min_anchor_sim_gap=float(retrieval_cfg.get("min_anchor_sim_gap")) if retrieval_cfg.get("min_anchor_sim_gap") is not None else None,            
            
            embed_docs=embed_docs if (sig_enabled or anchor_enabled) else None,

            enable_entity_resolve=bool(retrieval_cfg.get("enable_entity_resolve", False)),
            require_full_entity_coverage=bool(retrieval_cfg.get("require_full_entity_coverage", False)),
            entity_extractor=entity_extractor,
            
            keep_ambiguous_for_generic_queries=ambiguous_keep,
            generic_query_patterns=generic_query_patterns,
            facet_query_patterns=facet_query_patterns
        )
