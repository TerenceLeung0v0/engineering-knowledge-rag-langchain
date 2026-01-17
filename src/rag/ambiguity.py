from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TypeAlias, Any

EmbeddingsDocs: TypeAlias = Callable[[list[str]], list[list[float]]]

@dataclass(frozen=True)
class AmbiguityConfig:
    max_options: int = 3
    strict_sig: bool = False
    min_group_gap: float | None = None

    # query-aware signature embedding tie-breaker
    enable_sig_tiebreak: bool = False
    min_sig_sim: float | None = None
    min_sig_sim_gap: float | None = None
    embed_docs: EmbeddingsDocs | None=None
    
    # anchor-content embedding tie-breaker
    enable_anchor_tiebreak: bool = False
    min_anchor_sim: float | None = None
    min_anchor_sim_gap: float | None = None
    
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *, embed_docs: EmbeddingsDocs | None=None
    ) -> "AmbiguityConfig":
        sig_enabled = bool(data.get("enable_sig_tiebreak", False))
        anchor_enabled = bool(data.get("enable_anchor_tiebreak", False))
        
        return cls(
            max_options=int(data.get("max_options", 3)),
            strict_sig=bool(data.get("strict_sig", False)),
            min_group_gap=float(data.get("min_group_gap")) if data.get("min_group_gap") is not None else None,
            
            enable_sig_tiebreak=sig_enabled,
            min_sig_sim=float(data.get("min_sig_sim")) if data.get("min_sig_sim") is not None else None,
            min_sig_sim_gap=float(data.get("min_sig_sim_gap")) if data.get("min_sig_sim_gap") is not None else None,
            
            enable_anchor_tiebreak=anchor_enabled,
            min_anchor_sim=float(data.get("min_anchor_sim")) if data.get("min_anchor_sim") is not None else None,
            min_anchor_sim_gap=float(data.get("min_anchor_sim_gap")) if data.get("min_anchor_sim_gap") is not None else None,            
            
            embed_docs=embed_docs if (sig_enabled or anchor_enabled) else None
        )

