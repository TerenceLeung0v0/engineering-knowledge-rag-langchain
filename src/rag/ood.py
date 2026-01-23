from __future__ import annotations
from dataclasses import dataclass, field
from typing import Pattern, Iterable

from src.config import DEBUG_CONFIG
from src.schemas import RetrievalStatusEnum, RetrievalState
from src.utils.diagnostics import build_debug_logger

import re

_dbg = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.ood",
    key="print_ood"
)

@dataclass(frozen=True)
class OODConfig:
    enabled: bool = True
    
    allow_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    deny_patterns: tuple[Pattern[str], ...] = field(default_factory=tuple)
    
    @classmethod
    def from_dict(cls, cfg: dict | None) -> "OODConfig":
        if not cfg:
            return cls()
        
        enabled = bool(cfg.get("enabled", True))

        def _compile(patterns: Iterable[str]) -> tuple[Pattern[str], ...]:
            compiled: list[Pattern[str]] = []
            for p in patterns or []:
                if not isinstance(p, str) or not p.strip():
                    continue
                try:
                    compiled.append(re.compile(p, re.IGNORECASE))
                except re.error as e:
                    raise ValueError(f"Invalid OOD regex pattern: {p!r}. Error={e}") from e
            return tuple(compiled)

        allow = _compile(cfg.get("allow_patterns", []))
        deny = _compile(cfg.get("deny_patterns", []))
        
        return cls(
            enabled=enabled,
            allow_patterns=allow,
            deny_patterns=deny
        )

def _refuse_state(
    state: RetrievalState,
    *, reason: str
) -> RetrievalState:
    return {
        **state,
        "skip_llm": True,
        "status": RetrievalStatusEnum.REFUSE.value,
        "docs": [],
        "answer": "",
        "refusal_reason": reason
    }

def ood_gate(
    state: RetrievalState,
    *, cfg: OODConfig
) -> RetrievalState:
    if not cfg.enabled:
        _dbg("OOD gate is disabled")
        return state

    if state.get("skip_llm", False):
        _dbg("skip_llm = True")
        return state

    query = state.get("input", "")

    if not isinstance(query, str) or not query.strip():
        _dbg("Reject: Empty/invalid query")
        return _refuse_state(
            state,
            reason="Empty or invalid query"
        )

    for pattern in cfg.deny_patterns:
        if pattern.search(query):
            _dbg(f"Deny: {pattern.pattern!r}")
            return _refuse_state(
                state,
                reason="Out of domain"
            )

    for pattern in cfg.allow_patterns:
        if pattern.search(query):
            _dbg(f"Allow: {pattern.pattern!r}")
            return state
    
    _dbg(f"Default refuse")
    return _refuse_state(
        state,
        reason="Out of domain"
    )
