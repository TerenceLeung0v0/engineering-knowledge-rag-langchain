# src/eval/result_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EvalResult:
    id: str
    query: str

    expect_status: str
    actual_status: str

    status_ok: bool
    sources_ok: bool
    hygiene_ok: bool

    selected_option: int | None = None
    refusal_reason: str | None = None
    sources: tuple[dict[str, Any], ...] = ()
    answer_preview: str | None = None

def clip(text: str | None, n: int = 200) -> str | None:
    if text is None:
        return None
    t = str(text).strip()
    return t[:n] if len(t) > n else t
