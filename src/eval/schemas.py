from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

from src.schemas import RetrievalStatusEnum

ALLOWED_STATUS = {e.value for e in RetrievalStatusEnum} #  "ok", "refuse", "ambiguous"

@dataclass(frozen=True)
class QACase:
    """
    Evaluation contract for one test case (one JSONL line).
    """
    # Required
    id: str
    query: str
    expect_status: str
    # Optional
    expect_sources: tuple[str, ...] = ()
    expect_sources_any: tuple[str, ...] = ()
    min_sources: int = 0
    notes: str | None = None

def _clean_str(x: Any) -> str:
    return str(x).strip()

def _parse_sources_list(
    *,
    case_id: str,
    raw: Any,
    field_name: str
) -> tuple[str, ...]:
    if raw is None:
        raw = []
    
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"[{case_id}] {field_name} must be a list/tuple[str]")

    outs: list[str] = []
    for src in raw:
        norm_src = _clean_str(src)
        if norm_src:
            outs.append(norm_src)
    return tuple(outs)

def parse_case(case: dict[str, Any]) -> QACase:
    if not isinstance(case, dict):
        raise ValueError(f"Each line must be a JSON object, got {type(case).__name__}")

    case_id = _clean_str(case.get("id", ""))
    q = _clean_str(case.get("query", "")) or _clean_str(case.get("question", ""))
    expect_status = _clean_str(case.get("expect_status", ""))

    if not case_id:
        raise ValueError("Missing/empty required field: id")

    if not q:
        raise ValueError(f"[{case_id}] Missing/empty required field: query/question")

    if expect_status not in ALLOWED_STATUS:
        raise ValueError(
            f"[{case_id}] Invalid expect_status={expect_status!r}. "
            f"Allowed: {sorted(ALLOWED_STATUS)}"
        )

    expect_sources = _parse_sources_list(
        case_id=case_id,
        raw=case.get("expect_sources", []),
        field_name="expect_sources"        
    )

    expect_sources_any = _parse_sources_list(
        case_id=case_id,
        raw=case.get("expect_sources_any", []),
        field_name="expect_sources_any"        
    )

    min_sources_raw = case.get("min_sources", 0)
    try:
        min_sources = int(min_sources_raw)
    except Exception:
        raise ValueError(f"[{case_id}] min_sources must be an int")

    if min_sources < 0:
        raise ValueError(f"[{case_id}] min_sources must be >= 0")

    notes_raw = case.get("notes")
    notes = _clean_str(notes_raw) if notes_raw is not None else None
    if notes == "":
        notes = None

    return QACase(
        id=case_id,
        query=q,
        expect_status=expect_status,
        expect_sources=expect_sources,
        expect_sources_any=expect_sources_any,
        min_sources=min_sources,
        notes=notes,
    )

def validate_cases(cases: Iterable[QACase]) -> list[QACase]:
    """
    Validate id uniqueness.
    """
    outs = list(cases)
    seen: set[str] = set()
    
    for out in outs:
        if out.id in seen:
            raise ValueError(f"Duplicate case id: {out.id}")
        seen.add(out.id)
        
        if not out.id.strip():
            raise ValueError("Empty case id is found")

    return outs
