from __future__ import annotations
from typing import Any

from src.eval.schemas import QACase
from src.eval.result_types import EvalResult, clip
from src.eval.checks import (
    check_status, check_sources, check_hygiene,
    extract_normalized_sources,
)

def _get_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or "").strip()

def _get_answer_preview(payload: dict[str, Any]) -> str | None:
    ans = payload.get("answer") or payload.get("final_answer") or payload.get("text")

    return clip(str(ans), 200) if ans is not None else None

def run_case(
    chain: Any,
    case: QACase
) -> EvalResult:
    """
    chain: your RAG chain object with invoke() method.
    """
    payload = chain.invoke({"input": case.query})

    if not isinstance(payload, dict):
        payload = {"status": "refuse", "refusal_reason": "Chain returned non-dict payload"}

    actual_status = _get_status(payload)
    sources = extract_normalized_sources(payload)

    status_ok = check_status(case, actual_status)
    sources_ok = check_sources(case, sources)
    hygiene_ok = check_hygiene(payload)

    return EvalResult(
        id=case.id,
        query=case.query,
        expect_status=case.expect_status,
        actual_status=actual_status,
        status_ok=status_ok,
        sources_ok=sources_ok,
        hygiene_ok=hygiene_ok,
        selected_option=payload.get("selected_option"),
        refusal_reason=payload.get("refusal_reason"),
        sources=tuple(sources),
        answer_preview=_get_answer_preview(payload),
    )
