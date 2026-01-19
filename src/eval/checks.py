from __future__ import annotations
from typing import Any
from pathlib import Path

from src.schemas import RetrievalStatusEnum
from src.eval.schemas import QACase
from src.eval.normalize import stable_source_key, normalize_source_item

def check_status(
    case: QACase,
    actual_status: str
) -> bool:
    return (case.expect_status.strip() == str(actual_status).strip())

def check_sources(
    case: QACase,
    actual_sources: tuple[dict[str, Any], ...]
) -> bool:
    if case.expect_sources:
        expect = {Path(s).name for s in case.expect_sources}
        actual = {str(src.get("source") or "") for src in actual_sources}
        return expect.issubset(actual)
    
    return len(actual_sources) >= int(case.min_sources)

def check_hygiene(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip()
    answer = (payload.get("answer") or payload.get("final_answer") or payload.get("text") or "")
    answer = str(answer).strip()

    match status:
        case RetrievalStatusEnum.REFUSE.value:
            reason = str(payload.get("refusal_reason") or "").strip()
            return bool(reason)

        case RetrievalStatusEnum.AMBIGUOUS.value:
            if payload.get("selected_option") is not None:
                return True
            
            options = payload.get("options")
            return isinstance(options, list) and len(options) >= 2

        case RetrievalStatusEnum.OK.value:
            return len(answer) > 0

        case _:
            return False    

def extract_normalized_sources(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = (
        payload.get("source_documents")
        or payload.get("sources")
        or payload.get("evidence")
        or payload.get("retrieved")
        or []
    )
    outs: list[dict[str, Any]] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
                if isinstance(meta, dict):
                    outs.append(normalize_source_item(meta))
                continue
            
            # Object-style (e.g. LangChain)
            meta = getattr(item, "metadata", None)
            if isinstance(meta, dict):
                outs.append(normalize_source_item(meta))

    outs.sort(key=stable_source_key)
    return tuple(outs)
