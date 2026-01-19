# src/eval/normalize.py
from __future__ import annotations
from pathlib import Path
from typing import Any

def _basename(x: Any) -> str:
    if not x:
        return ""
    try:
        return Path(str(x)).name
    except Exception:
        return str(x)

def normalize_page(x: Any) -> int | None:
    if x is None:
        return None

    if isinstance(x, int):
        return x

    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def normalize_source_item(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": _basename(meta.get("source")),
        "page": normalize_page(meta.get("page")),
        "page_label": str(meta.get("page_label")).strip() if meta.get("page_label") is not None else None,
        "tag_signature": tuple(meta.get("tag_signature")) if isinstance(meta.get("tag_signature"), (list, tuple)) else None,
    }

def stable_source_key(payload: dict[str, Any]) -> tuple:
    return (
        payload.get("source") or "",
        payload.get("page") if payload.get("page") is not None else -1,
        payload.get("page_label") or "",
        payload.get("tag_signature") or (),
    )
