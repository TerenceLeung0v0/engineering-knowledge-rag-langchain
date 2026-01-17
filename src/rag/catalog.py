# src/rag/catalog.py
from __future__ import annotations
from typing import Any
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import json
import re

_REGISTRY_REL = Path("data/catalog/docs_registry.json")

_TAG_KEYS_CORE: tuple[str, ...] = ("domain", "doc_type", "product")
_TAG_KEYS_STRICT: tuple[str, ...] = _TAG_KEYS_CORE + ("vendor", "version")

@dataclass(frozen=True)
class DocTags:
    domain: str | None = None     # e.g. mqtt, aws_iot, iot_general
    doc_type: str | None = None   # spec, guide, whitepaper, runbook, notes
    vendor: str | None = None     # aws, oasis, internal
    product: str | None = None    # iot_core, mqtt
    version: str | None = None    # 3.1.1, etc

    def to_metadata(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.domain:
            out["domain"] = self.domain
        if self.doc_type:
            out["doc_type"] = self.doc_type
        if self.vendor:
            out["vendor"] = self.vendor
        if self.product:
            out["product"] = self.product
        if self.version:
            out["version"] = self.version
        return out

def _norm_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out if out else None

def _norm_tag_value(value: Any) -> str | None:
    out = _norm_opt_str(value)
    return out.lower() if out else None

def _is_rule_match(
    filename: str,
    match: dict[str, Any]
) -> bool:
    """
    Determines a match based on the presence of specific keys in the match dict.
    """
    if not isinstance(match, dict):
        return False
    
    if "filename" in match:
        return filename == str(match["filename"])

    if "filename_regex" in match:
        pat = str(match["filename_regex"])
        return re.search(pat, filename, flags=re.IGNORECASE) is not None

    return False

@lru_cache(maxsize=8)
def _load_rules_cached(
    abs_path: Path | str,
    mtime_ns: int
) -> list[dict[str, Any]]:
    path = Path(abs_path)
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        
        rules = data.get("rules", [])
        if not isinstance(rules, list):
            return []
        
        return [rule for rule in rules if isinstance(rule, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def _load_rules(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / _REGISTRY_REL
    
    if not path.exists():
        return []
    try:
        stat = path.stat()
        return _load_rules_cached(path.resolve(), stat.st_mtime_ns)
    except OSError:
        return []

def resolve_doc_tags(
    *,
    project_root: Path,
    source: str | Path
) -> DocTags:
    filename = Path(source).name
    rules = _load_rules(project_root)

    for rule in rules:
        match = rule.get("match", {})
        tags = rule.get("tags", {})

        if not isinstance(match, dict) or not isinstance(tags, dict):
            continue

        if _is_rule_match(filename, match):
            return DocTags(
                domain=_norm_tag_value(tags.get("domain")),
                doc_type=_norm_tag_value(tags.get("doc_type")),
                vendor=_norm_tag_value(tags.get("vendor")),
                product=_norm_tag_value(tags.get("product")),
                version=_norm_tag_value(tags.get("version"))
            )

    return DocTags()

def enrich_metadata(
    *,
    project_root: Path,
    source: str | Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    tags = resolve_doc_tags(
        project_root=project_root,
        source=source
    ).to_metadata()

    for key, value in tags.items():
        meta.setdefault(key, value)

    return meta

def tag_signature(
    meta: dict[str, Any],
    *, strict: bool=False
) -> tuple[str | None, ...]:
    keys = _TAG_KEYS_STRICT if strict else _TAG_KEYS_CORE
    
    return tuple(_norm_tag_value(meta.get(key)) for key in keys)
