from __future__ import annotations
from typing import Any

from src.ingest.entities.schemas import EntityRegistry, EntitySpec

def _parse_doc_aliases(v: Any) -> tuple[list[str], int]:
    match v:
        case dict():
            patterns = v.get("patterns", [])
            hits = int(v.get("min_hits", 1) or 1)
            return list(patterns), max(1, hits)
        case list() | tuple() | set():
            return list(v), 1
        case _:
            return [], 1

def build_entity_registry(cfg: dict[str, Any] | None) -> EntityRegistry:
    """
    Expected shape (example):
    cfg = {
      "entity_aliases": {...},      # qurey-side
      "entity_doc_aliases": {...}   # doc-side
    }
    """
    cfg = cfg or {}
    raw_aliases = cfg.get("entity_aliases", {}) or {}
    raw_doc_aliases = cfg.get("entity_doc_aliases", {}) or {}

    entities: dict[str, EntitySpec] = {}
    for key, aliases in raw_aliases.items():
        k = str(key)
        alias_list = list(aliases) if aliases else []
        doc_list, doc_min_hits = _parse_doc_aliases(raw_doc_aliases.get(k))
        entities[k] = EntitySpec(
            key=k,
            aliases=EntitySpec.compile_patterns(alias_list, ignore_case=True),
            doc_aliases=EntitySpec.compile_patterns(doc_list, ignore_case=True),
            doc_min_hits=doc_min_hits
        )
    
    for key, doc_aliases in raw_doc_aliases.items():
        k = str(key)
        if k in entities:
            continue
        doc_list, doc_min_hits = _parse_doc_aliases(doc_aliases)
        entities[k] = EntitySpec(
            key=k,
            aliases=tuple(),
            doc_aliases=EntitySpec.compile_patterns(doc_list, ignore_case=True),
            doc_min_hits=doc_min_hits
        )
        
    return EntityRegistry(entities=entities)
