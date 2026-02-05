from __future__ import annotations
from dataclasses import dataclass
from typing import Pattern

def _any_match(
    patterns: tuple[Pattern[str], ...],
    text: str
) -> bool:
    for pattern in patterns:
        if pattern.search(text):
            return True
    return False

@dataclass(frozen=True)
class EntityExtractor:
    entity_patterns: dict[str, tuple[Pattern[str], ...]]

    def extract(self, query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        hits: list[str] = []
        for entity, aliases in (self.entity_patterns or {}).items():
            if _any_match(aliases, q):
                hits.append(entity)
        return hits
