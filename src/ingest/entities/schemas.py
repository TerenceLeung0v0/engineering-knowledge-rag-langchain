from __future__ import annotations
from dataclasses import dataclass, field
from typing import Pattern

import re

@dataclass(frozen=True)
class EntitySpec:
    """
    - key: canonical entity name (e.g., "mqtt", "http", "aws_iot_jobs")
    - aliases: regex patterns to detect the entity in user query / text
    - doc_aliases: regex patterns to detect entity inside doc text/metadata
    """
    key: str
    aliases: tuple[Pattern[str], ...] = field(default_factory=tuple)
    doc_aliases: tuple[Pattern[str], ...] = field(default_factory=tuple)
    doc_min_hits: int = 1

    @staticmethod
    def compile_patterns(
        patterns: list[str] | tuple[str, ...],
        *, ignore_case: bool=True
    ) -> tuple[Pattern[str], ...]:
        flags = re.IGNORECASE if ignore_case else 0
        out: list[Pattern[str]] = []
        for p in patterns or []:
            if not isinstance(p, str) or not p.strip():
                continue
            try:
                out.append(re.compile(p, flags))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {p!r}. Error={e}") from e
        return tuple(out)

@dataclass(frozen=True)
class EntityRegistry:
    entities: dict[str, EntitySpec]

    def keys(self) -> list[str]:
        return list(self.entities.keys())

    def get(self, key: str) -> EntitySpec | None:
        return self.entities.get(key)
