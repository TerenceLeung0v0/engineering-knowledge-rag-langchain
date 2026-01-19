from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

def _path_serializer(obj: Any) -> str:
    """
    Path is not serializable in JSON: Path -> str
    """
    if isinstance(obj, Path):
        return str(obj)
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSONL serializable")

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {p}:{i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be a JSON object at {p}:{i}")
            rows.append(obj)

    return rows

def write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]]
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_path_serializer) + "\n")
