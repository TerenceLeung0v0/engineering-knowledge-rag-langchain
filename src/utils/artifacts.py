from typing import Any, Iterable
from pathlib import Path

from src.utils.files import is_file_non_empty
from src.utils.diagnostics import info

import json

def _path_serializer(obj: Any) -> str:
    """
    Path is not serializable in JSON: Path -> str
    """
    if isinstance(obj, Path):
        return str(obj)
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_jsonl(
    path: Path | str,
    rows: Iterable[dict[str, Any]],
    overwrite: bool=True
) -> Path:
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}. Set overwrite=True to overwrite.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_path_serializer) + "\n")
            
    info("artifacts", f"Saved to {path}")
    
    return path
    
def append_jsonl(
    path: Path | str,
    row: dict[str, Any]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_path_serializer) + "\n")
        
    info("artifacts", f"Appended to {path}")
    
    return path

def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    
    if not is_file_non_empty(path): 
        raise FileNotFoundError(path)
        
    rows: list[dict[str, Any]] = []
    
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue
            
            rows.append(json.loads(line))
    
    return rows