from typing import Iterable
from pathlib import Path

from src.utils.files import is_file_non_empty
from src.config import (
    REQUIRED_DIRS, VECTORSTORE_DIR,
)

def ensure_required_dirs(dirs: Iterable[str | Path]) -> None:
    """
    Ensure required directories exist:
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def ensure_project_dirs() -> None:
    """
    Ensure all required project directories exist.
    Call at application startup (scripts, CLI entrypoints, services).
    """   
    ensure_required_dirs(REQUIRED_DIRS)

def check_artifacts_dir_ready(
    path: str | Path,
    *, hint: str | None=None
) -> None:
    """
    Ensure artifacts directory is usable.
    Raises RuntimeError if directory is missing, empty, or contains only empty files
    """
    p = Path(path)
    
    hint_msg = f"\nHint: {hint}" if hint else ""
    
    if not p.exists():
        raise RuntimeError(f"{p} is not found.{hint_msg}")
    
    if not p.is_dir():
        raise RuntimeError(f"{p} is not a directory.{hint_msg}")
    
    items = list(p.glob("*"))
    if not items:
        raise RuntimeError(f"Artifacts directory is empty: {p}{hint_msg}")
    
    files = [item for item in items if item.is_file()]
    if not files:
        raise RuntimeError(f"Artifacts directory contains no files: {p}{hint_msg}")
    
    if not any(is_file_non_empty(file) for file in files):
        raise RuntimeError(f"{p} files are empty or corrupted{hint_msg}")

def check_vectorstore_ready() -> None:
    check_artifacts_dir_ready(
        path=VECTORSTORE_DIR,
        hint="Run: python scripts/ingest_docs.py"
    )
    