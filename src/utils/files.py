from __future__ import annotations
from typing import Callable, TypeAlias
from pathlib import Path

import shutil

Logger:TypeAlias = Callable[[str], None]

def _void_logger(_: str) -> None:
    return

def is_file_non_empty(path: str | Path) -> bool:
    """
    Check if path is a file and non-empty. 
    Returns False if file is missing, empty, or inaccessible.
    """
    p = Path(path)
    
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except (PermissionError, OSError):
        return False

def empty_directory_contents(
    path: Path,
    *,
    dry_run: bool=False,
    logger: Logger | None=None
) -> int:
    """
    Delete all contents under a directory (excluding the directory itself).
    - dry_run: If True, only print what would be deleted without concrete action
    Return
    - int
    - if dry_run=False, number of items deleted
    - if dry_run=True, returns number of items that would be deleted
    Notes:
    - Symlink handling:
      - If item is a symlink, the symlink is only unlinked.
    """
    log = logger or _void_logger
    
    if not path.exists():
        return 0
        
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    
    items: list[Path] = list(path.iterdir())
    
    if not items:
        log(f"{path} is already empty.")
        return 0
    
    log(f"{len(items)} items under {path}:")
    for item in items:
        try:
            log(f" - {item.relative_to(path)}")
        except ValueError:
            log(f" - {item}")
        
    if dry_run:
        log("Dry run enabled. No changes mode (items not deleted).")
        return len(items)   # Forecast items to be deleted
    
    deleted = 0
    for item in items:
        try:
            if item.is_symlink():
                item.unlink()
                log(f"Unlinked symlink: {item.name}")       
            elif item.is_file():
                item.unlink()
                log(f"Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                log(f"Deleted directory: {item.name}")
            else:
                continue
            
            deleted += 1           
        except Exception as e:
            log(f"Failed to delete {item.name}: {type(e).__name__}: {e}")
    
    return deleted
