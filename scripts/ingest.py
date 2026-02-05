from __future__ import annotations
from src.startup_checks import ensure_project_dirs
from src.config import RAW_DOCS_DIR, VECTORSTORE_DIR, DEBUG_CONFIG
from src.ingest.ingest import build_vectorstore
from src.utils.files import empty_directory_contents
from src.utils.diagnostics import build_debug_logger 

import argparse
import sys

_fs_logger = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="fs",
    key="log_deletes"
)

def _vectorstore_exists() -> bool:
    return (VECTORSTORE_DIR / "index.faiss").exists() and (VECTORSTORE_DIR / "index.pkl").exists()

def _empty_vectorstore_contents(
    dry_run: bool=False,
) -> None:
    empty_directory_contents(
        path=VECTORSTORE_DIR,
        dry_run=dry_run,
        logger=_fs_logger
    )

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vectorstore even if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Show items would be deleted during --rebuild, without acutally deleting.")
    args = parser.parse_args()
        
    ensure_project_dirs()

    if args.dry_run:
        print("Rebuilding vectorstore (dry-run): would erase existing artifacts...")
        _empty_vectorstore_contents(dry_run=True)
        return 0
    
    if args.rebuild:
        print("Rebuilding vectorstore: erasing existing artifacts...")
        _empty_vectorstore_contents()
    else:            
        if _vectorstore_exists():
            print("Vectorstore already exists. Skipping ingest.")
            print("To rebuild: python scripts/ingest.py --rebuild")
            return 0
    
    stats = build_vectorstore(
        src_dir=RAW_DOCS_DIR,
        out_dir=VECTORSTORE_DIR,
        file_exts=["pdf", "html"]
    )
    print(f"Vectorstore built. files={stats.num_files}, pages={stats.num_pages}, chunks={stats.num_chunks}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
