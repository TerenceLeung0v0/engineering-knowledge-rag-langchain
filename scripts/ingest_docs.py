from __future__ import annotations
from src.startup_checks import ensure_project_dirs
from src.config import VECTORSTORE_DIR, DEBUG_CONFIG
from src.rag.ingest import build_vectorstore
from src.utils.files import empty_directory_contents
from src.utils.diagnostics import build_debug_logger 

import argparse

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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild vectorstore even if it exists")
    args = parser.parse_args()
        
    ensure_project_dirs()
    
    if args.rebuild:  
        print("Rebuilding vectorstore: erasing existing artifacts...")
        _empty_vectorstore_contents()
    else:            
        if _vectorstore_exists():
            print("Vectorstore already exists. Skipping ingest.")
            print("To rebuild: python scripts/ingest_docs.py --rebuild")
            return    
    
    n_chunks = build_vectorstore()
    print(f"Vectorstore built. Total chunks: {n_chunks}")

if __name__ == "__main__":
    main()
