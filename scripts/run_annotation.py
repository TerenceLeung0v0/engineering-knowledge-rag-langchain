from __future__ import annotations
from typing import Iterable
from pathlib import Path

from langchain_core.documents import Document

from src.config import RETRIEVAL_CONFIG
from src.utils.artifacts import save_jsonl, load_jsonl
from src.ingest.annotate import annotate_docs

import argparse

def _load_jsonl_docs(path: Path) -> Iterable[Document]:
    rows = load_jsonl(path) 

    return [
        Document(
            page_content=row.get("text", ""),
            metadata=row.get("metadata", {}) or {}
        )
        for row in rows
    ]

def _save_jsonl_docs(
    path: Path,
    docs: list[Document]
) -> Path:
    rows_stream = ({"text": d.page_content, "metadata": d.metadata or {}} for d in docs)
    
    save_jsonl(
        path,
        rows_stream,
        overwrite=True
    )
    
    return path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--source", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    docs = _load_jsonl_docs(in_path)
    cfg_entities = (RETRIEVAL_CONFIG.get("coverage") or {})

    annotated = annotate_docs(
        docs,
        source=args.source,
        cfg_entities=cfg_entities,
    )

    _save_jsonl_docs(out_path, annotated)

if __name__ == "__main__":
    main()
