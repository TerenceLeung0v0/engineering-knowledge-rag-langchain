from __future__ import annotations
from pathlib import Path
from typing import Iterable
from dataclasses import dataclass
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import (
    RAW_DOCS_DIR, CI_FIXTURES_RAW_DOCS_DIR,
    TEXT_SPLITTING_CONFIG, EMBEDDING_CONFIG,
    DEBUG_CONFIG, PROJECT_ROOT, RETRIEVAL_CONFIG
)

from src.utils.text import clean_text
from src.utils.diagnostics import build_debug_logger, warn as _warn
from src.rag.embeddings import build_embeddings
from src.rag.catalog import enrich_metadata
from src.ingest.loaders.pdf_loader import load_pdfs_documents
from src.ingest.loaders.html_loader import load_htmls_documents
from src.ingest.loaders.md_loader import load_mds_documents
from src.ingest.annotate import annotate_docs

_debug_diag = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.ingest",
    key="print_diagnostics"
)

_debug_meta = build_debug_logger(
    cfg=DEBUG_CONFIG,
    domain_path="rag.ingest",
    key="print_metadata"
)

@dataclass(frozen=True)
class IngestStats:
    num_files: int
    num_pages: int
    num_chunks: int

def _count_files(
    path: Path,
    ext: str
) -> int:
    supported = {"pdf", "html", "txt", "docx", "md"}
    ext = ext.lower().lstrip(".")
    if ext not in supported:
        raise ValueError(f"Expected ext in {supported}, got {ext!r}.")

    return len(list(path.glob(f"*.{ext}")))

def _clean_documents(
    chunks: list[Document],
    *, min_length: int=20
) -> list[Document]:
    """
    Normalize and clean document chunks:
    - Ensure page_content is str
    - Strip whitespace
    - Drop empty / too-short chunks
    """
    cleaned_docs: list[Document] = []
    
    for chunk in chunks:
        text = chunk.page_content
        if not isinstance(text, str):
            text = "" if text is None else str(text)    # Normalize to string
        
        text = clean_text(text) 
        if len(text) < min_length:      # Skip empty and too-short chunks
            continue
        
        chunk.page_content = text
        cleaned_docs.append(chunk)
    
    return cleaned_docs

def _emit_ingest_diagnostics(chunks: list[Document]) -> None:
    """
    Emit ingestion diagnostics:
    - source distribution (Controlled via DEBUG_CONFIG["rag"]["ingest"]["print_diagnostics"])
    - missing source / page metadata warnings
    """
    source_counter = Counter()
    missing_source = 0
    missing_page = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")

        if source is None:
            missing_source += 1
            source = "unknown"

        if page is None:
            missing_page += 1

        source_counter[source] += 1

    _debug_diag(f"Sources indexed ({len(source_counter)}):")
    for source, count in sorted(source_counter.items(), key=lambda x: x[0]):
        _debug_diag(f" - {Path(source).name} ({count} chunks)")

    if missing_source > 0: _warn("rag.ingest", f"{missing_source} chunks have no source path.")
    if missing_page > 0: _warn("rag.ingest", f"{missing_page} chunks have no page number.")    

def _enrich_chunks_metadata(chunks: list[Document]) -> list[Document]:
    enriched: list[Document] = []

    for chunk in chunks:
        src = chunk.metadata.get("source")

        if not src:
            _warn("rag.ingest", "Chunk missing metadata['source']")
            enriched.append(chunk)      # Prevent silent data loss
            continue

        chunk.metadata = enrich_metadata(
            project_root=PROJECT_ROOT,
            source=src,
            metadata=chunk.metadata
        )
        enriched.append(chunk)

    for i, chunk in enumerate(enriched[:10]):
        _debug_meta(f"--- Chunk {i} metadata ---")
        _debug_meta(chunk.metadata)

    return enriched

def _load_documents(
    *,
    base_dir: Path | str,
    file_exts: Iterable[str],
) -> tuple[list[Document], dict[str, int]]:
    """
    Wrapper for loading different documents types from base_dir/<type>/...
    """
    base_dir = Path(base_dir)
    normalized_exts = [e.lower().lstrip('.') for e in file_exts]

    docs: list[Document] = []
    if "pdf" in normalized_exts:
        docs.extend(load_pdfs_documents(base_dir / "pdf"))
    
    if "html" in normalized_exts:
        docs.extend(load_htmls_documents(base_dir / "html"))
    
    if "md" in normalized_exts:
        docs.extend(load_mds_documents(base_dir / "md"))

    supported = {"pdf", "html", "txt", "docx", "md"}
    files_counts: dict[str, int] = {}
    for ext in supported:
        folder = base_dir / ext
        files_counts[ext] = _count_files(folder, ext)  if folder.exists() else 0

    return docs, files_counts
    
def build_vectorstore(
    *,
    src_dir: Path | str,
    out_dir: Path | str,
    file_exts: Iterable[str]
) -> IngestStats:
    """
    Build FAISS vectorstore from docs under src_dir/<pdf|html|md>
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    docs, files_counts = _load_documents(
        base_dir=src_dir,
        file_exts=file_exts,
    )
    
    if not docs:
        raise ValueError(f"Error: No documents found under {src_dir} for file_exts={list(file_exts)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTING_CONFIG["chunk_size"],
        chunk_overlap=TEXT_SPLITTING_CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    chunks = _clean_documents(chunks)
    chunks = _enrich_chunks_metadata(chunks)

    chunks = annotate_docs(
        chunks,
        cfg_entities=RETRIEVAL_CONFIG.get("coverage")
    )

    _emit_ingest_diagnostics(chunks)
    
    embeddings = build_embeddings(EMBEDDING_CONFIG)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))

    num_files =  sum(files_counts.values())
    num_pages = sum(1 for doc in docs if doc.metadata.get("page") is not None)
    num_chunks = len(chunks)

    count_types = Counter(doc.metadata.get("doc_type") for doc in docs)
    _debug_meta(f"doc_type count: {dict(count_types)}")

    return IngestStats(
        num_files=num_files,
        num_pages=num_pages,
        num_chunks=num_chunks
    )
