from __future__ import annotations
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs_documents(
    path: Path | str,
    *, glob_pattern: str = "**/*.pdf"
) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    
    pdfs = list(path.glob(glob_pattern))
    if not pdfs:
        raise RuntimeError(f"No PDF documents matching '{glob_pattern}' found in {path}")
        
    for pdf_path in sorted(pdfs):
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", str(pdf_path))
            d.metadata["doc_type"] = "pdf"
        
        docs.extend(loaded)
    
    return docs
