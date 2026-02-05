from __future__ import annotations
from pathlib import Path

from langchain_core.documents import Document

import re

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

def _strip_front_matter(text: str) -> str:
    return _FRONT_MATTER_RE.sub("", text or "")

def _read_markdown_file(
    path: Path | str,
    *, strip_front_matter: bool=True
) -> Document:
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    if strip_front_matter:
        text = _strip_front_matter(text)

    meta = {
        "source": str(path),
        "page": None,
        "page_label": None,
        "filetype": "md",
        "doc_type": "md"
    }

    return Document(page_content=text, metadata=meta)

def load_mds_documents(
    path: Path | str,
    *,
    glob_pattern: str = "**/*.md",
    strip_front_matter: bool = True,
) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    
    if not path.exists():
        raise FileNotFoundError(f"MD directory no found: {path}")

    mds = [md for md in path.glob(glob_pattern) if md.is_file()]
    if not mds:
        raise RuntimeError(f"No MD documents matching '{glob_pattern}' found in {path}")
    
    for md_path in sorted(mds):
        docs.append(
            _read_markdown_file(
                md_path,
                strip_front_matter=strip_front_matter
            )
        )

    return docs
