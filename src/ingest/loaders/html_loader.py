from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

@dataclass(frozen=True)
class HtmlDoc:
    path: Path
    text: str
    title: str | None = None

def _read_html(path: Path) -> HtmlDoc:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)

    return HtmlDoc(path=path, text=cleaned, title=title)

def load_htmls_documents(
    path: Path | str,
    *, glob_pattern: str = "**/*.html",
) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    
    htmls = [html for html in path.glob(glob_pattern) if html.is_file()]
    if not htmls:
        raise RuntimeError(f"No HTML documents matching '{glob_pattern}' found in {path}")
    
    for html_path in sorted(htmls):
        html_doc = _read_html(html_path)
        meta = {
            "source": str(html_path),
            "doc_type": "html",
            "title": html_doc.title or html_path.stem,
        }
        docs.append(Document(
            page_content=html_doc.text,
            metadata=meta
        ))

    return docs
