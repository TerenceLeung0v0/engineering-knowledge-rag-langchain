from pathlib import Path
from collections import defaultdict
from langchain_core.documents import Document

from src.startup_checks import ensure_project_dirs, check_vectorstore_ready
from src.schemas import RetrievalOption, RetrievalStatusEnum
from src.rag.chain import build_rag_chain
from src.rag.formatting import collect_sources, normalize_page

import re

_WS_RE = re.compile(r"\s+")

def _truncate_text(
    text: str,
    max_chars: int=140
) -> str:
    snippet = _WS_RE.sub(" ", (text or "").strip())
    
    if len(snippet) <= max_chars:
        return snippet
    
    return snippet[: max_chars - 1] + "…"

def _preview_line(
    doc: Document,
    *, max_chars: int=140
) -> str:
    meta = doc.metadata or {}
    src = Path(meta.get("source", "unknown")).name
    page = normalize_page(meta.get("page"))
    text = _truncate_text(
        doc.page_content,
        max_chars=max_chars
    )
    
    return f"{text} ({src}, page {page})"

def _top_previews(
    option: RetrievalOption,
    *, max_chars: int=140
) -> tuple[str, str]:
    if not option.docs:
        return "", ""
    
    p1 = _preview_line(
        option.docs[0],
        max_chars=max_chars
    )

    p2 = _preview_line(
        option.docs[1],
        max_chars=max_chars
    ) if len(option.docs) >= 2 else ""
    
    return p1, p2

def _to_similarity_score(l2_distance: float) -> float:
    """
    UI-only score. Monotonic transform of L2 distance.
    Higher is better.
    """
    safe_l2 = max(0.0, float(l2_distance))
    
    return 100.0 / (1.0 + safe_l2)

def _print_options(options: list[RetrievalOption]) -> None:
    for option in options:
        pages = ", ".join(str(s.page) for s in option.sources)
        files = sorted(set(s.filename for s in option.sources))
        score = _to_similarity_score(option.best_l2)
        
        p1, p2 = _top_previews(
            option,
            max_chars=140
        )
        
        print(f"- Option {option.option_id}: best_l2={option.best_l2:.3f}, score={score:.1f}")
        if p1:
            print(f"  Preview 1: {p1}")
        if p2:
            print(f"  Preview 2: {p2}")            
            
        print(f"  Files: {', '.join(files)}")  
        print(f"  Pages: {pages}")

def _print_sources_list(
    docs: list[Document],
    group: bool=False
) -> None:
    sources = collect_sources(docs)

    if not sources:
        print("- (none)")
        return

    if group:
        grouped = defaultdict(list)
        for src in sources:
            grouped[src.filename].append(str(src.page))

        for filename, pages in grouped.items():
            pages_str = ", ".join(pages)
            print(f"- [{filename} (pages: {pages_str})]")        
    else:
        for src in sources:
            print(f"- [{src.filename}, (page {src.page})]")

def main():
    ensure_project_dirs()
    check_vectorstore_ready()    
    
    chain = build_rag_chain()
    print("Engineering Knowledge RAG (type 'exit' to quit)\n")

    while True:
        question = input(">> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = chain.invoke({"input": question})
        options = result.get("options")
        
        if result.get("status") == RetrievalStatusEnum.AMBIGUOUS.value and options:
            print("\nAmbiguous Options:")
            _print_options(options)
            
            valid_opt_ids = {opt.option_id for opt in options}
            valid_ids_str = ", ".join(str(i) for i in sorted(valid_opt_ids))
            choice = input(f"\nChoose option ({valid_ids_str}), or 0 to cancel: ").strip()
            
            if choice == "0":
                continue
            if not choice.isdigit():
                print("Invalid input. Cancelled.")
                continue
            
            selected = int(choice)
            if selected not in valid_opt_ids:
                print("Invalid option. Cancelled.")
                continue
            
            print(f"\nSelected option: {selected}\n")
            
            result = chain.invoke({
                "input": question,
                "selected_option": selected,
                "options": options,
            })
       
        docs = result.get("source_documents", [])

        print("\nAnswer:")
        print(result.get("answer", ""))

        if docs:
            print("\nSources:")
            _print_sources_list(docs, True)

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
