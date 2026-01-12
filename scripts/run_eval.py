from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schemas import RetrievalStatusEnum
from src.rag.chain import build_rag_chain
from src.utils.artifacts import load_jsonl

import argparse
import json
import re

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```")
_SOURCES_HEADER_RE = re.compile(r"^\s*sources\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_LABEL_LINE_RE = re.compile(r"^\s*(Answer|Context|Question)\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"^\s*(None|N/A|\(none\)|-)\s*$", re.IGNORECASE | re.MULTILINE)
_EMPTY_EXAMPLES_HEADER_RE = re.compile(r"^\s*Examples\s*:\s*$", re.IGNORECASE | re.MULTILINE)

@dataclass
class HygieneResult:
    ok: bool
    flags: dict[str, bool]

def _validate_retrieval_status(status: Any) -> str:
    if status not in {item.value for item in RetrievalStatusEnum}:
        return "unknown"
    
    return status

def _validate_match(
    expect: str,
    actual: str
) -> bool:
    if expect == actual:
        return True
    
    if expect == "ambiguous_or_ok":
        return actual in (RetrievalStatusEnum.AMBIGUOUS, RetrievalStatusEnum.OK)
        
    if expect == "refuse_or_ok":
        return actual in (RetrievalStatusEnum.REFUSE, RetrievalStatusEnum.OK)
        
    return False

def _check_hygiene(answer: str) -> HygieneResult:
    text = (answer or "").strip()
    flags = {
        "has_markdown_heading": bool(_MD_HEADING_RE.search(text)),
        "has_code_fence": bool(_CODE_FENCE_RE.search(text)),
        "has_sources_header": bool(_SOURCES_HEADER_RE.search(text)),
        "has_label_lines": bool(_LABEL_LINE_RE.search(text)),
        "has_placeholders": bool(_PLACEHOLDER_RE.search(text)),
        "ends_with_empty_examples_header": bool(_EMPTY_EXAMPLES_HEADER_RE.search(text)) and ("-" not in text),
    }
    ok = not any(flags.values())
    
    return HygieneResult(
        ok=ok,
        flags=flags
    )

def _get_first_option(result: dict[str, Any]) -> dict[str, Any] | None:
    options = result.get("options") or []
    
    if not options:
        return None

    chosen_id = min(opt.option_id for opt in options)
    return {
        "selected_option": chosen_id,
        "options": options,
    }

def run_one(
    chain,
    item: dict[str, Any],
    auto_resolve_ambiguous: bool
) -> dict[str, Any]:
    query = item["query"]
    expect = item.get("expect_status", "ok")
    min_sources = int(item.get("min_sources", 0))

    result = chain.invoke({"input": query})
    actual = _validate_retrieval_status(result.get("status"))

    resolved = False
    if auto_resolve_ambiguous and actual == "ambiguous":
        payload = _get_first_option(result)
        if payload is not None:
            result = chain.invoke({"input": query, **payload})
            actual = _validate_retrieval_status(result.get("status"))
            resolved = True

    answer = result.get("answer", "") or ""
    docs = result.get("source_documents", []) or []
    num_sources = len(docs)

    hygiene = _check_hygiene(answer)

    status_ok = _validate_match(expect, actual)
    sources_ok = True
    if actual == "ok" and min_sources > 0:
        sources_ok = num_sources >= min_sources
    if actual == "refuse":
        sources_ok = (num_sources == 0)

    passed = status_ok and sources_ok and hygiene.ok

    return {
        "id": item.get("id", ""),
        "query": query,
        "expect_status": expect,
        "actual_status": actual,
        "auto_resolved": resolved,
        "min_sources": min_sources,
        "num_sources": num_sources,
        "passed": passed,
        "status_ok": status_ok,
        "sources_ok": sources_ok,
        "hygiene_ok": hygiene.ok,
        "hygiene_flags": hygiene.flags,
        "refusal_reason": result.get("refusal_reason"),
        "notes": item.get("notes", ""),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=str, required=True, help="Path to QA JSONL")
    ap.add_argument("--out", type=str, default="reports/eval_mini_v0.jsonl", help="Output JSONL path")
    ap.add_argument("--auto-resolve-ambiguous", action="store_true", help="If ambiguous, auto pick option 1 and re-run")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(qa_path)
    chain = build_rag_chain()

    total = 0
    passed = 0
    by_status = {"ok": 0, "refuse": 0, "ambiguous": 0, "unknown": 0}
    fails: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            total += 1
            row = run_one(chain, item, auto_resolve_ambiguous=args.auto_resolve_ambiguous)
            by_status[row["actual_status"]] = by_status.get(row["actual_status"], 0) + 1
            if row["passed"]:
                passed += 1
            else:
                fails.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_path.with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Pass rate: {passed/total:.2%}\n\n")
        f.write("Actual status counts:\n")
        for k, v in by_status.items():
            f.write(f"- {k}: {v}\n")
        f.write("\nTop failures (first 10):\n")
        for row in fails[:10]:
            f.write(f"- {row['id']} | expect={row['expect_status']} actual={row['actual_status']} "
                    f"status_ok={row['status_ok']} sources_ok={row['sources_ok']} hygiene_ok={row['hygiene_ok']}\n")

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {summary_path}")
    print(f"Pass rate: {passed/total:.2%}")
    
    if fails:
        print("Some tests failed. See summary for first 10 fails.")

if __name__ == "__main__":
    main()
