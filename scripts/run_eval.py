from __future__ import annotations
from pathlib import Path
from typing import Any

from src.eval.io_jsonl import read_jsonl, write_jsonl
from src.eval.schemas import parse_case, validate_cases
from src.eval.runner import run_case
from src.eval.reporting import summarize
from src.rag.chain import build_rag_chain

import argparse

def _as_dict(r: Any) -> dict[str, Any]:
    # EvalResult is dataclass (frozen), so __dict__ works
    return dict(r.__dict__)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="Path to QA jsonl")
    ap.add_argument("--out", required=True, help="Path to output eval jsonl")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if not 100% pass")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of cases (0=all)")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = ap.parse_args()

    qa_path = Path(args.qa)
    out_path = Path(args.out)
    summary_path = out_path.with_suffix(".summary.txt")

    rows = read_jsonl(qa_path)
    cases = validate_cases([parse_case(x) for x in rows])

    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    chain = build_rag_chain()

    results = []
    for case in cases:
        r = run_case(chain, case)
        results.append(r)

        ok = (r.status_ok and r.sources_ok and r.hygiene_ok)
        if args.fail_fast and not ok:
            break

    write_jsonl(out_path, [_as_dict(r) for r in results])

    summary_text = summarize(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")

    passed = sum(1 for r in results if (r.status_ok and r.sources_ok and r.hygiene_ok))
    total = len(results)
    rate = (passed / total * 100.0) if total else 0.0

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {summary_path}")
    print(f"Pass rate: {rate:.2f}%")

    if args.strict and (total == 0 or passed != total):
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
