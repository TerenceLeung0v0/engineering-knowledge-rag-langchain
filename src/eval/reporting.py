from __future__ import annotations
from typing import Iterable
from src.eval.result_types import EvalResult

def summarize(eval_results: Iterable[EvalResult]) -> str:
    results = list(eval_results)
    total = len(results)
    passed = sum(1 for r in results if (r.status_ok and r.sources_ok and r.hygiene_ok))
    rate = (passed / total * 100.0) if total else 0.0

    failed = [r for r in results if not (r.status_ok and r.sources_ok and r.hygiene_ok)]

    lines: list[str] = []
    lines.append(f"Total: {total}")
    lines.append(f"Passed: {passed}")
    lines.append(f"Pass rate: {rate:.2f}%")
    lines.append("")

    if failed:
        lines.append("Failures:")
        for r in failed:
            reasons = []
            if not r.status_ok:
                reasons.append(f"status({r.expect_status}->{r.actual_status})")
            if not r.sources_ok:
                reasons.append("sources")
            if not r.hygiene_ok:
                reasons.append("hygiene")
            lines.append(f"- {r.id}: " + ", ".join(reasons))
    else:
        lines.append("Failures: none")

    return "\n".join(lines) + "\n"
