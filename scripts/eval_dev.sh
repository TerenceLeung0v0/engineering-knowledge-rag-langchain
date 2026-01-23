#!/usr/bin/env bash
set -euo pipefail

QA_PATH="${1:-data/curated_qa/qa_dev_v1.jsonl}"
OUT_PATH="${2:-reports/eval_dev_v1.jsonl}"

python scripts/run_eval.py --qa "${QA_PATH}" --out "${OUT_PATH}"
echo ""
echo "[DONE] Summary:"
cat "${OUT_PATH%.jsonl}.summary.txt"
