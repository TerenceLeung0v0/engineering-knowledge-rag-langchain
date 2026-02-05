#!/usr/bin/env bash
set -euo pipefail

QA_PATH="data/curated_qa/qa_ci_v1.jsonl"
OUT_PATH="reports/eval_ci_v1.jsonl"

VS_DIR="artifacts/ci_vectorstore"
FAISS_INDEX="$VS_DIR/index.faiss"
FAISS_META="$VS_DIR/index.pkl"

echo "[CI] Ensure fixture vectorstore exists..."
if [ ! -f "$FAISS_INDEX" ] || [ ! -f "$FAISS_META" ]; then
  echo "[CI] Building fixture vectorstore..."
  make ingest-ci
else
  echo "[CI] CI Vectorstore found. Skipping ingest."
fi

echo "[CI] Run eval (strict)..."
python scripts/run_eval.py --qa "$QA_PATH" --out "$OUT_PATH" --strict --ci

echo "[CI] Done."
