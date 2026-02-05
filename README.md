# engineering-knowledge-rag-langchain

A document-grounded Retrieval-Augmented Generation (RAG) system for
engineering and IoT documentation, built with LangChain and FAISS.

This project focuses on **retrieval correctness, explicit gating,
and evaluation**, rather than open-domain question answering.

---

## Scope

- Domain-specific RAG (IoT / MQTT / AWS IoT / engineering documentation)
- Answer **only when supported by retrieved documents**
- Explicit refusal for out-of-scope or unsupported questions
- Ambiguity surfaced to the user instead of hallucinated answers

---

## Key Design Goals

- Deterministic retrieval behavior
- Explicit, multi-stage retrieval gating and guardrails
- No hallucination by construction
- Clear separation between:
  - retrieval
  - gating
  - answer generation
  - output hygiene
  - evaluation

---

## Current Features

- FAISS vectorstore with document-level metadata
- L2 distance retrieval with:
  - absolute gate (hard and soft)
  - density gate
  - confidence gap gate
- Out-of-domain (OOD) gate with explicit allow / deny patterns
- Coverage gate to ensure retrieved documents actually support queried entities
- Injection-aware entity tagging at ingestion time
- Ambiguous retrieval handling with explicit selectable options
- Strict refusal policy when no evidence is available
- Output hygiene pipeline enforcement:
  - no empty answers
  - explicit refusal reasons
  - structured ambiguity output
- CLI-friendly and scriptable evaluation workflow

---

## High-Level Architecture

Query flow:
1. Out-of-domain (OOD) gate
2. Vector retrieval (FAISS, L2 distance)
3. Retrieval gating:
  - absolute distance
  - density
  - confidence gap
4. Ambiguity resolution:
  - entity-aware grouping
  - score gap resolution
  - embedding-based tie-breakers
5. Coverage gate (entity support validation)
6. Answer generation (only for `ok` state)
7. Output hygiene enforcement

Each stage can short-circuit the pipeline.
LLM execution is skipped unless retrieval is valid.

---

## Project Status

- [x] Document ingestion & vectorstore
- [x] Retrieval gating logic
- [x] Out-of-domain (OOD) gating
- [x] Coverage gating (entity-aware)
- [x] Injection-time entity tagging
- [x] Ambiguous retrieval handling (auto-resolve + user-selectable paths)
- [x] Output hygiene enforcement
- [x] CLI interface
- [x] Evaluation harness (mini set) + baseline report [`v0.1.0` (~77% pass rate)]
- [x] Baseline stabilized (31/31 pass, 100%) [`v0.2.0` (100% pass rate)]
- [x] QA set expanded (31 -> 60 curated cases) [`v0.3.0` (100% pass rate)]
- [x] Guardrails milestone (OOD + coverage + tagging) [`v0.4.0`]
- [x] Ambiguity policy documented (see below)
- [ ] Ambiguity stress-testing with larger document corpus
- [x] Non-PDF ingestion (HTML)
- [x] Markdown ingestion (CI / Fixture corpus)
- [ ] Non-PDF ingestion (Markdown)
- [ ] LoRA fine-tuning (planned, not started)

---

## Quick Start

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate
pip install -e .

# 2. Download documents
python scripts/download_docs.py

# 3. Ingest documents (Required before any non-CI evaluation)
python scripts/ingest.py

# 4. Run evaluation
make eval-mini-strict
make eval-dev-strict

```

---

## Evaluation

### Current Results
| Dataset            | Pass Rate  |
|--------------------|------------|
| eval-dev (v2)      | **100%**   |
| eval-mini-strict   | **100%**   |

> Note: `*-strict` is intended as a **policy gate** (e.g. CI blocking), not a metric of model quality.

### What is evaluated
- **status_ok**
  Expected retrieval outcome: `ok` / `refuse` / `ambiguous`
- **source_ok**
  Expected source documents appear in retrieved evidence if applicable
- **hygiene_ok**
  Outcome correctness rules:
  - no hallucinated answers
  - refusal must include reason
  - answers must not be empty
  - ambiguity must be explicit if applicable

A case is considered **passed only if all checks pass**.

### Running evaluation
```bash
python scripts/run_eval.py --qa data/curated_qa/<yourTest.jsonl> --out reports/<yourEval.jsonl>
```

Example:
```bash
python scripts/run_eval.py --qa data/curated_qa/qa_mini_v1.jsonl --out reports/eval_mini_v1.jsonl
```

Strict mode (non-zero exit unless 100% pass):
```bash
python scripts/run_eval.py --qa data/curated_qa/<yourTest.jsonl> --out reports/<yourEval.jsonl> --strict
```

### Convenience Scripts
```bash
make eval-mini
make eval-mini-strict
make eval-dev
make eval-dev-strict
```

The evaluation is deterministic and intended for:
- threshold tuning
- guardrail policy refinement
- ambiguity policy refinement
- regression detection

---

## Ambiguity Policy
Ambiguity is treated as a **first-class output**. The system may return `ambiguous` instead of guessing, even when multiple retrieved clusters look relevant.

See: `docs/ambiguity_policy.md`

---

## Data: Source PDFs

The raw PDFs are not committed to this repository (size/licensing).
Download them to `data/raw_docs/` using one of the options below.

### Option A: Manual download
Download and save these files into `data/raw_docs/`:

PDFs:
- mqtt-v3.1.1-os.pdf — MQTT v3.1.1 Standard
  - http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.pdf
- iot-dg.pdf — AWS IoT Developer Guide
  - https://docs.aws.amazon.com/iot/latest/developerguide/iot-dg.pdf
- designing-mqtt-topics-aws-iot-core.pdf — Designing MQTT Topics
  - https://docs.aws.amazon.com/whitepapers/latest/designing-mqtt-topics-aws-iot-core/designing-mqtt-topics-aws-iot-core.pdf
- white-paper-iot-july-2018.pdf — IoT White Paper (2018)
  - https://portail-qualite.public.lu/dam-assets/publications/normalisation/2018/white-paper-iot-july-2018.pdf

HTMLs:
- aws-iot-core-mqtt.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/iot-mqtt.html",
- aws-iot-core-topics.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/topics.html",
- aws-iot-jobs-overview.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs.html",
- aws-iot-jobs-workflows.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/jobs-workflow-jobs-online.html"
- aws-iot-job-execution-states.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/iot-jobs-lifecycle.html",
- aws-iot-thing-groups.html
  - "https://docs.aws.amazon.com/iot/latest/developerguide/thing-groups.html",
- mqtt-v3-1-1-spec.html
  - "https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html",


### Option B: Download via script
Run:
```bash
python scripts/download_docs.py
```

---

## CI (Continuous Integration)

This repository can be configurated with GitHub Actions to automatically run:
- unit tests (if any)
- eval-ci-strict

This prevents regressions in retrieval gating / ambiguity policy when code changes.

CI is intentionally limited to a small, deterministic **fixture corpus** and curated QA cases.
It is a **policy gate**, not a "full-corpus correctness" gate.

CI does NOT:
- ingest the full document corpus
- validate model quality or fluency

Instead, CI runs against:
- a minimal, deterministic fixture corpus
- curated QA cases designed to stress guardrails

Full-corpus ingestion and full evaluation is intended to run locally.

See `.github/workflows/ci.yml`

---

## Non-Goals

This project intentionally does NOT:
- perform open-domain question answering
- optimize for conversational fluency
- auto-resolve ambiguous queries silently
- claim model accuracy beyond document support
- fine-tune models before retrieval behavior is stable

These constraints are deliberate and enforced by design.

---

## Notes

- This system **does not answer general knowledge questions**
- If no relevant documents are retrieved, the system will refuse
- Ambiguity is treated as a first-class output, not an error
- Current QA sets are curated to validate refusal and correctness paths
- True ambiguity is expected to emerge naturally as the document corpus expands
- LoRA is intentionally postponed until retrieval + evaluation stabilize
- Refusal is treated as a correct outcome when document coverage is insufficient

---

## Disclaimer

This project is under active development.
Interfaces, thresholds, and evaluation criteria may change.
