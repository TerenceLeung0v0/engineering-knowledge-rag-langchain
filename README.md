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
- [ ] Ambiguity stress-testing with larger document corpus
- [ ] Non-PDF ingestion (HTML / Markdown)
- [ ] LoRA fine-tuning (planned, not started)

---

## Evaluation

### Current Results
| Dataset            | Pass Rate  |
|--------------------|------------|
| eval-dev           | **100%**   |
| eval-mini-strict   | **88.33%** |

- The drop in `eval-mini-strict` is expected and reflects stricter OOD and coverage guardrails introduced in v0.4.0. Several previously accepted queries are now correctly refused due to missing document coverage.


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

## Data: Source PDFs

The raw PDFs are not committed to this repository (size/licensing).
Download them to `data/raw_docs/` using one of the options below.

### Option A: Manual download
Download and save these files into `data/raw_docs/`:

- mqtt-v3.1.1-os.pdf — MQTT v3.1.1 Standard
  - http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.pdf
- iot-dg.pdf — AWS IoT Developer Guide
  - https://docs.aws.amazon.com/iot/latest/developerguide/iot-dg.pdf
- designing-mqtt-topics-aws-iot-core.pdf — Designing MQTT Topics
  - https://docs.aws.amazon.com/whitepapers/latest/designing-mqtt-topics-aws-iot-core/designing-mqtt-topics-aws-iot-core.pdf
- white-paper-iot-july-2018.pdf — IoT White Paper (2018)
  - https://portail-qualite.public.lu/dam-assets/publications/normalisation/2018/white-paper-iot-july-2018.pdf


### Option B: Download via script
Run:
```bash
python scripts/download_docs.py
```

## Notes

- This system **does not answer general knowledge questions**
- If no relevant documents are retrieved, the system will refuse
- Ambiguity is treated as a first-class output, not an error
- Current QA sets are curated to validate refusal and correctness paths
- True ambiguity is expected to emerge naturally as the document corpus expands
- Absence of ambiguous cases reflects current dataset design, not a system limitation
- LoRA is intentionally postponed until retrieval + evaluation stabilize
- Refusal is treated as a correct outcome when document coverage is insufficient

---

## Disclaimer

This project is under active development.
Interfaces, thresholds, and evaluation criteria may change.
