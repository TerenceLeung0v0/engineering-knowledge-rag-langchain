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
- Explicit retrieval gating (L2 distance–based)
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
- [x] Ambiguous retrieval handling (logic implemented; stress cases pending)
- [x] Output hygiene enforcement
- [x] CLI interface
- [x] Evaluation harness (mini set) + baseline report [`v0.1.0` (~77% pass rate)]
- [x] Baseline stabilized (31/31 pass, 100%) [`v0.2.0` (100% pass rate)]
- [x] QA set expanded (31 -> 60 curated cases) [`v0.3.0` (100% pass rate)]
- [ ] Ambiguity stress-testing with larger document corpus
- [ ] LoRA fine-tuning (planned, not started)

---

## Evaluation

### Current Results
- Total test cases: **60**
- Passed: **60**
- Pass rate: **100%**

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
python scripts/run_eval.py --qa data/curated_qa/qa_mini_v0.jsonl --out reports/eval_mini_v0.jsonl
```

Strict mode (non-zero exit unless 100% pass):
```bash
python scripts/run_eval.py --qa data/curated_qa/<yourTest.jsonl> --out reports/<yourEval.jsonl> --strict
```

### Convenience Scripts
```bash
make eval-mini
make eval-mini-strict
```

The evaluation is deterministic and intended for:
- threshold tuning
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

---

## Disclaimer

This project is under active development.
Interfaces, thresholds, and evaluation criteria may change.
