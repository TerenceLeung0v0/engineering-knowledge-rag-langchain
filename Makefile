.PHONY: eval-mini eval-mini-strict

# base dirs
QA_DIR ?= data/curated_qa
REPORT_DIR ?= reports

# mini set
QA_MINI ?= $(QA_DIR)/qa_mini_v1.jsonl
OUT_MINI ?= $(REPORT_DIR)/eval_mini_v1.jsonl

eval-mini:
	./scripts/eval_mini.sh $(QA_MINI) $(OUT_MINI)

eval-mini-strict:
	./scripts/eval_mini_strict.sh $(QA_MINI) $(OUT_MINI)
