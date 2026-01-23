.PHONY: eval-mini eval-mini-strict eval-dev eval-dev-strict

# base dirs
QA_DIR ?= data/curated_qa
REPORT_DIR ?= reports

# mini set
QA_MINI ?= $(QA_DIR)/qa_mini_v1.jsonl
OUT_MINI ?= $(REPORT_DIR)/eval_mini_v1.jsonl

# dev set
QA_DEV ?=  $(QA_DIR)/qa_dev_v1.jsonl
OUT_DEV ?= $(REPORT_DIR)/eval_dev_v1.jsonl

eval-mini:
	./scripts/eval_mini.sh $(QA_MINI) $(OUT_MINI)

eval-mini-strict:
	./scripts/eval_mini_strict.sh $(QA_MINI) $(OUT_MINI)

eval-dev:
	./scripts/eval_dev.sh $(QA_DEV) $(OUT_DEV)

eval-dev-strict:
	./scripts/eval_dev_strict.sh $(QA_DEV) $(OUT_DEV)
