PY ?= .venv/bin/python
MODEL ?= qwen3:8b
TAG ?= $(subst :,-,$(MODEL))

.PHONY: install test lint eval eval-mock live-tests report clean

install:            ## create .venv and install with dev extras
	python3 -m venv .venv && $(PY) -m pip install -e ".[dev]"

test:               ## unit tests + eval suite on the deterministic mock (what CI runs)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

eval:               ## live eval against Ollama: make eval MODEL=qwen3:8b TAG=baseline
	$(PY) -m llm_eval_lab run --backend ollama --model $(MODEL) --tag $(TAG)

eval-mock:
	$(PY) -m llm_eval_lab run --backend mock --tag mock

live-tests:         ## pytest eval suite against the real model, saving the run
	$(PY) -m pytest tests/evals --backend ollama --model $(MODEL) --save-run --tag $(TAG)

report:             ## compare the two most recent runs (or RUNS="runs/a.json runs/b.json")
	$(PY) -m llm_eval_lab report $(RUNS)

clean:
	rm -f runs/*.json reports/*.md reports/*.html reports/*.xml
