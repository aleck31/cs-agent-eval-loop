# One documented command per thing you might want to do.
PY := ./.venv/bin/python
export PYTHONPATH := src

.PHONY: help setup validate demo replay propose test clean

help:
	@echo "make setup     - create .venv and install pinned deps"
	@echo "make validate  - check dataset + evaluators, no judge spend"
	@echo "make demo      - THE DEMO: run the full before/after loop (costs money)"
	@echo "make replay    - render the committed run, no AWS calls, \$$0"
	@echo "make propose   - regenerate prompts/improved.md from baseline findings"
	@echo "make test      - unit tests (no AWS calls)"

setup:
	python3 -m venv .venv
	./.venv/bin/pip install -q --upgrade pip
	./.venv/bin/pip install -q -r requirements.txt
	@echo "ready. now: make validate"

validate:
	$(PY) -m cs_eval.cli validate

demo:
	$(PY) -m cs_eval.cli demo --mode live

replay:
	$(PY) -m cs_eval.cli demo --mode replay

propose:
	$(PY) -m cs_eval.cli propose

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf __pycache__ src/cs_eval/__pycache__ .pytest_cache
