# One documented command per thing you might want to do.
# Dependency management is uv; `uv run` resolves the environment itself, so there
# is nothing to activate and no PYTHONPATH to export.

.PHONY: help setup validate demo replay propose test lock clean

help:
	@echo "cp .env.example .env  - configure first (no exports needed)"
	@echo "make setup     - create the environment from uv.lock"
	@echo "make validate  - check config, creds, dataset, evaluators; free"
	@echo "make demo      - THE DEMO: run the full before/after loop (costs money)"
	@echo "make replay    - render the committed run, no AWS calls, \$$0"
	@echo "make propose   - regenerate prompts/improved.md from baseline findings"
	@echo "make test      - unit tests (no AWS calls)"
	@echo "make lock      - re-resolve dependencies and update uv.lock"

setup:
	uv sync
	@echo "ready. now: cp .env.example .env && make validate"

validate:
	uv run cs-eval validate

demo:
	uv run cs-eval demo --mode live

replay:
	uv run cs-eval demo --mode replay

propose:
	uv run cs-eval propose

test:
	uv run pytest -q

lock:
	uv lock --upgrade

clean:
	rm -rf .pytest_cache **/__pycache__
