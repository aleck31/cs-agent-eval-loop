# One documented command per thing you might want to do.
# Dependency management is uv; `uv run` resolves the environment itself, so there
# is nothing to activate and no PYTHONPATH to export.

.PHONY: help setup evaluators validate demo replay replay-model propose stability rejudge test lock clean

help:
	@echo "cp .env.example .env  - configure first (no exports needed)"
	@echo "make setup     - create the environment from uv.lock"
	@echo "make evaluators - create/update the custom red-line judge in your account"
	@echo "make validate  - check config, creds, dataset, evaluators; free"
	@echo "make demo      - THE DEMO: run the full before/after loop (costs money)"
	@echo "make replay    - render the committed prompt change, no AWS calls, \$$0"
	@echo "make replay-model - render the committed model comparison (Sonnet 5.5 vs Opus 5.5), \$$0"
	@echo "make propose   - regenerate prompts/improved.md from dev-case findings"
	@echo "make stability - re-score saved transcripts to measure judge noise (costs money)"
	@echo "make rejudge   - fill missing red-line judge votes on saved runs (costs money)"
	@echo "make test      - unit tests (no AWS calls)"
	@echo "make lock      - re-resolve dependencies and update uv.lock"

setup:
	uv sync
	@echo "ready. now: cp .env.example .env && make validate"

evaluators:
	uv run cs-eval setup

validate:
	uv run cs-eval validate

demo:
	uv run cs-eval demo --mode live

replay:
	uv run cs-eval demo --mode replay

replay-model:
	uv run cs-eval report --before baseline --after baseline-opus-5-5

propose:
	uv run cs-eval propose

stability:
	uv run cs-eval stability

rejudge:
	uv run cs-eval rejudge

test:
	uv run pytest -q

lock:
	uv lock --upgrade

clean:
	rm -rf .pytest_cache **/__pycache__
