.PHONY: install test lint typecheck docs serve-docs \
        experiments reproduce clean help

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync --all-extras

# ── Quality gates ─────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ tests/ experiments/
	uv run ruff format --check src/ tests/ experiments/

typecheck:
	uv run mypy src/sfm

format:
	uv run ruff format src/ tests/ experiments/
	uv run ruff check --fix src/ tests/ experiments/

check: lint typecheck test

# ── Documentation ─────────────────────────────────────────────────────────────

docs:
	uv run mkdocs build --strict

serve-docs:
	uv run mkdocs serve

# ── Experiments ───────────────────────────────────────────────────────────────

# Run all experiments in order, writing results to experiments/*/results/
experiments: e1 e2 e3 e4 e5

e1:
	uv run python experiments/e1_recovery/run.py experiments/e1_recovery/config.yaml

e2:
	uv run python experiments/e2_misspecification/run.py experiments/e2_misspecification/config.yaml

e3:
	uv run python experiments/e3_identifiability/run.py experiments/e3_identifiability/config.yaml

e4:
	uv run python experiments/e4_calibration/run.py experiments/e4_calibration/config.yaml

e5:
	uv run python experiments/e5_scaling/run.py experiments/e5_scaling/config.yaml

# Regenerate all figures from stored results
figures:
	for dir in experiments/e*/; do \
	  if [ -f "$$dir/plot.py" ]; then \
	    uv run python $$dir/plot.py $$dir/results/ $$dir/figures/; \
	  fi; \
	done

# Full reproduce: run experiments then regenerate figures
reproduce: experiments figures

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__ \
	       src/sfm/__pycache__ tests/__pycache__ site/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install      Install all dependencies (uv sync)"
	@echo "  test         Run test suite"
	@echo "  lint         Ruff lint + format check"
	@echo "  typecheck    mypy --strict on src/sfm"
	@echo "  format       Auto-format with ruff"
	@echo "  check        lint + typecheck + test"
	@echo "  docs         Build MkDocs site"
	@echo "  serve-docs   Serve docs locally"
	@echo "  experiments  Run all experiments (e1–e5)"
	@echo "  reproduce    Reproduce all figures from scratch"
	@echo "  clean        Remove generated artefacts"
