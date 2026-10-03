# Confforge developer tasks.
# Every target is runnable from a clean checkout after `pip install -e ".[backend,dev]"`.

PY ?= python

.PHONY: help install install-locked lint format fmt test cov cov-check demo smoke quick gates clean verify

help:
	@echo "Confforge"
	@echo ""
	@echo "  make install      install the package with backend + dev extras"
	@echo "  make install-locked  reproducible install from requirements.lock.txt"
	@echo "  make lint      ruff check (hard gate, mirrors CI)"
	@echo "  make format    ruff format --check"
	@echo "  make fmt       apply ruff format"
	@echo "  make test      run the test suite"
	@echo "  make cov       test suite with a coverage report"
	@echo "  make verify    lint + format + tests + MAPIE self-test"
	@echo "  make demo      full benchmark, writes results/benchmark.json"
	@echo "  make quick     single-dataset smoke test"
	@echo "  make gates     print the release gate definitions"
	@echo "  make clean     remove caches and generated results"

install:
	$(PY) -m pip install -e ".[backend,dev]"

# Reproducible install from the lock file (D4): exactly the combination CI uses.
# `make install-locked` proves the pinned set works, not just the version ranges.
install-locked:
	$(PY) -m pip install -r requirements.lock.txt
	$(PY) -m pip install -e . --no-deps

lint:
	ruff check .

format:
	ruff format --check .

fmt:
	ruff format .

test:
	$(PY) -m pytest -q -W ignore::UserWarning

# D6: enforce the coverage claim the README badge makes.
cov-check:
	$(PY) -m coverage report --fail-under=88

cov:
	$(PY) -m pytest -q --cov --cov-report=term-missing

selftest:
	$(PY) cli.py selftest

verify: lint format test cov-check selftest smoke

demo:
	$(PY) examples/run_demo.py

# Smoke test: asserts the pipeline RAN, not that gates pass. Exits 0 on a
# reduced grid where MIN_PARTICIPANTS makes G1 fail by construction.
smoke:
	$(PY) examples/run_demo.py --datasets homoscedastic,heteroscedastic --smoke

quick:
	$(PY) cli.py quick

gates:
	$(PY) cli.py gates

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
