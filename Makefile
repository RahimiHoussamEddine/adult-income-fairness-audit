PYTHON ?= .venv/bin/python
export PIP_CACHE_DIR := .cache/pip
export TMPDIR := $(CURDIR)/.tmp

.PHONY: install-dev lock fetch build verify test quality check-deps app api

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

lock:
	$(PYTHON) -m pip freeze --all | sed '/^-e /d' > requirements.lock

fetch:
	$(PYTHON) -m fairness_audit.fetch

build:
	$(PYTHON) -m fairness_audit.build

verify:
	$(PYTHON) -m fairness_audit.verify

test:
	$(PYTHON) -m pytest

quality:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

check-deps:
	$(PYTHON) -m pip check

app:
	$(PYTHON) -m streamlit run app.py

api:
	$(PYTHON) -m uvicorn fairness_audit.api:app --host 0.0.0.0 --port 8000
