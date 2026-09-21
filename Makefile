PYTHON ?= python3
VENV_PYTHON := .venv/bin/python
NETWORK ?= Facebook

.PHONY: setup run-comparison run-comparison-help run-kcore-ci run-kcore-ci-help run-lfr run-lfr-help

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run-comparison:
	$(VENV_PYTHON) scripts/real_networks_comparison.py --network $(NETWORK)

run-comparison-help:
	$(VENV_PYTHON) scripts/real_networks_comparison.py --help

run-kcore-ci:
	$(VENV_PYTHON) scripts/kcore_ci_baselines.py

run-kcore-ci-help:
	$(VENV_PYTHON) scripts/kcore_ci_baselines.py --help

run-lfr:
	$(VENV_PYTHON) scripts/lfr_experiments.py

run-lfr-help:
	$(VENV_PYTHON) scripts/lfr_experiments.py --help
