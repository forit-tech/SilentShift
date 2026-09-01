.PHONY: help setup data provenance calibrate main policy attribution control analyse test lint typecheck all clean

PY ?= python
CONFIG ?= configs/default.yaml

help:
	@echo "setup       install the package with dev extras"
	@echo "data        download SMD (466 MB, MIT licence) into data/raw/SMD"
	@echo "calibrate   freeze one threshold per machine on its own clean history"
	@echo "main        score development streams  (~25 min)"
	@echo "policy      reference-window policy study"
	@echo "control     negative control against SMD's labelled anomalies"
	@echo "analyse     build tables and figures from the artefacts"
	@echo "all         calibrate -> main -> policy -> attribution -> control -> analyse"

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	@test -d data/raw/SMD || ( \
	  mkdir -p data/raw && \
	  git clone --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git data/raw/_omni && \
	  git -C data/raw/_omni sparse-checkout set ServerMachineDataset && \
	  mv data/raw/_omni/ServerMachineDataset data/raw/SMD && \
	  rm -rf data/raw/_omni )
	@echo "SMD ready at data/raw/SMD"

provenance:
	$(PY) scripts/run_experiment.py --stage provenance --config $(CONFIG)

calibrate:
	$(PY) scripts/run_experiment.py --stage calibrate --config $(CONFIG)

main:
	$(PY) scripts/run_experiment.py --stage main --split development --config $(CONFIG)

heldout:
	$(PY) scripts/run_experiment.py --stage main --split heldout --config $(CONFIG)

policy:
	$(PY) scripts/run_experiment.py --stage policy --config $(CONFIG)

attribution:
	$(PY) scripts/run_experiment.py --stage attribution --config $(CONFIG)

control:
	$(PY) scripts/run_experiment.py --stage negative_control --config $(CONFIG)

analyse:
	$(PY) scripts/analyse.py --split development --config $(CONFIG)

all: provenance calibrate main policy attribution control analyse

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

typecheck:
	$(PY) -m mypy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
