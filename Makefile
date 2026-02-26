.PHONY: help setup install prepare train register predict evaluate test format all clean

VENV ?= .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  setup       Create virtual environment and install dependencies"
	@echo "  install     Install dependencies from requirements.txt"
	@echo "  prepare     Run data preparation script"
	@echo "  train       Train the model"
	@echo "  register    Register best model to MLflow"
	@echo "  inference   Run inference with latest model"
	@echo "  evaluate    Send RMSE to Grafana via Prometheus"
	@echo "  test        Run unit tests with pytest"
	@echo "  format      Format code with black and check style with flake8"
	@echo "  all         Run the full pipeline: prepare → train → register → predict"
	@echo "  clean       Remove virtualenv and Python cache files"
	@echo ""

setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

prepare:
	PYTHONPATH=. $(PYTHON) src/data_preparation.py

train:
	PYTHONPATH=. $(PYTHON) src/models/train.py

register:
	PYTHONPATH=. $(PYTHON) src/models/register.py

inference:
	PYTHONPATH=. $(PYTHON) src/inference.py

evaluate:
	PYTHONPATH=. $(PYTHON) src/models/evaluate.py

test:
	PYTHONPATH=. $(PYTHON) -m pytest

format:
	PYTHONPATH=. $(PYTHON) -m black src
	PYTHONPATH=. $(PYTHON) -m flake8 src

all: prepare train register predict

clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
