PY ?= python

.PHONY: install all prepare train contract-demo simulate monitor dashboard test lint mlflow-ui clean

install:            ## instala o projeto + dependências de desenvolvimento
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

all:                ## pipeline completo
	$(PY) -m credit_monitoring all

prepare:
	$(PY) -m credit_monitoring prepare
train:
	$(PY) -m credit_monitoring train
contract-demo:
	$(PY) -m credit_monitoring contract-demo
simulate:
	$(PY) -m credit_monitoring simulate
monitor:
	$(PY) -m credit_monitoring monitor
dashboard:
	$(PY) -m credit_monitoring dashboard

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

mlflow-ui:          ## abre o MLflow em http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf data/processed data/reference data/production data/quarantine logs/*.jsonl mlflow.db mlartifacts
