.PHONY: help setup install ingest train-all train-churn train-revenue api dashboard test format lint clean

help:
	@echo ""
	@echo "  Churn Platform — Commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install        Install dependencies"
	@echo "  make ingest         Download + ingest datasets"
	@echo "  make train-all      Train all ML models"
	@echo "  make train-churn    Train churn model only"
	@echo "  make api            Run FastAPI server"
	@echo "  make dashboard      Run Streamlit dashboard"
	@echo "  make test           Run tests"
	@echo "  make format         Format code (black + isort)"
	@echo "  make lint           Lint code (flake8)"
	@echo "  make clean          Clean cache files"
	@echo ""

install:
	pip install -r requirements.txt

ingest:
	python data_ingestion/ingest_telco.py
	python data_ingestion/ingest_ecommerce.py

train-all:
	python ml_training/churn/train_churn.py
	python ml_training/revenue/train_revenue.py
	python ml_training/segmentation/train_segmentation.py
	python ml_training/anomaly/train_anomaly.py

train-churn:
	python ml_training/churn/train_churn.py

train-revenue:
	python ml_training/revenue/train_revenue.py

train-segment:
	python ml_training/segmentation/train_segmentation.py

train-anomaly:
	python ml_training/anomaly/train_anomaly.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --tb=short

format:
	black . --line-length 88
	isort . --profile black

lint:
	flake8 . --max-line-length 88 --exclude=.venv,__pycache__,dbt_project/target

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true