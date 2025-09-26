.PHONY: help install test lint format clean run-app docker-build docker-run

help:
	@echo "Available commands:"
	@echo "	install				Install dependencies"
	@echo "	test				Run tests"
	@echo " lint				Run linting"
	@echo " format				Format code"
	@echo " clean				Clean cache files"
	@echo " run-app				Run Gradio app"
	@echo " docker-build		Build Docker image"
	@echo " docker-run			Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirements/dev.txt
	pre-commit install

test:
	pytest test/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache

run-app:
	python app/gradio_app.py

docker-build:
	docker build -f docker/Dockerfile -t sentiment-mlops:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up
# 	docker run -p 7860:7860 sentiment-mlops:latest

docker-teardown:
	docker-compose -f docker/docker-compose.yml down

train:
	python scripts/train.py --config configs/training_config.yaml

evaluate:
	python scripts/evaluate.py --model-path models/latest

deploy:
	python scripts/deploy.py --stage production
