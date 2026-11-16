.PHONY: help install dev test test-fast review run split docker-build docker-up clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

help:
	@echo "ClipBot Management Commands:"
	@echo "  make dev          Install package in editable mode with dev/test extras"
	@echo "  make test         Run zero-mock integration test suite"
	@echo "  make test-fast    Run fast unit tests"
	@echo "  make review       Launch FastAPI review dashboard on http://localhost:8080"
	@echo "  make run          Run highlight pipeline (usage: make run URL=<video_url> [PROFILE=gta6_chaos])"
	@echo "  make split        Run movie splitter (usage: make split URL=<video_url> [TITLE=Movie])"
	@echo "  make docker-build Build production Docker container image"
	@echo "  make docker-up    Start Docker Compose background services"
	@echo "  make clean        Remove temporary files, caches, and build artifacts"

dev:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test]"

test:
	$(PYTEST) tests/ -v --durations=10

test-fast:
	$(PYTEST) tests/test_models.py tests/test_scoring.py -v

review:
	$(PYTHON) -m clipbot.cli review --host 0.0.0.0 --port 8080

run:
	@if [ -z "$(URL)" ]; then echo "Error: URL is required. Example: make run URL=https://... [PROFILE=general_viral] [MAX_CLIPS=3]"; exit 1; fi
	$(PYTHON) -m clipbot.cli clip-url "$(URL)" $(if $(PROFILE),--profile "$(PROFILE)",) $(if $(MAX_CLIPS),--max-clips "$(MAX_CLIPS)",)

split:
	@if [ -z "$(URL)" ]; then echo "Error: URL is required. Example: make split URL=https://... [TITLE='Movie Title']"; exit 1; fi
	$(PYTHON) -m clipbot.cli split-movie "$(URL)" $(if $(TITLE),--title "$(TITLE)",)

docker-build:
	docker compose -f deploy/docker-compose.yml build

docker-up:
	docker compose -f deploy/docker-compose.yml up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov dist build
