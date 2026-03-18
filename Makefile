.DEFAULT_GOAL := help
.PHONY: help install dev lint format typecheck test test-integration test-cov test-all \
	ci clean compose-up compose-down compose-logs

UV := uv

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_.-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies and enable git hooks
	$(UV) sync --all-groups
	$(UV) run pre-commit install --install-hooks
	$(UV) run pre-commit install --hook-type commit-msg

dev: compose-up ## Start local development services

lint: ## Run linters (ruff)
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format: ## Auto-format and auto-fix
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck: ## Run mypy in strict mode
	$(UV) run mypy src tests

test: ## Run unit tests
	$(UV) run pytest -m "not integration"

test-integration: ## Run integration tests
	$(UV) run pytest -m integration

test-cov: ## Run tests with coverage report
	$(UV) run pytest -m "not integration" \
		--cov=alphamind \
		--cov-report=term-missing \
		--cov-report=html

test-all: ## Run the complete test suite
	$(UV) run pytest

ci: lint typecheck test ## Run the full local CI pipeline

clean: ## Remove build artefacts and caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

compose-up: ## Start local services (Postgres, Redis)
	docker compose -f infra/docker-compose.yml up -d

compose-down: ## Stop local services
	docker compose -f infra/docker-compose.yml down

compose-logs: ## Tail local service logs
	docker compose -f infra/docker-compose.yml logs -f
