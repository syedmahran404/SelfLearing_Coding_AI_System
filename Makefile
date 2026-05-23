.PHONY: help install install-backend install-frontend dev up down logs ps \
        test test-backend test-frontend lint format typecheck \
        migrate seed clean nuke

help:
	@echo "Self-Learning Coding AI — common commands"
	@echo ""
	@echo "  make install            install backend + frontend deps"
	@echo "  make up                 start the full stack (docker compose)"
	@echo "  make down               stop the stack"
	@echo "  make logs               tail backend logs"
	@echo "  make dev                run backend hot-reload (no docker)"
	@echo "  make test               run all tests"
	@echo "  make lint               ruff + eslint"
	@echo "  make typecheck          mypy + tsc"
	@echo "  make format             black + prettier"
	@echo "  make migrate            run db migrations"
	@echo "  make clean              remove caches"
	@echo "  make nuke               clean + drop docker volumes"

install: install-backend install-frontend

install-backend:
	cd backend && python -m pip install -e ".[dev]"

install-frontend:
	cd frontend && npm install

dev:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up -d --build

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f backend

test: test-backend

test-backend:
	cd backend && pytest -q

test-frontend:
	cd frontend && npm test --silent

lint:
	cd backend && ruff check app tests
	cd frontend && npm run lint --silent || true

format:
	cd backend && ruff format app tests
	cd frontend && npm run format --silent || true

typecheck:
	cd backend && mypy app
	cd frontend && npm run typecheck --silent || true

migrate:
	cd backend && python -m app.db.migrate

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	rm -rf backend/.coverage backend/htmlcov

nuke: clean
	docker compose down -v
	rm -rf infra/volumes
