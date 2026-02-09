.PHONY: dev test lint backtest paper live clean install dashboard dashboard-dev rust-build rust-test rust-backtest rust-bench

install:
	pip install -r requirements.txt

dev:
	docker compose up -d

dev-down:
	docker compose down

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -x -q --tb=short

backtest:
	python -m src.cli --backtest

lint:
	ruff check src/ tests/
	mypy src/

lint-fix:
	ruff check --fix src/ tests/

paper:
	python -m src.cli --paper --strategy momentum_confirmation --env paper

paper-momentum:
	python -m src.cli --paper --strategy momentum_confirmation --env paper

paper-singularity:
	python -m src.cli --paper --strategy singularity --env paper

paper-both:
	@echo "Starting Momentum Confirmation (background)..."
	python -m src.cli --paper --strategy momentum_confirmation --env paper &
	@echo "Starting Singularity (foreground)..."
	python -m src.cli --paper --strategy singularity --env paper

live:
	python -m src.cli --live

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

dashboard:
	docker compose build dashboard && docker compose up -d dashboard

dashboard-dev:
	cd dashboard && npm install && npm run dev

rust-build:
	cd rust-backtest && cargo build --release

rust-test:
	cd rust-backtest && cargo test

rust-backtest:
	rust-backtest/target/release/mvhe-backtest --candles data/btc_1m_2y.csv --config config/default.toml --output json

rust-bench:
	cd rust-backtest && cargo bench
