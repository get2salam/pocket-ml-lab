.PHONY: install test lint clean demo-classification demo-regression reports help

install:
	pip install -e ".[dev]"

test:
	pytest -v

test-cov:
	pytest --cov=pocket_ml_lab --cov-report=term-missing

lint:
	python -m py_compile pocket_ml_lab/*.py pocket_ml_lab/**/*.py
	python -c "import pocket_ml_lab; print('Import OK')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ reports/

demo-classification:
	pocket-ml profile examples/iris_small.csv --target species
	pocket-ml run examples/iris_small.csv --target species --task classification

demo-regression:
	pocket-ml profile examples/house_prices_small.csv --target price
	pocket-ml run examples/house_prices_small.csv --target price --task regression

reports:
	mkdir -p reports
	pocket-ml run examples/iris_small.csv --target species --task classification --out reports/
	pocket-ml run examples/house_prices_small.csv --target price --task regression --out reports/
	@echo "Reports written to reports/"

help:
	@echo "Pocket ML Lab — available targets:"
	@echo "  install              Install package and dev dependencies"
	@echo "  test                 Run the test suite"
	@echo "  test-cov             Run tests with coverage report"
	@echo "  lint                 Syntax-check all Python source files"
	@echo "  clean                Remove build artifacts and caches"
	@echo "  demo-classification  Run the iris classification demo"
	@echo "  demo-regression      Run the house-prices regression demo"
	@echo "  reports              Generate all reports into reports/"
