PYTEST := poetry run pytest
FORMATTER := poetry run ruff format
LINTER := poetry run ruff check
TYPE_CHECKER := poetry run mypy
SPHINX_APIDOC := poetry run sphinx-apidoc


PROJECT_DIR := scikit_quri
TEST_DIR := tests
CHECK_DIR := $(PROJECT_DIR) $(TEST_DIR)

COVERAGE_OPT := --cov scikit_quri --cov-branch
BENCHMARK_OPT := --benchmark-autosave -v
PORT := 8000

# Idiom found at https://www.gnu.org/software/make/manual/html_node/Force-Targets.html
FORCE:

.PHONY: fix
fix:
	$(FORMATTER) $(CHECK_DIR)
	$(LINTER) $(CHECK_DIR) --fix

.PHONY: check
check:
	$(FORMATTER) $(CHECK_DIR) --diff
	$(LINTER) $(CHECK_DIR) --diff
# $(TYPE_CHECKER) $(CHECK_DIR)

.PHONY: test
test:
	$(PYTEST) -v $(TEST_DIR)

tests/%.py: FORCE
	$(PYTEST) $@

.PHONY: cov
cov:
	$(PYTEST) $(COVERAGE_OPT) --cov-report html $(TEST_DIR)

.PHONY: cov_ci
cov_ci:
	$(PYTEST) $(COVERAGE_OPT) --cov-report xml $(TEST_DIR)

.PHONY: serve_cov
serve_cov: cov
	poetry run python -m http.server --directory htmlcov $(PORT)

.PHONY: html
html:
	poetry run $(MAKE) -C docs html