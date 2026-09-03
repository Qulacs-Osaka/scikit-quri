PYTEST := uv run pytest
# OQTOPUS tests reach the cloud and need an account, so they are not part of the
# default run. Use `make test-oqtopus` when an account is configured.
DESELECT_OQTOPUS := -m "not oqtopus"
FORMATTER := uv run ruff format
LINTER := uv run ruff check
TYPE_CHECKER := uv run mypy
SPHINX_APIDOC := uv run sphinx-apidoc


PROJECT_DIR := scikit_quri
TEST_DIR := tests
# The vendored adapter is source we ship, so it is linted too. Leaving it out
# meant an unused import there was invisible to both `make check` and CI.
VENDOR_DIR := quri_parts_scaluq
CHECK_DIR := $(PROJECT_DIR) $(TEST_DIR) $(VENDOR_DIR)

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
	$(PYTEST) -v $(DESELECT_OQTOPUS) $(TEST_DIR)

.PHONY: test-oqtopus
test-oqtopus:
	$(PYTEST) -v -m oqtopus $(TEST_DIR)

tests/%.py: FORCE
	$(PYTEST) $@

.PHONY: cov
cov:
	$(PYTEST) $(COVERAGE_OPT) $(DESELECT_OQTOPUS) --cov-report html $(TEST_DIR)

.PHONY: cov_ci
cov_ci:
	$(PYTEST) $(COVERAGE_OPT) $(DESELECT_OQTOPUS) --cov-report xml $(TEST_DIR)

.PHONY: serve_cov
serve_cov: cov
	uv run python -m http.server --directory htmlcov $(PORT)

.PHONY: html
html:
	uv run $(MAKE) -C docs html