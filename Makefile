# VideoLingo Makefile
# Development workflow automation and convenience commands

.PHONY: help install test test-fast test-coverage test-gpu lint format clean docs dev setup-dev
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
REPORTS_DIR := tests/reports
VENV_DIR := .venv

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Help target - shows available commands
help: ## 📚 Show this help message
	@echo "$(BLUE)VideoLingo Development Commands$(RESET)"
	@echo "=================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make install     # Install dependencies"
	@echo "  make test-fast   # Run fast tests"
	@echo "  make test-coverage # Run with coverage"
	@echo "  make lint        # Check code quality"

# ============================================================================
# INSTALLATION AND SETUP
# ============================================================================

install: ## 🔧 Install project dependencies
	@echo "$(BLUE)Installing VideoLingo dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]
	@echo "$(GREEN)✅ Dependencies installed successfully$(RESET)"

install-dev: ## 🛠️  Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -e .[dev,gpu,audio,all]
	$(PIP) install tox pre-commit
	pre-commit install
	@echo "$(GREEN)✅ Development environment ready$(RESET)"

setup-dev: install-dev ## 🚀 Complete development environment setup
	@echo "$(BLUE)Setting up complete development environment...$(RESET)"
	mkdir -p $(REPORTS_DIR)
	$(PYTHON) -c "import nltk; nltk.download('punkt')" || true
	$(PYTHON) -c "import spacy; spacy.download('en_core_web_sm')" || true
	@echo "$(GREEN)✅ Development environment setup complete$(RESET)"

# ============================================================================
# TESTING COMMANDS
# ============================================================================

test: ## 🧪 Run default test suite
	@echo "$(BLUE)Running VideoLingo test suite...$(RESET)"
	$(PYTHON) run_tests.py
	@echo "$(GREEN)✅ Tests completed$(RESET)"

test-fast: ## ⚡ Run fast tests only
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(PYTHON) run_tests.py --fast
	@echo "$(GREEN)✅ Fast tests completed$(RESET)"

test-smoke: ## 🚨 Run essential smoke tests
	@echo "$(BLUE)Running smoke tests...$(RESET)"
	$(PYTHON) run_tests.py --smoke
	@echo "$(GREEN)✅ Smoke tests completed$(RESET)"

test-coverage: ## 📈 Run tests with coverage analysis
	@echo "$(BLUE)Running tests with coverage analysis...$(RESET)"
	$(PYTHON) run_tests.py --coverage
	@echo "$(GREEN)✅ Coverage analysis completed$(RESET)"

test-gpu: ## 🖥️  Run GPU-specific tests (requires CUDA)
	@echo "$(BLUE)Running GPU tests...$(RESET)"
	$(PYTHON) run_tests.py --gpu --include-gpu
	@echo "$(GREEN)✅ GPU tests completed$(RESET)"

test-integration: ## 🔗 Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) tests/integration/ tests/e2e/ -v --tb=short -m "integration or e2e"
	@echo "$(GREEN)✅ Integration tests completed$(RESET)"

test-performance: ## 📊 Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) run_tests.py --benchmark --profile
	@echo "$(GREEN)✅ Performance tests completed$(RESET)"

test-all: ## 🎯 Run comprehensive test suite (all tests)
	@echo "$(BLUE)Running comprehensive test suite...$(RESET)"
	$(PYTHON) run_tests.py --coverage --include-slow --include-network --profile
	@echo "$(GREEN)✅ Comprehensive tests completed$(RESET)"

test-parallel: ## ⚡ Run tests in parallel mode
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTEST) -n auto --dist worksteal -v
	@echo "$(GREEN)✅ Parallel tests completed$(RESET)"

test-watch: ## 👁️  Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Starting test watch mode...$(RESET)"
	ptw --runner "$(PYTHON) run_tests.py --fast"

# ============================================================================
# CODE QUALITY AND LINTING
# ============================================================================

lint: ## 🔍 Run all linting checks
	@echo "$(BLUE)Running code quality checks...$(RESET)"
	black --check --diff core/ tests/
	isort --check-only --diff core/ tests/
	flake8 core/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy core/ --ignore-missing-imports --no-strict-optional || true
	pylint core/ --output-format=text || true
	@echo "$(GREEN)✅ Code quality checks completed$(RESET)"

lint-fix: format ## 🔧 Fix linting issues automatically
	@echo "$(GREEN)✅ Linting issues fixed$(RESET)"

format: ## 🎨 Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black core/ tests/
	isort core/ tests/
	@echo "$(GREEN)✅ Code formatting completed$(RESET)"

security: ## 🔒 Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r core/ -f json -o $(REPORTS_DIR)/security_report.json || true
	safety check --json --output $(REPORTS_DIR)/safety_report.json || true
	@echo "$(GREEN)✅ Security checks completed$(RESET)"

# ============================================================================
# MULTI-ENVIRONMENT TESTING
# ============================================================================

tox-test: ## 🐍 Run tests across Python versions with tox
	@echo "$(BLUE)Running tox tests across Python versions...$(RESET)"
	tox
	@echo "$(GREEN)✅ Tox tests completed$(RESET)"

tox-parallel: ## ⚡ Run tox tests in parallel
	@echo "$(BLUE)Running tox tests in parallel...$(RESET)"
	tox --parallel auto
	@echo "$(GREEN)✅ Parallel tox tests completed$(RESET)"

tox-coverage: ## 📈 Run tox coverage tests
	@echo "$(BLUE)Running tox coverage analysis...$(RESET)"
	tox -e coverage
	@echo "$(GREEN)✅ Tox coverage completed$(RESET)"

tox-lint: ## 🔍 Run tox linting checks
	@echo "$(BLUE)Running tox linting...$(RESET)"
	tox -e lint
	@echo "$(GREEN)✅ Tox linting completed$(RESET)"

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

coverage: test-coverage ## 📊 Generate coverage reports
	@echo "$(BLUE)Generating coverage reports...$(RESET)"
	coverage html --directory=$(REPORTS_DIR)/coverage_html
	coverage xml -o $(REPORTS_DIR)/coverage.xml
	coverage json -o $(REPORTS_DIR)/coverage.json
	coverage report --show-missing
	@echo "$(GREEN)✅ Coverage reports generated$(RESET)"

coverage-serve: coverage ## 🌐 Serve coverage report locally
	@echo "$(BLUE)Starting coverage report server...$(RESET)"
	@echo "$(YELLOW)Open http://localhost:8000 to view coverage report$(RESET)"
	cd $(REPORTS_DIR)/coverage_html && $(PYTHON) -m http.server 8000

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## 📖 Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	sphinx-build -W -b html docs/ $(REPORTS_DIR)/docs_html/ || echo "$(YELLOW)Sphinx not available, skipping docs$(RESET)"
	@echo "$(GREEN)✅ Documentation generated$(RESET)"

docs-serve: docs ## 🌐 Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(RESET)"
	@echo "$(YELLOW)Open http://localhost:8001 to view documentation$(RESET)"
	cd $(REPORTS_DIR)/docs_html && $(PYTHON) -m http.server 8001

# ============================================================================
# DEVELOPMENT UTILITIES
# ============================================================================

dev: ## 🚀 Start development environment
	@echo "$(BLUE)Starting development environment...$(RESET)"
	@echo "$(YELLOW)Running smoke tests first...$(RESET)"
	$(PYTHON) run_tests.py --smoke --fail-fast
	@echo "$(GREEN)✅ Development environment ready$(RESET)"
	@echo "$(YELLOW)Available commands: make test-fast, make lint, make format$(RESET)"

profile: ## 📈 Profile application performance
	@echo "$(BLUE)Profiling application performance...$(RESET)"
	$(PYTHON) -m cProfile -o $(REPORTS_DIR)/profile.stats st.py || true
	$(PYTHON) -c "import pstats; p = pstats.Stats('$(REPORTS_DIR)/profile.stats'); p.sort_stats('cumulative').print_stats(20)" || true
	@echo "$(GREEN)✅ Profiling completed$(RESET)"

benchmark: ## 🏁 Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) run_tests.py --benchmark
	@echo "$(GREEN)✅ Benchmarks completed$(RESET)"

memory-profile: ## 💾 Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(RESET)"
	$(PYTHON) -m memory_profiler st.py || echo "$(YELLOW)memory_profiler not available$(RESET)"
	@echo "$(GREEN)✅ Memory profiling completed$(RESET)"

# ============================================================================
# CLEANUP AND MAINTENANCE
# ============================================================================

clean: ## 🧹 Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up temporary files...$(RESET)"
	rm -rf $(REPORTS_DIR)/* || true
	rm -rf htmlcov*/ || true
	rm -rf .coverage* || true
	rm -rf .pytest_cache/ || true
	rm -rf .tox/ || true
	rm -rf build/ || true
	rm -rf dist/ || true
	rm -rf *.egg-info/ || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

clean-deep: clean ## 🧽 Deep clean including virtual environments
	@echo "$(BLUE)Performing deep clean...$(RESET)"
	rm -rf $(VENV_DIR)/ || true
	rm -rf .env/ || true
	rm -rf venv/ || true
	rm -rf env/ || true
	@echo "$(GREEN)✅ Deep cleanup completed$(RESET)"

reset: clean-deep install-dev ## 🔄 Reset development environment
	@echo "$(BLUE)Resetting development environment...$(RESET)"
	@echo "$(GREEN)✅ Environment reset completed$(RESET)"

# ============================================================================
# DOCKER COMMANDS
# ============================================================================

docker-build: ## 🐳 Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t videolingo:latest .
	@echo "$(GREEN)✅ Docker image built$(RESET)"

docker-run: ## 🚀 Run application in Docker
	@echo "$(BLUE)Running VideoLingo in Docker...$(RESET)"
	docker run -d --name videolingo --gpus all -p 8501:8501 videolingo:latest
	@echo "$(GREEN)✅ VideoLingo running at http://localhost:8501$(RESET)"

docker-test: ## 🧪 Run tests in Docker
	@echo "$(BLUE)Running tests in Docker container...$(RESET)"
	docker run --rm videolingo:latest python run_tests.py --fast
	@echo "$(GREEN)✅ Docker tests completed$(RESET)"

docker-clean: ## 🧹 Clean Docker containers and images
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	docker stop videolingo || true
	docker rm videolingo || true
	docker rmi videolingo:latest || true
	@echo "$(GREEN)✅ Docker cleanup completed$(RESET)"

# ============================================================================
# RELEASE AND DEPLOYMENT
# ============================================================================

check-release: ## ✅ Check if codebase is ready for release
	@echo "$(BLUE)Checking release readiness...$(RESET)"
	$(PYTHON) run_tests.py --coverage --fail-fast
	make lint
	make security
	@echo "$(GREEN)✅ Release checks completed$(RESET)"

package: ## 📦 Package application for distribution
	@echo "$(BLUE)Packaging application...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ Package created in dist/$(RESET)"

# ============================================================================
# UTILITY TARGETS
# ============================================================================

status: ## 📊 Show project status and statistics
	@echo "$(BLUE)VideoLingo Project Status$(RESET)"
	@echo "========================"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Project root: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') files changed"
	@echo "Test files: $$(find tests/ -name '*.py' | wc -l | tr -d ' ') files"
	@echo "Core modules: $$(find core/ -name '*.py' | wc -l | tr -d ' ') files"
	@echo "Reports: $$(ls -la $(REPORTS_DIR) 2>/dev/null | wc -l | tr -d ' ') files"
	@echo "$(GREEN)✅ Status check completed$(RESET)"

install-hooks: ## 🪝 Install git hooks
	@echo "$(BLUE)Installing git hooks...$(RESET)"
	pre-commit install --hook-type pre-commit
	pre-commit install --hook-type pre-push
	@echo "$(GREEN)✅ Git hooks installed$(RESET)"

validate-config: ## ⚙️ Validate configuration files
	@echo "$(BLUE)Validating configuration files...$(RESET)"
	$(PYTHON) -m yaml config.yaml || echo "$(YELLOW)YAML validation not available$(RESET)"
	$(PYTHON) -c "import configparser; configparser.ConfigParser().read('keys.ini')" || echo "$(YELLOW)keys.ini not found$(RESET)"
	@echo "$(GREEN)✅ Configuration validation completed$(RESET)"

# ============================================================================
# SPECIAL TARGETS FOR CI/CD
# ============================================================================

ci-test: ## 🤖 CI/CD optimized test run
	@echo "$(BLUE)Running CI/CD tests...$(RESET)"
	$(PYTHON) run_tests.py --coverage --fail-fast --no-parallel
	@echo "$(GREEN)✅ CI/CD tests completed$(RESET)"

ci-lint: ## 🤖 CI/CD optimized linting
	@echo "$(BLUE)Running CI/CD linting...$(RESET)"
	black --check core/ tests/
	isort --check-only core/ tests/
	flake8 core/ tests/ --max-line-length=100
	@echo "$(GREEN)✅ CI/CD linting completed$(RESET)"

ci-security: ## 🤖 CI/CD security checks
	@echo "$(BLUE)Running CI/CD security checks...$(RESET)"
	bandit -r core/ -f json -o $(REPORTS_DIR)/ci_security.json
	safety check --json --output $(REPORTS_DIR)/ci_safety.json
	@echo "$(GREEN)✅ CI/CD security checks completed$(RESET)"

# ============================================================================
# DEBUG AND TROUBLESHOOTING
# ============================================================================

debug-env: ## 🐛 Debug environment setup
	@echo "$(BLUE)Debug Environment Information$(RESET)"
	@echo "============================="
	@echo "Python executable: $$(which $(PYTHON))"
	@echo "Pip executable: $$(which $(PIP))"
	@echo "Python path: $$($(PYTHON) -c 'import sys; print(sys.path)')"
	@echo "Installed packages:"
	@$(PIP) list | head -20
	@echo "Git status:"
	@git status 2>/dev/null || echo "Not a git repository"
	@echo "$(GREEN)✅ Debug information displayed$(RESET)"

test-debug: ## 🐛 Run tests in debug mode
	@echo "$(BLUE)Running tests in debug mode...$(RESET)"
	$(PYTHON) run_tests.py --verbose --fail-fast --no-parallel --pytest-args="--pdb"
	@echo "$(GREEN)✅ Debug tests completed$(RESET)"

# Catch-all target for undefined commands
%:
	@echo "$(RED)❌ Command '$@' not found$(RESET)"
	@echo "$(YELLOW)💡 Run 'make help' to see available commands$(RESET)"
	@exit 1