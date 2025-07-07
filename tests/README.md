# VideoLingo Testing Strategy

This document outlines the comprehensive testing strategy for VideoLingo, targeting 90% branch coverage with robust unit and integration tests.

## 📋 Testing Overview

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini              # Test configuration
├── requirements-test.txt    # Test dependencies
├── test_runner.py          # Custom test runner
├── .coveragerc            # Coverage configuration
├── unit/                  # Unit tests
│   ├── test_config_utils.py
│   ├── test_video_manager.py
│   ├── test_ask_gpt.py
│   ├── test_translate_lines.py
│   └── test_streamlit_ui.py
├── integration/           # Integration tests
│   ├── test_full_pipeline.py
│   └── test_backend_integration.py
├── fixtures/             # Test data and fixtures
└── reports/             # Generated test reports
    ├── coverage_html/
    ├── coverage.xml
    └── junit.xml
```

### Test Categories

#### 🚀 Unit Tests (HIGH PRIORITY)
- **Configuration Management** (`test_config_utils.py`)
  - Thread-safe YAML operations
  - Key validation and updates
  - Path management
  - Error handling and recovery

- **Video File Management** (`test_video_manager.py`)
  - Video ID generation and uniqueness
  - File path resolution
  - Safe overwrite operations
  - Metadata persistence

- **LLM Integration** (`test_ask_gpt.py`)
  - API request/response handling
  - JSON parsing and repair
  - Caching mechanisms
  - Error recovery and fallback

- **Translation System** (`test_translate_lines.py`)
  - Batch processing logic
  - Context preservation
  - Validation and error handling
  - Multi-language support

- **Streamlit UI** (`test_streamlit_ui.py`)
  - Component rendering
  - State management
  - User interactions
  - Error handling

#### 🔗 Integration Tests (MEDIUM PRIORITY)
- **Full Pipeline** (`test_full_pipeline.py`)
  - End-to-end workflow
  - State persistence between steps
  - Error propagation
  - Performance monitoring

- **Backend Integration** (`test_backend_integration.py`)
  - ASR backends (WhisperX, ElevenLabs)
  - TTS backends (OpenAI, Azure, Edge, GPT-SoVITS)
  - Audio processing pipeline
  - Fallback mechanisms

## 🛠 Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Install main project dependencies
pip install -r requirements.txt
```

### Basic Test Execution
```bash
# Run all tests
python tests/test_runner.py

# Run specific test categories
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py ui

# Run with coverage report
python tests/test_runner.py --coverage-only

# Run specific test file
python tests/test_runner.py --specific tests/unit/test_config_utils.py

# Run tests in parallel
python tests/test_runner.py --parallel
```

### Advanced Test Options
```bash
# Run performance tests
python tests/test_runner.py --performance

# Run network tests (requires API keys)
python tests/test_runner.py network

# Run GPU tests (requires CUDA)
python tests/test_runner.py gpu

# Check test dependencies
python tests/test_runner.py --check-deps
```

### Pytest Direct Usage
```bash
# Run with pytest directly
pytest tests/unit/ -v --cov=core

# Run specific markers
pytest -m "unit" -v
pytest -m "integration and not slow" -v
pytest -m "not network" -v

# Generate coverage report
pytest --cov=core --cov-report=html:tests/reports/coverage_html
```

## 🎯 Coverage Goals

### Target Metrics
- **Overall Coverage**: ≥90% branch coverage
- **Critical Modules**: ≥95% coverage
  - `core/utils/config_utils.py`
  - `core/utils/video_manager.py`
  - `core/utils/ask_gpt.py`
  - `core/translate_lines.py`

### Coverage Exclusions
- Test files themselves
- Debug and development code
- Platform-specific fallback code
- Abstract methods and interfaces
- `if __name__ == "__main__"` blocks

## 📊 Test Categories and Markers

### Pytest Markers
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests with dependencies
- `@pytest.mark.slow` - Tests that may take minutes to run
- `@pytest.mark.gpu` - Tests requiring GPU acceleration
- `@pytest.mark.network` - Tests requiring network access
- `@pytest.mark.ui` - Streamlit UI component tests

### Test Selection Examples
```bash
# Run only fast tests
pytest -m "unit and not slow"

# Run all tests except network tests
pytest -m "not network"

# Run integration tests excluding slow ones
pytest -m "integration and not slow"
```

## 🏗 Test Architecture

### Fixtures and Mocking
- **Shared Fixtures** (`conftest.py`)
  - Temporary directories and files
  - Mock API responses
  - Mock external dependencies
  - Database and cache fixtures

- **Mock Strategy**
  - External APIs (OpenAI, Azure, etc.)
  - File system operations
  - Network requests
  - GPU operations
  - Streamlit components

### Test Data Management
- **Mock Video Files**: Small binary files for testing
- **Mock Audio Files**: Sample audio data
- **Configuration Fixtures**: Test configuration files
- **API Response Fixtures**: Cached API responses

## 🔧 Development Workflow

### Pre-commit Testing
```bash
# Run fast tests before commit
pytest tests/unit/ -x --tb=short

# Check specific module
pytest tests/unit/test_config_utils.py -v
```

### Continuous Integration
- **GitHub Actions**: Automated testing on push/PR
- **Multi-platform**: Ubuntu, Windows, macOS
- **Multi-version**: Python 3.8-3.11
- **Coverage Reporting**: Codecov integration
- **Security Scanning**: Dependency and code security checks

### Test-Driven Development
1. Write failing test for new feature
2. Implement minimum code to pass test
3. Refactor while maintaining test coverage
4. Add edge cases and error scenarios

## 📈 Performance Testing

### Benchmarking
```bash
# Run performance benchmarks
python tests/test_runner.py --performance

# View benchmark results
cat tests/reports/benchmark.json
```

### Performance Targets
- **Configuration Loading**: <100ms
- **Video Registration**: <500ms
- **Translation Batch**: <2s per 10 lines
- **UI Rendering**: <200ms initial load

## 🐛 Debugging Tests

### Debug Options
```bash
# Run with detailed output
pytest tests/unit/test_config_utils.py -v -s

# Drop into debugger on failure
pytest tests/unit/test_config_utils.py --pdb

# Run specific test method
pytest tests/unit/test_config_utils.py::TestConfigUtils::test_load_key_existing_config -v
```

### Common Issues
- **File Permissions**: Ensure test runner has write access to temp directories
- **Environment Variables**: Use test environment isolation
- **API Rate Limits**: Mock external APIs in unit tests
- **GPU Memory**: Use CPU-only mode for CI tests

## 📋 Test Maintenance

### Regular Tasks
- **Weekly**: Review coverage reports
- **Monthly**: Update test dependencies
- **Per Release**: Run full test suite including slow/network tests
- **Quarterly**: Performance benchmark comparison

### Adding New Tests
1. Identify test category (unit/integration)
2. Create test file following naming convention
3. Add appropriate markers and fixtures
4. Update this documentation if needed
5. Ensure coverage targets are met

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd VideoLingo

# Install dependencies
pip install -r requirements.txt
pip install -r tests/requirements-test.txt

# Run tests
python tests/test_runner.py unit --verbose

# View coverage report
open tests/reports/coverage_html/index.html
```

## 📚 Best Practices

### Test Writing Guidelines
- **Isolation**: Each test should be independent
- **Clarity**: Test names should describe what they test
- **Coverage**: Aim for edge cases and error conditions
- **Speed**: Keep unit tests fast (<1s each)
- **Reliability**: Tests should not be flaky

### Mock Guidelines
- Mock external dependencies, not internal logic
- Use realistic mock data
- Verify mock interactions when important
- Keep mocks simple and focused

### Assertion Guidelines
- Use specific assertions (`assert x == 5` not `assert x`)
- Test both positive and negative cases
- Include error message context when useful
- Group related assertions logically

This comprehensive testing strategy ensures VideoLingo maintains high quality and reliability while providing confidence for continuous development and deployment.