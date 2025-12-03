# VideoLingo Test Execution Guide

## Overview
This guide provides instructions for running the VideoLingo test suite after simplification. The tests have been restructured to focus on essential functionality and reduce import/dependency issues.

## Test Structure

```
tests/
├── conftest.py                     # Pytest fixtures and configuration
├── test_simple.py                  # Basic functionality tests
├── unit/                           # Unit tests for individual modules
│   ├── test_config_utils_minimal.py # Configuration management tests
│   └── ...                         # Other unit tests
├── integration/                    # Integration tests
└── e2e/                           # End-to-end tests
```

## Prerequisites

1. **Python Environment**
   ```bash
   conda activate videolingo  # or your virtual environment
   ```

2. **Install Test Dependencies**
   ```bash
   pip install pytest pytest-cov pytest-mock
   ```

3. **Set Test Mode**
   The tests automatically set `VIDEOLINGO_TEST_MODE=1` to avoid conflicts with production config.

## Running Tests

### Quick Test Commands

1. **Run All Tests**
   ```bash
   pytest tests/
   ```

2. **Run Basic Smoke Tests**
   ```bash
   pytest tests/test_simple.py -v
   ```

3. **Run Unit Tests Only**
   ```bash
   pytest tests/unit/ -v
   ```

4. **Run With Coverage Report**
   ```bash
   pytest tests/ --cov=core --cov-report=term-missing
   ```

5. **Run Fast Tests Only (exclude slow/network tests)**
   ```bash
   pytest tests/ -m "not slow and not network and not gpu"
   ```

## Test Categories

Tests are marked with different categories for selective execution:

- `unit`: Unit tests for individual functions
- `integration`: Integration tests across modules  
- `e2e`: End-to-end workflow tests
- `slow`: Tests taking >5 seconds
- `network`: Tests requiring internet
- `gpu`: Tests requiring GPU/CUDA

### Run Specific Categories

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Current Test Status

### Working Tests ✅
- `test_simple.py`: 16/16 tests passing
  - Basic YAML operations
  - Path operations
  - Dictionary access patterns
  - File handling
  - Error handling
  - String operations
  - Data type handling
  - Mock usage

- `test_config_utils_minimal.py`: 4/8 tests passing
  - Load configuration with defaults
  - Get storage paths
  - Ensure storage directories

### Known Issues ⚠️

1. **Import Errors**: Some tests may fail due to missing imports. This is being addressed by:
   - Creating `core/utils/__init__.py` with proper exports
   - Fixing circular dependencies
   - Mocking external dependencies

2. **Configuration Tests**: Some config tests fail because they expect specific test data. Use fixtures from `conftest.py`.

3. **Coverage**: Current coverage is ~5-10%. This is expected as many modules aren't tested yet.

## Troubleshooting

### Import Errors
If you see `ImportError: cannot import name 'X' from 'core.utils'`:
1. Check that `core/utils/__init__.py` exists
2. Verify the function/class exists in the source module
3. Add the import to `__init__.py` if missing

### Fixture Not Found
If you see `fixture 'X' not found`:
1. Check `tests/conftest.py` for available fixtures
2. Add missing fixtures to `conftest.py`
3. Common fixtures: `temp_dir`, `temp_config_dir`, `mock_config`

### Test Isolation Issues
Tests should be isolated from system configuration:
- Tests automatically set `VIDEOLINGO_TEST_MODE=1`
- Use temporary directories for file operations
- Mock external API calls

## Best Practices

1. **Use Fixtures**: Leverage pytest fixtures from `conftest.py` for common setup
2. **Mock External Dependencies**: Use `mock_openai`, `mock_subprocess` fixtures
3. **Temporary Files**: Use `temp_dir` fixture for file operations
4. **Test Categories**: Mark tests appropriately (`@pytest.mark.unit`, etc.)
5. **Descriptive Names**: Use clear test function names that describe what's being tested

## Adding New Tests

1. **Create Test File**
   ```python
   # tests/unit/test_my_module.py
   import pytest
   from core.my_module import my_function
   
   def test_my_function_basic():
       """Test basic functionality of my_function"""
       result = my_function("input")
       assert result == "expected"
   ```

2. **Use Appropriate Fixtures**
   ```python
   def test_with_config(temp_config_dir):
       """Test using temporary config directory"""
       # temp_config_dir is automatically cleaned up
       ...
   ```

3. **Mark Test Categories**
   ```python
   @pytest.mark.unit
   @pytest.mark.fast
   def test_quick_unit():
       ...
   ```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run tests with strict mode
pytest tests/ --strict-markers --maxfail=5

# Generate XML report for CI
pytest tests/ --junit-xml=test-results.xml

# Generate coverage report
pytest tests/ --cov=core --cov-report=xml --cov-report=html
```

## Next Steps

1. **Expand Test Coverage**: Add tests for remaining core modules
2. **Fix Import Issues**: Resolve remaining import errors in unit tests
3. **Add Integration Tests**: Test complete workflows
4. **Performance Tests**: Add benchmarks for critical paths
5. **Documentation Tests**: Test code examples in documentation

## Support

If you encounter issues:
1. Check this guide for common solutions
2. Review test output carefully for specific error messages
3. Check GitHub issues for known problems
4. Create a new issue with test output if needed
