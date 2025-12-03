# VideoLingo Test Data Management System

A comprehensive, elegant test data management system designed specifically for the VideoLingo project. This system provides unified interfaces for various data formats, dynamic data generation, intelligent caching, and automatic lifecycle management.

## 🎯 Key Features

### **Unified Data Interface**
- Abstract layer supporting JSON, YAML, binary, and media files
- Consistent API across all data types
- Metadata tracking with integrity checks

### **Dynamic Data Generation**
- Programmatic test data generation to avoid large static files
- Reproducible results with seeded generators
- Specialized generators for configurations, API responses, media samples

### **Intelligent Caching**
- Memory and disk-based caching with TTL
- Automatic cache eviction (LRU, size limits)
- Scope-based cache management

### **Automatic Cleanup**
- Safe cleanup with multiple protection mechanisms
- Scope-based lifecycle management
- Background cleanup with configurable intervals

### **Environment Isolation**
- Strict separation between test and production data
- Configurable storage paths with safety checks
- Cross-platform compatibility

## 🏗️ Architecture

```
tests/fixtures/
├── base.py              # Abstract base classes and interfaces
├── fixtures.py          # Concrete data providers and fixture management
├── generators.py        # Dynamic test data generators
├── cache.py            # Intelligent caching system
├── cleanup.py          # Data lifecycle and cleanup management
├── samples/            # Pre-built test data samples
│   ├── configs/        # Configuration variants
│   ├── api_responses/  # API response mocks
│   ├── media/          # Media samples and metadata
│   ├── language/       # Multilingual test data
│   └── errors/         # Error scenarios
└── example_usage.py    # Comprehensive usage examples
```

## 🚀 Quick Start

### Basic Setup

```python
from tests.fixtures import initialize_test_data_system, get_videolingo_fixtures

# Initialize the system
manager = initialize_test_data_system("/path/to/test/data")
fixtures = get_videolingo_fixtures(manager)

# Load predefined configuration
config = fixtures.get_config_fixture("minimal")

# Create test media files
video_file = fixtures.create_mock_video_file(duration_seconds=10)
audio_file = fixtures.create_mock_audio_file(duration_seconds=5)
```

### Dynamic Data Generation

```python
from tests.fixtures.generators import ConfigGenerator, APIResponseGenerator

# Generate configurations
config_gen = ConfigGenerator(seed=42)  # Reproducible
openrouter_config = config_gen.generate(variant="openrouter")
azure_config = config_gen.generate(variant="azure")

# Generate API responses
api_gen = APIResponseGenerator(seed=42)
success_response = api_gen.generate(api_type="openai", scenario="success")
error_response = api_gen.generate(api_type="openai", scenario="error")
```

### Caching System

```python
from tests.fixtures.cache import get_cache_manager
from tests.fixtures.base import DataScope

cache = get_cache_manager()
scoped_cache = cache.get_cache()

# Cache expensive computation
result = expensive_computation()
scoped_cache.set("expensive_result", result, DataScope.TEST_MODULE)

# Retrieve cached data
cached_result = scoped_cache.get("expensive_result")
```

### Automatic Cleanup

```python
from tests.fixtures.cleanup import get_session_cleaner
from tests.fixtures.base import DataScope

cleaner = get_session_cleaner()

# Register files for automatic cleanup
cleaner.register_temp_file(Path("/tmp/test_file.txt"))
cleaner.register_temp_dir(Path("/tmp/test_directory"))

# Manual cleanup by scope
cleaner.cleanup_by_scope(DataScope.TEMPORARY)
```

## 📁 Data Categories

The system organizes test data into logical categories:

- **CONFIG**: Configuration files and variants
- **API_RESPONSE**: Mock API responses and error scenarios
- **MEDIA_SAMPLE**: Audio, video, and image test files
- **LANGUAGE_DATA**: Multilingual text samples and translations
- **ERROR_SCENARIO**: Error conditions and edge cases
- **MOCK_DATA**: General mock data for testing
- **CACHE_DATA**: Cached computation results
- **TEMP_DATA**: Temporary test data

## 🔄 Data Scopes

Automatic lifecycle management based on scope:

- **SESSION**: Persistent across entire test session
- **TEST_MODULE**: Cleaned up after test module completion
- **TEST_FUNCTION**: Cleaned up after test function completion  
- **TEMPORARY**: Cleaned up immediately after use

## 🛡️ Safety Features

### Path Protection
- Multiple layers of path validation
- Protected system directories
- Configurable safe zones
- Path traversal prevention

### Data Integrity
- SHA-256 checksums for all data
- Automatic corruption detection
- Metadata validation
- Size tracking

### Error Handling
- Comprehensive error types
- Graceful degradation
- Detailed error messages
- Recovery suggestions

## 🎨 Sample Data

### Configuration Variants

```yaml
# Minimal configuration for basic testing
minimal.yaml:
  api:
    key: "test-api-key"
    base_url: "https://test.example.com/v1"
  max_workers: 1

# OpenRouter configuration
openrouter.yaml:
  api:
    key: "sk-or-v1-test-key-example"
    base_url: "https://openrouter.ai/api/v1"
    model: "anthropic/claude-3.5-sonnet"

# Error-prone configuration for testing error handling
error_prone.yaml:
  api:
    key: ""  # Missing API key
    base_url: "not-a-valid-url"  # Invalid URL
  max_workers: -1  # Invalid worker count
```

### API Response Mocks

```json
{
  "openai_success": {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [{
      "message": {
        "role": "assistant", 
        "content": "这是一个测试翻译响应。"
      }
    }]
  }
}
```

### Media Samples

Pre-configured metadata for test media files:

```json
{
  "audio_samples": {
    "short_english": {
      "duration": 3.5,
      "format": "wav",
      "language": "en",
      "content": "Hello, this is a test audio sample."
    }
  }
}
```

## 🔧 Integration with Test Frameworks

### pytest Integration

```python
# conftest.py
import pytest
from tests.fixtures import initialize_test_data_system
from tests.fixtures.cleanup import get_session_cleaner

@pytest.fixture(scope="session")
def test_data_system():
    return initialize_test_data_system()

@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data():
    yield
    cleaner = get_session_cleaner()
    cleaner.cleanup_by_scope(DataScope.TEST_FUNCTION)
```

### Custom Test Base Class

```python
class VideoLingoTestBase:
    @classmethod
    def setup_class(cls):
        cls.data_manager = initialize_test_data_system()
        cls.fixtures = get_videolingo_fixtures(cls.data_manager)
    
    def setup_method(self):
        self.cleaner = get_session_cleaner()
        self.cache = get_cache_manager()
    
    def teardown_method(self):
        self.cleaner.cleanup_by_scope(DataScope.TEST_FUNCTION)
```

## 📊 Performance Considerations

### Caching Strategy
- Memory cache for frequently accessed small data
- Disk cache for large or persistent data
- Hybrid cache for optimal performance
- TTL-based expiration

### Data Generation
- Seeded generators for reproducible results
- Lazy loading to minimize memory usage
- Efficient binary data generation
- Batch generation capabilities

### Cleanup Efficiency
- Background cleanup threads
- Batch operations for better performance
- Smart dependency tracking
- Configurable cleanup intervals

## 🧪 Testing the System

Run the comprehensive example:

```bash
# Interactive demonstration
python tests/fixtures/example_usage.py

# Run full test suite
pytest tests/fixtures/example_usage.py -v

# Test individual components
pytest tests/fixtures/test_*.py
```

## 🔍 Debugging and Monitoring

### Enable Debug Logging

```python
import logging
logging.getLogger('tests.fixtures').setLevel(logging.DEBUG)
```

### Monitor Cache Performance

```python
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
print(f"Cache hit ratio: {stats['memory']['hit_ratio']:.2%}")
```

### Track Cleanup Operations

```python
cleaner = get_session_cleaner()
stats = cleaner.cleaner.get_stats()
print(f"Tracked files: {stats['tracked_files']}")
```

## 🤝 Contributing

### Adding New Generators

1. Inherit from `AbstractDataGenerator`
2. Implement `generate()` and `get_metadata_template()`
3. Register with the data manager
4. Add sample data and tests

### Adding New Data Providers

1. Inherit from `AbstractDataProvider`
2. Implement all abstract methods
3. Add format-specific handling
4. Register with appropriate category

### Adding New Cache Implementations

1. Inherit from `AbstractDataCache`
2. Implement caching strategy
3. Add TTL and eviction logic
4. Test with existing cache manager

## 📋 Best Practices

### Data Organization
- Use descriptive names for test data
- Group related data in categories
- Tag data for easy filtering
- Document complex test scenarios

### Performance
- Use caching for expensive operations
- Generate data lazily when possible
- Clean up temporary data promptly
- Monitor cache hit rates

### Security
- Never commit real API keys
- Use mock credentials in samples
- Validate all file paths
- Implement proper access controls

### Testing
- Write tests for generators
- Validate data integrity
- Test error scenarios
- Measure performance impact

## 🗂️ File Structure

```
tests/fixtures/
├── __init__.py                 # Main system integration
├── base.py                     # Abstract base classes
├── fixtures.py                 # Concrete implementations
├── generators.py               # Dynamic data generators
├── cache.py                    # Caching system
├── cleanup.py                  # Cleanup management
├── example_usage.py            # Usage examples
├── README.md                   # This documentation
└── samples/                    # Pre-built test data
    ├── __init__.py
    ├── configs/
    │   ├── minimal.yaml
    │   ├── openrouter.yaml
    │   ├── azure.yaml
    │   ├── full_featured.yaml
    │   └── error_prone.yaml
    ├── api_responses/
    │   ├── openai_success.json
    │   ├── openai_rate_limit.json
    │   ├── whisper_transcription.json
    │   ├── azure_tts_success.json
    │   ├── elevenlabs_error.json
    │   └── translation_batch.json
    ├── media/
    │   ├── sample_metadata.json
    │   ├── english_sample.srt
    │   └── chinese_sample.ass
    ├── language/
    │   ├── english_sentences.json
    │   ├── chinese_sentences.json
    │   └── multilingual_test_cases.json
    └── errors/
        └── common_errors.json
```

---

**Created for VideoLingo Project** - A comprehensive test data management solution designed for quality, performance, and maintainability.
EOF < /dev/null