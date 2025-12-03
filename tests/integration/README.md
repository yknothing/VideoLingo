# Integration Tests

## Overview

This directory contains cross-module integration tests for VideoLingo, including smoke tests that validate the complete pipeline from video input through transcription, translation, and subtitle generation.

## Test Structure

### Smoke Tests (`test_smoke_integration.py`)

The main integration smoke test suite that:

1. **Generates synthetic test video at runtime** - Creates a 3-second sine-wave video with audio (no external assets required)
2. **Tests end-to-end pipeline** - Validates the flow: `download_video_ytdlp` → `transcribe` → `translate` → `subtitle insertion`
3. **Verifies artifact creation** - Ensures all intermediate files are created correctly
4. **Validates metadata propagation** - Confirms metadata flows through all pipeline stages

### Test Categories

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.integration` - Integration tests that test multiple modules
- `@pytest.mark.slow` - Tests that take >5 seconds to run
- `@pytest.mark.performance` - Performance-focused tests

## Running Tests

### Local Development

Use the provided script for easy test execution:

```bash
# Run smoke tests only (default)
./scripts/run_integration_tests.sh

# Run with coverage report
./scripts/run_integration_tests.sh --coverage

# Run full integration suite
./scripts/run_integration_tests.sh --full

# Run performance tests
./scripts/run_integration_tests.sh --performance

# Verbose output
./scripts/run_integration_tests.sh -v
```

### Direct pytest execution

```bash
# Run integration smoke tests
pytest tests/integration/test_smoke_integration.py -m "integration and slow" -v

# Run with coverage
pytest tests/integration/test_smoke_integration.py -m "integration and slow" --cov=core --cov-report=html

# Run specific test class
pytest tests/integration/test_smoke_integration.py::TestCrossModuleIntegrationSmoke -v

# Run specific test method
pytest tests/integration/test_smoke_integration.py::TestCrossModuleIntegrationSmoke::test_end_to_end_pipeline_smoke -v
```

### CI/CD Pipeline

Integration tests run automatically in CI:

- **On Push/PR**: Smoke tests run for quick validation
- **Nightly**: Full integration suite runs at 2 AM UTC
- **Manual**: Can be triggered via workflow dispatch

## Test Architecture

### Synthetic Video Generation

The `VideoSynthesizer` class generates test videos at runtime:

```python
# Generates 3-second video with 440Hz sine wave audio
VideoSynthesizer.generate_synthetic_video_ffmpeg(
    output_path="test.mp4",
    duration=3.0,
    width=320,
    height=240,
    fps=30,
    frequency=440.0
)
```

If FFmpeg is not available, it falls back to a mock video file generator.

### Pipeline Stages Tested

1. **Video Input/Download**
   - Mock download via `_1_ytdlp.download_video_ytdlp`
   - Video file validation

2. **ASR Transcription**
   - Audio extraction from video
   - Transcription via `_2_asr.transcribe`
   - Segment and word-level timing

3. **Translation**
   - Load transcription chunks
   - Batch translation via `_4_2_translate.translate_all`
   - Translation confidence tracking

4. **Subtitle Generation**
   - Timestamp alignment via `_6_gen_sub.align_timestamp_main`
   - SRT format generation
   - Subtitle file creation

### Test Fixtures

- `synthetic_video_fixture` - Generates test video once per test class
- `pipeline_environment` - Sets up complete directory structure and paths
- Mock helpers for transcription, translation, and subtitle generation

## Performance Considerations

- Tests are marked `@pytest.mark.slow` for CI filtering
- Timeout set to 300 seconds for smoke tests
- Memory usage monitored in performance tests
- Parallel execution supported via pytest-xdist

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Tests will use mock video generation
   - Install FFmpeg: `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)

2. **Import errors**
   - Ensure PYTHONPATH includes project root: `export PYTHONPATH=$PYTHONPATH:$(pwd)`
   - Install dependencies: `pip install -r requirements.txt`

3. **Test timeouts**
   - Increase timeout: `pytest --timeout=600`
   - Check system resources (CPU, memory)

4. **Mock failures**
   - Verify mock patches match actual module structure
   - Check for changes in core module interfaces

## Adding New Integration Tests

1. Create test file in `tests/integration/`
2. Use appropriate pytest markers
3. Follow naming convention: `test_<feature>_integration.py`
4. Include docstrings explaining test purpose
5. Use fixtures for common setup

Example template:

```python
import pytest
from unittest.mock import patch

@pytest.mark.integration
@pytest.mark.slow
class TestNewFeatureIntegration:
    """Integration tests for new feature"""
    
    def test_feature_pipeline(self, pipeline_environment):
        """Test feature integration with pipeline"""
        # Setup
        env = pipeline_environment
        
        # Mock dependencies
        with patch('core.module.function') as mock_func:
            mock_func.return_value = expected_result
            
            # Execute pipeline stage
            result = run_pipeline_stage()
            
            # Verify
            assert result == expected
            assert artifacts_exist()
```

## Coverage Goals

- Smoke tests: Cover critical path through all major modules
- Full suite: >80% coverage of integration points
- Performance tests: Validate no regressions in key metrics
