"""
Simplified VideoLingo Test Configuration
Provides essential fixtures for all tests
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config(temp_dir):
    """Provide a mock configuration for tests"""
    config = {
        "api": {
            "key": "test-key",
            "base_url": "https://api.test.com",
            "model": "gpt-4"
        },
        "paths": {
            "input": str(temp_dir / "input"),
            "output": str(temp_dir / "output"),
            "temp": str(temp_dir / "temp")
        },
        "whisper": {
            "model": "base",
            "device": "cpu"
        }
    }
    
    # Create directories
    for path in config["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return config


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for API tests"""
    with patch("openai.OpenAI") as mock_client:
        instance = Mock()
        response = Mock()
        response.choices = [Mock(message=Mock(content="test response"))]
        instance.chat.completions.create.return_value = response
        mock_client.return_value = instance
        yield instance


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing"""
    video_path = temp_dir / "test_video.mp4"
    video_path.write_bytes(b"fake video content")
    return video_path


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing"""
    audio_path = temp_dir / "test_audio.wav"
    audio_path.write_bytes(b"fake audio content")
    return audio_path


@pytest.fixture(autouse=True)
def isolate_tests(temp_dir):
    """Isolate tests from real config files"""
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
api:
  key: test-key
  model: gpt-4
paths:
  input: input
  output: output
""")
    
    with patch.dict(os.environ, {
        "CONFIG_PATH": str(config_path),
        "VIDEOLINGO_TEST": "true"
    }):
        yield


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for external command tests"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "success"
        yield mock_run
