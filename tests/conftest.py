"""
Pytest configuration and fixtures for VideoLingo tests
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with a basic config.yaml file for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a sample config.yaml
        config_data = {
            'api': {
                'key': 'test-api-key',
                'endpoint': 'https://api.example.com'
            },
            'asr': {
                'whisper': {
                    'model': 'base',
                    'language': 'en'
                }
            },
            'paths': {
                'input': str(temp_path / 'input'),
                'temp': str(temp_path / 'temp'),
                'output': str(temp_path / 'output')
            }
        }
        
        config_file = temp_path / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        yield temp_path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing"""
    video_path = temp_dir / "sample_video.mp4"
    # Create a dummy file (not a real video, just for testing file operations)
    video_path.write_bytes(b"dummy video content")
    return str(video_path)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing"""
    audio_path = temp_dir / "sample_audio.mp3"
    # Create a dummy file (not a real audio, just for testing file operations)
    audio_path.write_bytes(b"dummy audio content")
    return str(audio_path)


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary"""
    return {
        'api': {
            'key': 'test-key',
            'model': 'gpt-4'
        },
        'whisper': {
            'model': 'base',
            'runtime': 'local'
        },
        'paths': {
            'input': '/tmp/input',
            'temp': '/tmp/temp',
            'output': '/tmp/output'
        }
    }


@pytest.fixture
def mock_openai(mocker):
    """Mock OpenAI API calls"""
    mock = mocker.patch('openai.ChatCompletion.create')
    mock.return_value = {
        'choices': [{
            'message': {
                'content': 'Mocked response'
            }
        }]
    }
    return mock


@pytest.fixture
def mock_subprocess(mocker):
    """Mock subprocess calls"""
    mock_run = mocker.patch('subprocess.run')
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Mock output"
    mock_run.return_value.stderr = ""
    return mock_run


@pytest.fixture(autouse=True)
def isolate_tests(monkeypatch):
    """Isolate tests from system environment"""
    # Clear VideoLingo-specific environment variables
    env_vars_to_clear = [
        'VIDEOLINGO_CONFIG_DIR',
        'VIDEOLINGO_API_KEY',
        'VIDEOLINGO_BASE_PATH'
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    
    # Set test mode
    monkeypatch.setenv('VIDEOLINGO_TEST_MODE', '1')
