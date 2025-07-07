# VideoLingo Test Configuration
# pytest configuration and shared fixtures

import os
import sys
import tempfile
import shutil
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def temp_work_dir():
    """Create a temporary working directory for all tests"""
    temp_dir = tempfile.mkdtemp(prefix="videolingo_test_")
    os.chdir(temp_dir)
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def temp_config_dir(temp_work_dir):
    """Create temporary config directory structure"""
    config_dir = Path(temp_work_dir) / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create default test config
    test_config = {
        "api": {
            "key": "test-api-key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "llm_support_json": True
        },
        "paths": {
            "input": str(config_dir / "input"),
            "temp": str(config_dir / "temp"),
            "output": str(config_dir / "output")
        },
        "asr": {
            "whisper": {
                "model": "base",
                "language": "auto",
                "device": "cpu"
            }
        },
        "tts": {
            "engine": "openai",
            "voice": "alloy",
            "speed": 1.0
        },
        "translation": {
            "target_language": "Chinese (Simplified)",
            "chunk_size": 1000
        },
        "allowed_video_formats": ["mp4", "avi", "mov", "mkv"],
        "allowed_audio_formats": ["mp3", "wav", "m4a"]
    }
    
    config_file = config_dir / "config.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    # Create directory structure
    for path in ["input", "temp", "output"]:
        (config_dir / path).mkdir(exist_ok=True)
    
    return config_dir

@pytest.fixture
def mock_video_file(temp_config_dir):
    """Create a mock video file for testing"""
    video_path = temp_config_dir / "input" / "test_video.mp4"
    # Create a small mock video file (just bytes)
    with open(video_path, 'wb') as f:
        f.write(b'fake_video_content' * 1000)  # ~17KB file
    return str(video_path)

@pytest.fixture
def mock_audio_file(temp_config_dir):
    """Create a mock audio file for testing"""
    audio_path = temp_config_dir / "input" / "test_audio.mp3"
    with open(audio_path, 'wb') as f:
        f.write(b'fake_audio_content' * 500)  # ~8.5KB file
    return str(audio_path)

@pytest.fixture
def mock_whisperx_result():
    """Mock WhisperX transcription result"""
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello"},
                    {"start": 0.6, "end": 1.0, "word": "world"}
                ]
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "This is a test",
                "words": [
                    {"start": 2.5, "end": 2.8, "word": "This"},
                    {"start": 2.9, "end": 3.1, "word": "is"},
                    {"start": 3.2, "end": 3.3, "word": "a"},
                    {"start": 3.4, "end": 3.8, "word": "test"}
                ]
            }
        ],
        "language": "en"
    }

@pytest.fixture
def mock_llm_response():
    """Mock LLM API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"translation": "你好世界", "confidence": 0.95}'
                }
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    }

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for UI testing"""
    with patch('streamlit.session_state', {}), \
         patch('streamlit.sidebar'), \
         patch('streamlit.header'), \
         patch('streamlit.button', return_value=False), \
         patch('streamlit.text_input', return_value=""), \
         patch('streamlit.selectbox', return_value="test"), \
         patch('streamlit.file_uploader', return_value=None), \
         patch('streamlit.progress'), \
         patch('streamlit.empty'), \
         patch('streamlit.success'), \
         patch('streamlit.error'), \
         patch('streamlit.warning'), \
         patch('streamlit.info'):
        yield

@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls for external tools"""
    with patch('subprocess.run') as mock_run, \
         patch('subprocess.Popen') as mock_popen:
        
        # Mock successful subprocess calls
        mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
        
        # Mock Popen for streaming output
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_process.wait.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        yield {"run": mock_run, "popen": mock_popen}

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for LLM testing"""
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_torch():
    """Mock PyTorch for ASR testing"""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.load'), \
         patch('torch.save'):
        yield

@pytest.fixture
def mock_whisperx():
    """Mock WhisperX for ASR testing"""
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "test transcription"}
        ]
    }
    
    with patch('whisperx.load_model', return_value=mock_model), \
         patch('whisperx.load_align_model'), \
         patch('whisperx.align'):
        yield mock_model

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    # Store original environment
    original_env = dict(os.environ)
    
    # Set test environment
    os.environ.update({
        'VIDEOLINGO_TEST_MODE': '1',
        'OPENAI_API_KEY': 'test-key',
        'VIDEOLINGO_CONFIG_PATH': '',
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def mock_file_operations():
    """Mock file operations to prevent actual file I/O in tests"""
    def mock_exists(path):
        # Simulate existence for common test files
        return str(path).endswith(('.mp4', '.mp3', '.yaml', '.json'))
    
    with patch('os.path.exists', side_effect=mock_exists), \
         patch('os.makedirs'), \
         patch('shutil.copy2'), \
         patch('shutil.rmtree'):
        yield

class MockResponse:
    """Mock HTTP response for API testing"""
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP testing"""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        
        # Default successful responses
        mock_post.return_value = MockResponse({"status": "success"})
        mock_get.return_value = MockResponse({"status": "success"})
        
        yield {"post": mock_post, "get": mock_get}

# Test markers for different test categories
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on path and name"""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath) or item.name.startswith("test_unit_"):
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "integration" in str(item.fspath) or item.name.startswith("test_integration_"):
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.name or "test_full_pipeline" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark network tests
        if "download" in item.name or "api" in item.name:
            item.add_marker(pytest.mark.network)