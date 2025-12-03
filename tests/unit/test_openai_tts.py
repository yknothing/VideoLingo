import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import requests
import json
from pathlib import Path

from core.tts_backend.openai_tts import openai_tts, VOICE_LIST, BASE_URL


class TestOpenaiTts:
    """Test OpenAI TTS integration functionality."""
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "openai_test.wav")
    
    @pytest.fixture
    def mock_successful_response(self):
        """Create mock successful requests response."""
        mock_response = Mock()
        mock_response.content = b"fake_wav_audio_data_12345"
        mock_response.status_code = 200
        return mock_response
    
    @pytest.fixture
    def mock_error_response(self):
        """Create mock error requests response."""
        mock_response = Mock()
        mock_response.content = b""
        mock_response.status_code = 400
        mock_response.text = "Bad Request: Invalid voice parameter"
        return mock_response
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_openai_tts_success(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test successful OpenAI TTS generation."""
        # Setup mocks
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "alloy"
        }[key]
        mock_post.return_value = mock_successful_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            openai_tts("Hello, world!", temp_audio_file)
            
            # Verify API call
            mock_post.assert_called_once_with(
                BASE_URL,
                headers={
                    'Authorization': 'Bearer sk-test123456789',
                    'Content-Type': 'application/json'
                },
                data=json.dumps({
                    "model": "tts-1",
                    "input": "Hello, world!",
                    "voice": "alloy",
                    "response_format": "wav"
                })
            )
            
            # Verify file writing
            mock_file.assert_called_once_with(Path(temp_audio_file), 'wb')
            mock_file().write.assert_called_once_with(b"fake_wav_audio_data_12345")
    
    def test_voice_list_constants(self):
        """Test that VOICE_LIST contains expected voices."""
        expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        assert VOICE_LIST == expected_voices
        assert len(VOICE_LIST) == 6
        assert all(isinstance(voice, str) for voice in VOICE_LIST)
    
    def test_base_url_constant(self):
        """Test that BASE_URL is correct."""
        assert BASE_URL == "https://api.302.ai/v1/audio/speech"
    
    @patch('core.tts_backend.openai_tts.load_key')
    def test_invalid_voice_raises_error(self, mock_load_key, temp_audio_file):
        """Test that invalid voice raises ValueError."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "invalid_voice"
        }[key]
        
        with pytest.raises(ValueError, match="Invalid voice: invalid_voice. Please choose from"):
            openai_tts("Test text", temp_audio_file)
    
    @pytest.mark.parametrize("voice", VOICE_LIST)
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_all_valid_voices(self, mock_post, mock_load_key, voice, temp_audio_file, mock_successful_response):
        """Test that all voices in VOICE_LIST are accepted."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": voice
        }[key]
        mock_post.return_value = mock_successful_response
        
        with patch('builtins.open', mock_open()):
            # Should not raise exception
            openai_tts(f"Testing {voice} voice", temp_audio_file)
            
            # Verify correct voice is used
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['voice'] == voice
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_directory_creation(self, mock_post, mock_load_key, mock_successful_response, tmp_path):
        """Test that parent directories are created if they don't exist."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "nova"
        }[key]
        mock_post.return_value = mock_successful_response
        
        # Create nested path that doesn't exist
        nested_audio_file = str(tmp_path / "nested" / "dirs" / "audio.wav")
        
        with patch('builtins.open', mock_open()) as mock_file:
            openai_tts("Test directory creation", nested_audio_file)
            
            # Verify file is opened with correct path
            mock_file.assert_called_once_with(Path(nested_audio_file), 'wb')
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_http_error_response(self, mock_post, mock_load_key, temp_audio_file, mock_error_response):
        """Test handling of HTTP error response."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "echo"
        }[key]
        mock_post.return_value = mock_error_response
        
        # Should not raise exception, but should print error
        with patch('builtins.print') as mock_print:
            openai_tts("Test error handling", temp_audio_file)
            
            # Verify error is printed
            mock_print.assert_any_call("Error: 400")
            mock_print.assert_any_call("Bad Request: Invalid voice parameter")
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_network_exception(self, mock_post, mock_load_key, temp_audio_file):
        """Test handling of network exceptions."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "fable"
        }[key]
        
        # Mock network error
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        # Should raise exception due to @except_handler decorator retry logic
        with pytest.raises(Exception):
            openai_tts("Test network error", temp_audio_file)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_timeout_exception(self, mock_post, mock_load_key, temp_audio_file):
        """Test handling of timeout exceptions."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "onyx"
        }[key]
        
        # Mock timeout error
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(Exception):
            openai_tts("Test timeout", temp_audio_file)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_empty_text(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test OpenAI TTS with empty text."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "shimmer"
        }[key]
        mock_post.return_value = mock_successful_response
        
        with patch('builtins.open', mock_open()):
            openai_tts("", temp_audio_file)
            
            # Verify empty text is sent to API
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['input'] == ""
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_long_text(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test OpenAI TTS with very long text."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "alloy"
        }[key]
        mock_post.return_value = mock_successful_response
        
        long_text = "This is a very long text that exceeds normal length limits. " * 100
        
        with patch('builtins.open', mock_open()):
            openai_tts(long_text, temp_audio_file)
            
            # Verify long text is sent to API
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['input'] == long_text
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_unicode_text(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test OpenAI TTS with Unicode characters."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "nova"
        }[key]
        mock_post.return_value = mock_successful_response
        
        unicode_text = "Hello 世界! Bonjour 🌍! Привет мир!"
        
        with patch('builtins.open', mock_open()):
            openai_tts(unicode_text, temp_audio_file)
            
            # Verify Unicode text is properly handled
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['input'] == unicode_text
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_special_characters(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test OpenAI TTS with special characters."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "echo"
        }[key]
        mock_post.return_value = mock_successful_response
        
        special_text = "Testing: quotes \"hello\", apostrophes 'world', and symbols @#$%^&*()!"
        
        with patch('builtins.open', mock_open()):
            openai_tts(special_text, temp_audio_file)
            
            # Verify special characters are handled
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['input'] == special_text
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_json_payload_structure(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test that JSON payload has correct structure."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "fable"
        }[key]
        mock_post.return_value = mock_successful_response
        
        test_text = "Testing JSON structure"
        
        with patch('builtins.open', mock_open()):
            openai_tts(test_text, temp_audio_file)
            
            # Verify JSON payload structure
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            
            expected_keys = {"model", "input", "voice", "response_format"}
            assert set(payload_data.keys()) == expected_keys
            assert payload_data["model"] == "tts-1"
            assert payload_data["input"] == test_text
            assert payload_data["voice"] == "fable"
            assert payload_data["response_format"] == "wav"
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_headers_structure(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test that request headers have correct structure."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test-key-xyz",
            "openai_tts.voice": "onyx"
        }[key]
        mock_post.return_value = mock_successful_response
        
        with patch('builtins.open', mock_open()):
            openai_tts("Testing headers", temp_audio_file)
            
            # Verify headers structure
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            
            assert headers['Authorization'] == 'Bearer sk-test-key-xyz'
            assert headers['Content-Type'] == 'application/json'
    
    def test_missing_api_key(self, temp_audio_file):
        """Test OpenAI TTS with missing API key."""
        with patch('core.tts_backend.openai_tts.load_key') as mock_load_key:
            mock_load_key.side_effect = KeyError("openai_tts.api_key not found")
            
            with pytest.raises(KeyError):
                openai_tts("Test missing key", temp_audio_file)
    
    def test_missing_voice_config(self, temp_audio_file):
        """Test OpenAI TTS with missing voice configuration."""
        with patch('core.tts_backend.openai_tts.load_key') as mock_load_key:
            def mock_load_key_func(key):
                if key == "openai_tts.api_key":
                    return "sk-test123456789"
                elif key == "openai_tts.voice":
                    raise KeyError("openai_tts.voice not found")
                
            mock_load_key.side_effect = mock_load_key_func
            
            with pytest.raises(KeyError):
                openai_tts("Test missing voice", temp_audio_file)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_file_write_error(self, mock_post, mock_load_key, temp_audio_file, mock_successful_response):
        """Test handling of file write error."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "shimmer"
        }[key]
        mock_post.return_value = mock_successful_response
        
        # Mock file write error
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Cannot write to file")
            
            with pytest.raises(IOError):
                openai_tts("Test file error", temp_audio_file)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_successful_response_content_types(self, mock_post, mock_load_key, temp_audio_file):
        """Test handling of different response content types."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "alloy"
        }[key]
        
        # Test different audio content types
        test_contents = [
            b"RIFF" + b"\x00" * 100,  # WAV format
            b"\xFF\xFB" + b"\x00" * 100,  # MP3 format
            b"OggS" + b"\x00" * 100,  # OGG format
            b"\x00" * 100,  # Generic binary
        ]
        
        for content in test_contents:
            mock_response = Mock()
            mock_response.content = content
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            with patch('builtins.open', mock_open()) as mock_file:
                openai_tts("Test content types", temp_audio_file)
                
                mock_file().write.assert_called_once_with(content)
                mock_file.reset_mock()


class TestOpenaiTtsDecorator:
    """Test the @except_handler decorator functionality."""
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_decorator_retry_logic(self, mock_post, mock_load_key, tmp_path):
        """Test that decorator retries failed requests."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "alloy"
        }[key]
        
        # First two calls fail, third succeeds
        mock_post.side_effect = [
            requests.exceptions.RequestException("First failure"),
            requests.exceptions.RequestException("Second failure"),
            Mock(content=b"success", status_code=200)
        ]
        
        audio_file = str(tmp_path / "retry_test.wav")
        
        with patch('builtins.open', mock_open()):
            openai_tts("Test retry", audio_file)
            
            # Should be called 3 times (2 failures + 1 success)
            assert mock_post.call_count == 3
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_decorator_max_retries_exceeded(self, mock_post, mock_load_key, tmp_path):
        """Test that decorator raises exception after max retries."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test123456789",
            "openai_tts.voice": "alloy"
        }[key]
        
        # All calls fail
        mock_post.side_effect = requests.exceptions.RequestException("Persistent failure")
        
        audio_file = str(tmp_path / "max_retry_test.wav")
        
        with pytest.raises(Exception, match="Failed to generate audio using OpenAI TTS"):
            openai_tts("Test max retries", audio_file)
            
            # Should be called 3 times (max retries)
            assert mock_post.call_count == 3


class TestOpenaiTtsIntegration:
    """Integration tests for OpenAI TTS functionality."""
    
    def test_openai_tts_main_function_exists(self):
        """Test that main function can be imported."""
        from core.tts_backend.openai_tts import openai_tts
        assert callable(openai_tts)
    
    def test_openai_tts_main_module_structure(self):
        """Test that main module has expected structure."""
        import core.tts_backend.openai_tts as openai_module
        
        # Check required constants
        assert hasattr(openai_module, 'BASE_URL')
        assert hasattr(openai_module, 'VOICE_LIST')
        
        # Check main function
        assert hasattr(openai_module, 'openai_tts')
        assert callable(openai_module.openai_tts)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_realistic_integration_workflow(self, mock_post, mock_load_key, tmp_path):
        """Test realistic OpenAI TTS integration workflow."""
        # Setup realistic configuration
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-proj-realistic-key-12345",
            "openai_tts.voice": "nova"
        }[key]
        
        # Mock realistic API response
        mock_response = Mock()
        mock_response.content = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 1000  # Realistic WAV header + data
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        test_audio_file = str(tmp_path / "integration_test.wav")
        realistic_text = "Welcome to VideoLingo! This is a comprehensive video translation and dubbing application."
        
        with patch('builtins.open', mock_open()) as mock_file:
            openai_tts(realistic_text, test_audio_file)
            
            # Verify realistic API interaction
            call_args = mock_post.call_args
            assert call_args[0][0] == BASE_URL
            
            headers = call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer sk-proj-realistic-key-12345'
            assert headers['Content-Type'] == 'application/json'
            
            payload = json.loads(call_args[1]['data'])
            assert payload['model'] == 'tts-1'
            assert payload['input'] == realistic_text
            assert payload['voice'] == 'nova'
            assert payload['response_format'] == 'wav'
            
            # Verify file handling
            mock_file.assert_called_once_with(Path(test_audio_file), 'wb')
            mock_file().write.assert_called_once_with(mock_response.content)
    
    @patch('core.tts_backend.openai_tts.load_key')
    @patch('core.tts_backend.openai_tts.requests.post')
    def test_error_recovery_workflow(self, mock_post, mock_load_key, tmp_path):
        """Test error recovery in realistic workflow."""
        mock_load_key.side_effect = lambda key: {
            "openai_tts.api_key": "sk-test-recovery-key",
            "openai_tts.voice": "echo"
        }[key]
        
        # Simulate transient error followed by success
        mock_error_response = Mock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"
        
        mock_success_response = Mock()
        mock_success_response.content = b"recovered_audio_data"
        mock_success_response.status_code = 200
        
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_success_response
        ]
        
        test_audio_file = str(tmp_path / "recovery_test.wav")
        
        with patch('builtins.open', mock_open()) as mock_file:
            openai_tts("Test error recovery", test_audio_file)
            
            # Should succeed after retry
            assert mock_post.call_count == 2
            mock_file().write.assert_called_once_with(b"recovered_audio_data")
