"""
Comprehensive test coverage for core.tts_backend.openai_tts module.
Tests OpenAI TTS integration, voice validation, error handling with retry decorator.
Covers API requests, response processing, and file path management.
"""

import pytest
import json
import requests
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call, MagicMock

# Import the module under test
from core.tts_backend.openai_tts import openai_tts, BASE_URL, VOICE_LIST


class TestOpenAITTS:
    """Comprehensive test suite for OpenAI TTS functionality."""
    
    @pytest.fixture
    def mock_load_key(self):
        """Mock config loading for OpenAI TTS settings."""
        with patch('core.tts_backend.openai_tts.load_key') as mock_key:
            config = {
                'openai_tts.api_key': 'sk-test-api-key-12345',
                'openai_tts.voice': 'alloy'
            }
            mock_key.side_effect = lambda key: config.get(key, 'default_value')
            yield mock_key
    
    @pytest.fixture
    def mock_requests_success(self):
        """Mock successful requests response."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'fake_audio_wav_content_binary_data'
            mock_response.headers = {'Content-Type': 'audio/wav'}
            mock_post.return_value = mock_response
            yield mock_post
    
    @pytest.fixture
    def mock_requests_failure(self):
        """Mock failed requests response."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = '{"error": "Invalid API key"}'
            mock_post.return_value = mock_response
            yield mock_post
    
    @pytest.fixture
    def mock_file_operations(self):
        """Mock file system operations."""
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            yield {
                'open': mock_file,
                'mkdir': mock_mkdir
            }
    
    def test_successful_openai_tts_generation(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test successful OpenAI TTS audio generation."""
        text = "Hello, this is a test message for OpenAI TTS"
        save_path = "/tmp/openai_test.wav"
        
        with patch('builtins.print') as mock_print:
            openai_tts(text, save_path)
        
        # Verify API request
        mock_requests_success.assert_called_once()
        call_args = mock_requests_success.call_args
        
        # Check URL
        assert call_args[0][0] == BASE_URL
        
        # Check headers
        headers = call_args[1]['headers']
        assert headers['Authorization'] == 'Bearer sk-test-api-key-12345'
        assert headers['Content-Type'] == 'application/json'
        
        # Check payload
        payload_data = json.loads(call_args[1]['data'])
        expected_payload = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "wav"
        }
        assert payload_data == expected_payload
        
        # Verify directory creation
        mock_file_operations['mkdir'].assert_called_once_with(parents=True, exist_ok=True)
        
        # Verify file writing
        mock_file_operations['open'].assert_called_once_with(Path(save_path), 'wb')
        mock_file_operations['open'].return_value.write.assert_called_once_with(b'fake_audio_wav_content_binary_data')
        
        # Verify success message
        mock_print.assert_called_with(f"Audio saved to {Path(save_path)}")
    
    def test_all_valid_voices(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test all valid voice options."""
        for voice in VOICE_LIST:
            mock_requests_success.reset_mock()
            
            # Configure voice
            config = {
                'openai_tts.api_key': 'test-key',
                'openai_tts.voice': voice
            }
            mock_load_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                openai_tts("Test text", "output.wav")
            
            # Check voice in payload
            call_args = mock_requests_success.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['voice'] == voice
    
    def test_invalid_voice_raises_error(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test that invalid voice raises ValueError."""
        invalid_voices = ['invalid_voice', 'not_a_voice', '', None, 123, 'ALLOY']
        
        for invalid_voice in invalid_voices:
            if invalid_voice is not None:
                config = {
                    'openai_tts.api_key': 'test-key',
                    'openai_tts.voice': invalid_voice
                }
                mock_load_key.side_effect = lambda key: config.get(key)
                
                with pytest.raises(ValueError, match=f"Invalid voice: {invalid_voice}. Please choose from"):
                    openai_tts("Test text", "output.wav")
    
    def test_voice_list_constants(self):
        """Test that VOICE_LIST contains expected voices."""
        expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        assert VOICE_LIST == expected_voices
        assert len(VOICE_LIST) == 6
    
    def test_base_url_constant(self):
        """Test BASE_URL constant."""
        assert BASE_URL == "https://api.302.ai/v1/audio/speech"
    
    def test_different_text_inputs(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test various text input scenarios."""
        test_texts = [
            "Simple text",
            "Text with numbers 123 and symbols !@#$%",
            "Multi-line\ntext with\nline breaks",
            "Unicode text: 你好世界 🎵 🎤",
            "Very long text. " * 100,  # ~1500 characters
            "",  # Empty text
            "   ",  # Whitespace only
            "Single word",
            "Punctuation: Hello, world! How are you? I'm fine.",
            "Special chars: <>&\"'",
        ]
        
        for text in test_texts:
            mock_requests_success.reset_mock()
            
            with patch('builtins.print'):
                openai_tts(text, "output.wav")
            
            call_args = mock_requests_success.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['input'] == text
    
    def test_path_handling_and_directory_creation(self, mock_load_key, mock_requests_success):
        """Test path handling and automatic directory creation."""
        test_paths = [
            "/tmp/audio.wav",
            "./output/nested/deep/audio.wav",
            "~/audio/voice.wav", 
            "/Users/test/Documents/VideoLingo/audio.wav",
            "C:\\temp\\audio\\voice.wav"
        ]
        
        for path in test_paths:
            with patch('builtins.open', mock_open()) as mock_file, \
                 patch('pathlib.Path.mkdir') as mock_mkdir:
                
                with patch('builtins.print'):
                    openai_tts("Test text", path)
                
                # Verify directory creation is called
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                
                # Verify file opening with correct path
                mock_file.assert_called_once_with(Path(path), 'wb')
    
    def test_api_error_responses(self, mock_load_key, mock_file_operations):
        """Test handling of various API error responses."""
        error_cases = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (429, "Rate limit exceeded"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable")
        ]
        
        for status_code, error_text in error_cases:
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.text = error_text
                mock_post.return_value = mock_response
                
                with patch('builtins.print') as mock_print:
                    openai_tts("Test text", "output.wav")
                
                # Should print error status and text
                mock_print.assert_any_call(f"Error: {status_code}")
                mock_print.assert_any_call(error_text)
    
    def test_network_request_exceptions(self, mock_load_key, mock_file_operations):
        """Test handling of network-related exceptions."""
        exceptions = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.HTTPError("HTTP Error"),
            requests.exceptions.RequestException("General request error")
        ]
        
        for exception in exceptions:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = exception
                
                # Should raise the exception (due to @except_handler decorator)
                with pytest.raises(type(exception)):
                    openai_tts("Test text", "output.wav")
    
    def test_json_serialization_error(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test handling of JSON serialization errors."""
        # Create object that can't be JSON serialized
        with patch('json.dumps') as mock_dumps:
            mock_dumps.side_effect = TypeError("Object not JSON serializable")
            
            with pytest.raises(TypeError):
                openai_tts("Test text", "output.wav")
    
    def test_file_write_permissions_error(self, mock_load_key, mock_requests_success):
        """Test handling of file write permission errors."""
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open') as mock_open_file:
            mock_open_file.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                openai_tts("Test text", "/protected/path/audio.wav")
    
    def test_disk_space_error(self, mock_load_key, mock_requests_success):
        """Test handling of disk space errors."""
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
             patch('pathlib.Path.mkdir'):
            mock_file.return_value.write.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError):
                openai_tts("Test text", "output.wav")
    
    def test_config_loading_errors(self, mock_requests_success, mock_file_operations):
        """Test handling of configuration loading errors."""
        with patch('core.tts_backend.openai_tts.load_key') as mock_key:
            mock_key.side_effect = KeyError("Missing configuration key")
            
            with pytest.raises(KeyError):
                openai_tts("Test text", "output.wav")
    
    def test_missing_api_key_handling(self, mock_requests_success, mock_file_operations):
        """Test handling of missing API key."""
        with patch('core.tts_backend.openai_tts.load_key') as mock_key:
            config = {
                'openai_tts.api_key': None,
                'openai_tts.voice': 'alloy'
            }
            mock_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                openai_tts("Test text", "output.wav")
            
            # Should still make request with None API key
            call_args = mock_requests_success.call_args
            headers = call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer None'
    
    def test_retry_decorator_functionality(self, mock_load_key, mock_file_operations):
        """Test that retry decorator works correctly."""
        with patch('requests.post') as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("Connection failed"),
                requests.exceptions.Timeout("Timeout"),
                Mock(status_code=200, content=b'audio_data')
            ]
            
            with patch('builtins.print'):
                # Should succeed after retries
                openai_tts("Test text", "output.wav")
            
            # Should have been called 3 times (original + 2 retries)
            assert mock_post.call_count == 3
    
    def test_payload_structure_validation(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test that API payload has correct structure."""
        with patch('builtins.print'):
            openai_tts("Test message", "output.wav")
        
        call_args = mock_requests_success.call_args
        payload_data = json.loads(call_args[1]['data'])
        
        # Verify all required fields
        required_fields = ['model', 'input', 'voice', 'response_format']
        for field in required_fields:
            assert field in payload_data
        
        # Verify field values
        assert payload_data['model'] == 'tts-1'
        assert payload_data['input'] == 'Test message'
        assert payload_data['voice'] == 'alloy'
        assert payload_data['response_format'] == 'wav'
    
    def test_binary_audio_content_handling(self, mock_load_key, mock_file_operations):
        """Test handling of various binary audio content types."""
        audio_contents = [
            b'RIFF\x00\x00\x00\x00WAVE',  # WAV header
            b'\x00' * 1000,               # Null bytes
            b'\xff' * 1000,               # High bytes
            b'Mixed\x00\xff\x52\x49binary\x46\x46data',
            b''  # Empty content
        ]
        
        for content in audio_contents:
            mock_file_operations['open'].reset_mock()
            
            with patch('requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.content = content
                mock_post.return_value = mock_response
                
                with patch('builtins.print'):
                    openai_tts("Test text", "output.wav")
                
                mock_file_operations['open'].return_value.write.assert_called_once_with(content)
    
    def test_concurrent_requests_handling(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test concurrent request handling."""
        import threading
        import time
        
        results = []
        def run_tts(text, path):
            try:
                with patch('builtins.print'):
                    openai_tts(f"Text {text}", f"output_{path}.wav")
                results.append(True)
            except Exception as e:
                results.append(str(e))
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_tts, args=(i, i))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(result is True for result in results)
        assert len(results) == 5
    
    def test_large_audio_response_handling(self, mock_load_key, mock_file_operations):
        """Test handling of large audio responses."""
        # Simulate large audio file (10MB)
        large_content = b'A' * (10 * 1024 * 1024)
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = large_content
            mock_post.return_value = mock_response
            
            with patch('builtins.print'):
                openai_tts("Test text", "output.wav")
            
            mock_file_operations['open'].return_value.write.assert_called_once_with(large_content)
    
    def test_special_path_characters(self, mock_load_key, mock_requests_success):
        """Test paths with special characters."""
        special_paths = [
            "/tmp/audio with spaces.wav",
            "/tmp/audio-with-dashes.wav",
            "/tmp/audio_with_underscores.wav",
            "/tmp/audio (with) parentheses.wav",
            "/tmp/audio[with]brackets.wav",
            "/tmp/音频文件.wav",  # Unicode filename
        ]
        
        for path in special_paths:
            with patch('builtins.open', mock_open()) as mock_file, \
                 patch('pathlib.Path.mkdir'):
                
                with patch('builtins.print'):
                    openai_tts("Test text", path)
                
                mock_file.assert_called_once_with(Path(path), 'wb')
    
    def test_main_function_execution(self, mock_load_key, mock_requests_success, mock_file_operations):
        """Test the main function execution when run as script."""
        with patch('builtins.print'):
            openai_tts("Hi! Welcome to VideoLingo!", "test.wav")
        
        mock_requests_success.assert_called_once()
        mock_file_operations['open'].assert_called_once_with(Path("test.wav"), 'wb')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
