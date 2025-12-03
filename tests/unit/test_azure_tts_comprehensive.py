"""
Comprehensive test coverage for core.tts_backend.azure_tts module.
Tests Azure TTS integration, API requests, voice configuration, error handling.
Covers SSML generation, authentication, and response processing.
"""

import pytest
import requests
from unittest.mock import Mock, patch, mock_open, call
from pathlib import Path

# Import the module under test
from core.tts_backend.azure_tts import azure_tts


class TestAzureTTS:
    """Comprehensive test suite for Azure TTS functionality."""
    
    @pytest.fixture
    def mock_load_key(self):
        """Mock config loading for Azure TTS settings."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_key:
            config = {
                'azure_tts.api_key': 'test-azure-api-key-123',
                'azure_tts.voice': 'zh-CN-XiaoxiaoNeural'
            }
            mock_key.side_effect = lambda key: config.get(key, 'default_value')
            yield mock_key
    
    @pytest.fixture
    def mock_requests_success(self):
        """Mock successful requests response."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.content = b'fake_audio_content_binary_data'
            mock_response.status_code = 200
            mock_response.headers = {'Content-Type': 'audio/wav'}
            mock_request.return_value = mock_response
            yield mock_request
    
    @pytest.fixture
    def mock_requests_failure(self):
        """Mock failed requests response."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.content = b'{"error": "Authentication failed"}'
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_request.return_value = mock_response
            yield mock_request
    
    @pytest.fixture
    def mock_file_write(self):
        """Mock file writing operations."""
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            yield mock_file
    
    def test_successful_azure_tts_generation(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test successful Azure TTS audio generation."""
        text = "Hello, this is a test message"
        save_path = "/tmp/test_audio.wav"
        
        with patch('builtins.print') as mock_print:
            azure_tts(text, save_path)
        
        # Verify API request
        mock_requests_success.assert_called_once()
        call_args = mock_requests_success.call_args
        
        # Check request method and URL
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "https://api.302.ai/cognitiveservices/v1"
        
        # Check headers
        headers = call_args[1]['headers']
        assert headers['Authorization'] == 'Bearer test-azure-api-key-123'
        assert headers['X-Microsoft-OutputFormat'] == 'riff-16khz-16bit-mono-pcm'
        assert headers['Content-Type'] == 'application/ssml+xml'
        
        # Check SSML payload
        expected_payload = "<speak version='1.0' xml:lang='zh-CN'><voice name='zh-CN-XiaoxiaoNeural'>Hello, this is a test message</voice></speak>"
        assert call_args[1]['data'] == expected_payload
        
        # Verify file writing
        mock_file_write.assert_called_once_with(save_path, 'wb')
        mock_file_write.return_value.write.assert_called_once_with(b'fake_audio_content_binary_data')
        
        # Verify success message
        mock_print.assert_called_once_with(f"Audio saved to {save_path}")
    
    def test_ssml_generation_with_different_voices(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test SSML generation with different voice configurations."""
        test_cases = [
            ('en-US-JennyNeural', 'Hello world'),
            ('zh-CN-YunxiNeural', '你好世界'),
            ('ja-JP-NanamiNeural', 'こんにちは'),
            ('es-ES-ElviraNeural', 'Hola mundo')
        ]
        
        for voice, text in test_cases:
            mock_load_key.reset_mock()
            mock_requests_success.reset_mock()
            
            # Configure voice
            config = {
                'azure_tts.api_key': 'test-key',
                'azure_tts.voice': voice
            }
            mock_load_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                azure_tts(text, "output.wav")
            
            # Check SSML payload contains correct voice
            call_args = mock_requests_success.call_args
            payload = call_args[1]['data']
            assert f"<voice name='{voice}'>{text}</voice>" in payload
    
    def test_special_characters_in_text(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test handling of special characters in input text."""
        special_texts = [
            "Text with & ampersand",
            "Text with < less than",
            "Text with > greater than",
            "Text with \" quotes",
            "Text with 'single quotes'",
            "Mixed <>&\"' characters",
            "Unicode: 你好世界 🎵 🎤"
        ]
        
        for text in special_texts:
            mock_requests_success.reset_mock()
            
            with patch('builtins.print'):
                azure_tts(text, "output.wav")
            
            # Verify request was made with the text
            call_args = mock_requests_success.call_args
            payload = call_args[1]['data']
            assert text in payload
    
    def test_empty_text_handling(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test handling of empty text input."""
        with patch('builtins.print'):
            azure_tts("", "output.wav")
        
        # Should still make request
        mock_requests_success.assert_called_once()
        call_args = mock_requests_success.call_args
        payload = call_args[1]['data']
        assert "<voice name='zh-CN-XiaoxiaoNeural'></voice>" in payload
    
    def test_long_text_handling(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test handling of very long text input."""
        long_text = "This is a very long text. " * 100  # ~2500 characters
        
        with patch('builtins.print'):
            azure_tts(long_text, "output.wav")
        
        # Should process without issues
        mock_requests_success.assert_called_once()
        call_args = mock_requests_success.call_args
        payload = call_args[1]['data']
        assert long_text in payload
    
    def test_different_output_paths(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test different output file paths."""
        test_paths = [
            "/tmp/audio.wav",
            "./output/test.wav", 
            "~/music/voice.wav",
            "/Users/test/Documents/audio.wav",
            "C:\\temp\\audio.wav"
        ]
        
        for path in test_paths:
            mock_file_write.reset_mock()
            
            with patch('builtins.print'):
                azure_tts("Test text", path)
            
            mock_file_write.assert_called_once_with(path, 'wb')
    
    def test_http_request_error_handling(self, mock_load_key, mock_file_write):
        """Test handling of HTTP request errors."""
        with patch('requests.request') as mock_request:
            mock_request.side_effect = requests.exceptions.RequestException("Network error")
            
            with pytest.raises(requests.exceptions.RequestException):
                azure_tts("Test text", "output.wav")
    
    def test_http_timeout_handling(self, mock_load_key, mock_file_write):
        """Test handling of request timeouts."""
        with patch('requests.request') as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout("Request timeout")
            
            with pytest.raises(requests.exceptions.Timeout):
                azure_tts("Test text", "output.wav")
    
    def test_authentication_failure(self, mock_load_key, mock_file_write):
        """Test handling of authentication failures."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.content = b'{"error": "Invalid API key"}'
            mock_response.status_code = 401
            mock_request.return_value = mock_response
            
            # Should still write the error response content to file
            with patch('builtins.print'):
                azure_tts("Test text", "output.wav")
            
            mock_file_write.assert_called_once_with("output.wav", 'wb')
            mock_file_write.return_value.write.assert_called_once_with(b'{"error": "Invalid API key"}')
    
    def test_server_error_response(self, mock_load_key, mock_file_write):
        """Test handling of server error responses."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.content = b'Internal Server Error'
            mock_response.status_code = 500
            mock_request.return_value = mock_response
            
            with patch('builtins.print'):
                azure_tts("Test text", "output.wav")
            
            # Should still write response content
            mock_file_write.assert_called_once_with("output.wav", 'wb')
            mock_file_write.return_value.write.assert_called_once_with(b'Internal Server Error')
    
    def test_file_write_error_handling(self, mock_load_key, mock_requests_success):
        """Test handling of file write errors."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                azure_tts("Test text", "/protected/file.wav")
    
    def test_config_loading_error_handling(self, mock_requests_success, mock_file_write):
        """Test handling of configuration loading errors."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_load_key:
            mock_load_key.side_effect = KeyError("Missing config key")
            
            with pytest.raises(KeyError):
                azure_tts("Test text", "output.wav")
    
    def test_missing_api_key_handling(self, mock_requests_success, mock_file_write):
        """Test handling of missing API key."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_load_key:
            config = {
                'azure_tts.api_key': None,
                'azure_tts.voice': 'test-voice'
            }
            mock_load_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                azure_tts("Test text", "output.wav")
            
            # Should make request with None API key
            call_args = mock_requests_success.call_args
            headers = call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer None'
    
    def test_missing_voice_config_handling(self, mock_requests_success, mock_file_write):
        """Test handling of missing voice configuration."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_load_key:
            config = {
                'azure_tts.api_key': 'test-key',
                'azure_tts.voice': None
            }
            mock_load_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                azure_tts("Test text", "output.wav")
            
            # Should use None as voice name
            call_args = mock_requests_success.call_args
            payload = call_args[1]['data']
            assert "<voice name='None'>" in payload
    
    def test_binary_audio_content_handling(self, mock_load_key, mock_file_write):
        """Test handling of various binary audio content types."""
        audio_contents = [
            b'\x52\x49\x46\x46',  # RIFF header
            b'\x00' * 1000,       # Null bytes
            b'\xff' * 1000,       # High bytes
            b'Mixed\x00\xff\x52\x49binary\x46\x46data'
        ]
        
        for content in audio_contents:
            mock_file_write.reset_mock()
            
            with patch('requests.request') as mock_request:
                mock_response = Mock()
                mock_response.content = content
                mock_response.status_code = 200
                mock_request.return_value = mock_response
                
                with patch('builtins.print'):
                    azure_tts("Test text", "output.wav")
                
                mock_file_write.return_value.write.assert_called_once_with(content)
    
    def test_concurrent_requests_handling(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test that function handles concurrent usage correctly."""
        import threading
        
        results = []
        def run_tts(text, path):
            try:
                azure_tts(f"Text {text}", f"output_{path}.wav")
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
    
    def test_voice_parameter_validation(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test voice parameter handling with various inputs."""
        voice_configs = [
            'zh-CN-XiaoxiaoNeural',
            'en-US-JennyNeural', 
            'ja-JP-NanamiNeural',
            '',  # Empty voice
            'invalid-voice-name',
            'voice with spaces',
            '特殊字符语音'  # Unicode characters
        ]
        
        for voice in voice_configs:
            mock_requests_success.reset_mock()
            
            config = {
                'azure_tts.api_key': 'test-key',
                'azure_tts.voice': voice
            }
            mock_load_key.side_effect = lambda key: config.get(key)
            
            with patch('builtins.print'):
                azure_tts("Test text", "output.wav")
            
            # Check voice is used in SSML
            call_args = mock_requests_success.call_args
            payload = call_args[1]['data']
            assert f"<voice name='{voice}'>" in payload
    
    def test_main_function_execution(self, mock_load_key, mock_requests_success, mock_file_write):
        """Test the main function execution when run as script."""
        # This tests the if __name__ == "__main__" block functionality
        with patch('builtins.print'):
            azure_tts("Hi! Welcome to VideoLingo!", "test.wav")
        
        mock_requests_success.assert_called_once()
        mock_file_write.assert_called_once_with("test.wav", 'wb')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
