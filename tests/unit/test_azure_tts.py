import pytest
from unittest.mock import Mock, patch, mock_open
import requests
from pathlib import Path

from core.tts_backend.azure_tts import azure_tts


class TestAzureTts:
    """Test Azure TTS integration functionality."""
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "azure_test.wav")
    
    @pytest.fixture
    def mock_requests_response(self):
        """Create mock requests response."""
        mock_response = Mock()
        mock_response.content = b"fake_audio_data"
        mock_response.status_code = 200
        return mock_response
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_success(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test successful Azure TTS generation."""
        # Setup mocks
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_api_key_123",
            "azure_tts.voice": "zh-CN-XiaohanNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts("Hello, this is a test.", temp_audio_file)
            
            # Verify API call
            mock_request.assert_called_once_with(
                "POST",
                "https://api.302.ai/cognitiveservices/v1",
                headers={
                    'Authorization': 'Bearer test_api_key_123',
                    'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
                    'Content-Type': 'application/ssml+xml'
                },
                data="<speak version='1.0' xml:lang='zh-CN'><voice name='zh-CN-XiaohanNeural'>Hello, this is a test.</voice></speak>"
            )
            
            # Verify file writing
            mock_file.assert_called_once_with(temp_audio_file, 'wb')
            mock_file().write.assert_called_once_with(b"fake_audio_data")
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_different_voice(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with different voice settings."""
        # Setup mocks with different voice
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "different_key_456",
            "azure_tts.voice": "en-US-JennyNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts("Testing different voice", temp_audio_file)
            
            # Verify correct voice is used in SSML
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert "en-US-JennyNeural" in ssml_data
            assert "Testing different voice" in ssml_data
            
            # Verify correct API key
            headers = call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer different_key_456'
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_special_characters(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with special characters in text."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        special_text = "Hello & goodbye! How are you? 你好世界 123"
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts(special_text, temp_audio_file)
            
            # Verify special characters are preserved in SSML
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert special_text in ssml_data
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_empty_text(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with empty text."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts("", temp_audio_file)
            
            # Verify empty text is handled
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert "<voice name='zh-CN-XiaomoNeural'></voice>" in ssml_data
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_long_text(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with long text."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "en-US-AriaNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        long_text = "This is a very long text that should be handled properly by the Azure TTS service. " * 20
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts(long_text, temp_audio_file)
            
            # Verify long text is included in SSML
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert long_text in ssml_data
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_http_error(self, mock_request, mock_load_key, temp_audio_file):
        """Test Azure TTS with HTTP error response."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        
        # Mock HTTP error
        mock_response = Mock()
        mock_response.content = b"error_response_data"
        mock_response.status_code = 400
        mock_request.return_value = mock_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            # Should still write error content to file
            azure_tts("Test error handling", temp_audio_file)
            
            mock_file().write.assert_called_once_with(b"error_response_data")
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_network_error(self, mock_request, mock_load_key, temp_audio_file):
        """Test Azure TTS with network error."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        
        # Mock network error
        mock_request.side_effect = requests.exceptions.RequestException("Network error")
        
        with pytest.raises(requests.exceptions.RequestException):
            azure_tts("Test network error", temp_audio_file)
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_timeout(self, mock_request, mock_load_key, temp_audio_file):
        """Test Azure TTS with timeout error."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        
        # Mock timeout error
        mock_request.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(requests.exceptions.Timeout):
            azure_tts("Test timeout", temp_audio_file)
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_file_write_error(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with file write error."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        # Mock file write error
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Cannot write file")
            
            with pytest.raises(IOError):
                azure_tts("Test file error", temp_audio_file)
    
    def test_azure_tts_missing_api_key(self, temp_audio_file):
        """Test Azure TTS with missing API key."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_load_key:
            mock_load_key.side_effect = KeyError("azure_tts.api_key not found")
            
            with pytest.raises(KeyError):
                azure_tts("Test missing key", temp_audio_file)
    
    def test_azure_tts_missing_voice(self, temp_audio_file):
        """Test Azure TTS with missing voice configuration."""
        with patch('core.tts_backend.azure_tts.load_key') as mock_load_key:
            def mock_load_key_func(key):
                if key == "azure_tts.api_key":
                    return "test_key"
                elif key == "azure_tts.voice":
                    raise KeyError("azure_tts.voice not found")
                
            mock_load_key.side_effect = mock_load_key_func
            
            with pytest.raises(KeyError):
                azure_tts("Test missing voice", temp_audio_file)
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_unicode_text(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with Unicode text."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        unicode_text = "你好世界！This is English. こんにちは世界！🎵🎤"
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts(unicode_text, temp_audio_file)
            
            # Verify Unicode text is properly handled
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert unicode_text in ssml_data
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_xml_escaping(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test Azure TTS with text containing XML characters."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "en-US-AriaNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        xml_text = "Test with <tag> and & symbol"
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts(xml_text, temp_audio_file)
            
            # Verify XML characters are handled (Note: this may need escaping in real implementation)
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            assert xml_text in ssml_data
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_different_outputs_formats(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test that Azure TTS uses correct output format."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()):
            azure_tts("Test format", temp_audio_file)
            
            # Verify correct output format is used
            call_args = mock_request.call_args
            headers = call_args[1]['headers']
            assert headers['X-Microsoft-OutputFormat'] == 'riff-16khz-16bit-mono-pcm'
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_correct_content_type(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test that Azure TTS uses correct content type."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()):
            azure_tts("Test content type", temp_audio_file)
            
            # Verify correct content type is used
            call_args = mock_request.call_args
            headers = call_args[1]['headers']
            assert headers['Content-Type'] == 'application/ssml+xml'
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_correct_url(self, mock_request, mock_load_key, temp_audio_file, mock_requests_response):
        """Test that Azure TTS uses correct API URL."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        with patch('builtins.open', mock_open()):
            azure_tts("Test URL", temp_audio_file)
            
            # Verify correct URL is used
            call_args = mock_request.call_args
            assert call_args[0][1] == "https://api.302.ai/cognitiveservices/v1"
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_file_path_handling(self, mock_request, mock_load_key, mock_requests_response):
        """Test Azure TTS with various file path formats."""
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "test_key",
            "azure_tts.voice": "zh-CN-XiaomoNeural"
        }[key]
        mock_request.return_value = mock_requests_response
        
        test_paths = [
            "/absolute/path/test.wav",
            "relative/path/test.wav",
            "./current/dir/test.wav",
            "../parent/dir/test.wav"
        ]
        
        for test_path in test_paths:
            with patch('builtins.open', mock_open()) as mock_file:
                azure_tts("Test path handling", test_path)
                
                # Verify file is opened with correct path
                mock_file.assert_called_once_with(test_path, 'wb')


class TestAzureTtsIntegration:
    """Integration tests for Azure TTS functionality."""
    
    def test_azure_tts_main_function_signature(self):
        """Test that azure_tts main function has correct signature."""
        import inspect
        from core.tts_backend.azure_tts import azure_tts
        
        sig = inspect.signature(azure_tts)
        params = list(sig.parameters.keys())
        
        assert params == ['text', 'save_path']
        assert sig.parameters['text'].annotation == str
        assert sig.parameters['save_path'].annotation == str
        assert sig.return_annotation == type(None)
    
    @patch('core.tts_backend.azure_tts.load_key')
    @patch('core.tts_backend.azure_tts.requests.request')
    def test_azure_tts_realistic_workflow(self, mock_request, mock_load_key, tmp_path):
        """Test realistic Azure TTS workflow."""
        # Setup realistic mock responses
        mock_load_key.side_effect = lambda key: {
            "azure_tts.api_key": "sk-proj-realistic_key_12345",
            "azure_tts.voice": "zh-CN-XiaoxiaoNeural"
        }[key]
        
        # Mock realistic audio response
        mock_response = Mock()
        mock_response.content = b"\x52\x49\x46\x46" + b"\x00" * 1000  # RIFF header + fake audio data
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        test_audio_file = str(tmp_path / "realistic_test.wav")
        realistic_text = "欢迎使用VideoLingo！这是一个视频翻译和配音应用程序。"
        
        with patch('builtins.open', mock_open()) as mock_file:
            azure_tts(realistic_text, test_audio_file)
            
            # Verify realistic SSML structure
            call_args = mock_request.call_args
            ssml_data = call_args[1]['data']
            
            expected_ssml = f"<speak version='1.0' xml:lang='zh-CN'><voice name='zh-CN-XiaoxiaoNeural'>{realistic_text}</voice></speak>"
            assert ssml_data == expected_ssml
            
            # Verify file handling
            mock_file.assert_called_once_with(test_audio_file, 'wb')
            mock_file().write.assert_called_once_with(mock_response.content)
