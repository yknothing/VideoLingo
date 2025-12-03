import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import json

from core.tts_backend.fish_tts import fish_tts


class TestFishTts:
    """Test Fish TTS integration functionality."""
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "fish_test.wav")
    
    @pytest.fixture
    def mock_successful_response(self):
        """Create mock successful requests response."""
        mock_response = Mock()
        mock_response.json.return_value = {"url": "https://api.302.ai/download/audio123.wav"}
        mock_response.raise_for_status = Mock()
        return mock_response
    
    @pytest.fixture
    def mock_audio_response(self):
        """Create mock audio download response."""
        mock_response = Mock()
        mock_response.content = b"fake_fish_tts_audio_data_wav"
        mock_response.raise_for_status = Mock()
        return mock_response
    
    @pytest.fixture
    def mock_fish_config(self):
        """Create mock Fish TTS configuration."""
        return {
            "api_key": "test_fish_api_key_123",
            "character": "test_character",
            "character_id_dict": {
                "test_character": "char_12345abcde",
                "another_char": "char_67890fghij"
            }
        }
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_success(self, mock_get, mock_post, mock_load_key, temp_audio_file, 
                             mock_successful_response, mock_audio_response, mock_fish_config):
        """Test successful Fish TTS generation."""
        # Setup mocks
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = fish_tts("Hello, world!", temp_audio_file)
            
            assert result is True
            
            # Verify API call
            mock_post.assert_called_once_with(
                "https://api.302.ai/fish-audio/v1/tts",
                headers={
                    'Authorization': 'Bearer test_fish_api_key_123',
                    'Content-Type': 'application/json'
                },
                data=json.dumps({
                    "text": "Hello, world!",
                    "reference_id": "char_12345abcde",
                    "chunk_length": 200,
                    "normalize": True,
                    "format": "wav",
                    "latency": "normal"
                })
            )
            
            # Verify audio download
            mock_get.assert_called_once_with("https://api.302.ai/download/audio123.wav")
            
            # Verify file writing
            mock_file.assert_called_once_with(temp_audio_file, "wb")
            mock_file().write.assert_called_once_with(b"fake_fish_tts_audio_data_wav")
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_different_character(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                         mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with different character configuration."""
        # Modify config for different character
        config_copy = mock_fish_config.copy()
        config_copy["character"] = "another_char"
        
        mock_load_key.side_effect = lambda key: config_copy[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        with patch('builtins.open', mock_open()):
            result = fish_tts("Testing different character", temp_audio_file)
            
            assert result is True
            
            # Verify correct character ID is used
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['reference_id'] == "char_67890fghij"
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    def test_fish_tts_api_error(self, mock_post, mock_load_key, temp_audio_file, mock_fish_config):
        """Test Fish TTS with API error response."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        
        # Mock API error
        mock_post.side_effect = requests.exceptions.HTTPError("API Error")
        
        with pytest.raises(requests.exceptions.HTTPError):
            fish_tts("Test API error", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    def test_fish_tts_no_url_in_response(self, mock_post, mock_load_key, temp_audio_file, mock_fish_config):
        """Test Fish TTS when response doesn't contain URL."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        
        # Mock response without URL
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Failed to generate audio"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        with patch('builtins.print') as mock_print:
            result = fish_tts("Test no URL", temp_audio_file)
            
            assert result is False
            mock_print.assert_called_with("Request failed:", {"error": "Failed to generate audio"})
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_audio_download_error(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                          mock_successful_response, mock_fish_config):
        """Test Fish TTS with audio download error."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        
        # Mock audio download error
        mock_get.side_effect = requests.exceptions.HTTPError("Download failed")
        
        with pytest.raises(requests.exceptions.HTTPError):
            fish_tts("Test download error", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_empty_text(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with empty text."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        with patch('builtins.open', mock_open()):
            result = fish_tts("", temp_audio_file)
            
            assert result is True
            
            # Verify empty text is sent to API
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['text'] == ""
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_long_text(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                               mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with very long text."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        long_text = "This is a very long text that should be handled properly by Fish TTS. " * 50
        
        with patch('builtins.open', mock_open()):
            result = fish_tts(long_text, temp_audio_file)
            
            assert result is True
            
            # Verify long text is sent to API
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['text'] == long_text
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_unicode_text(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                  mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with Unicode characters."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        unicode_text = "Hello 世界! Bonjour 🌍! Привет мир! こんにちは！"
        
        with patch('builtins.open', mock_open()):
            result = fish_tts(unicode_text, temp_audio_file)
            
            assert result is True
            
            # Verify Unicode text is properly handled
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['text'] == unicode_text
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_special_characters(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                        mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with special characters."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        special_text = "Testing: quotes \"hello\", apostrophes 'world', and symbols @#$%^&*()!"
        
        with patch('builtins.open', mock_open()):
            result = fish_tts(special_text, temp_audio_file)
            
            assert result is True
            
            # Verify special characters are handled
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            assert payload_data['text'] == special_text
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_json_payload_structure(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                            mock_successful_response, mock_audio_response, mock_fish_config):
        """Test that JSON payload has correct structure."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        test_text = "Testing JSON structure"
        
        with patch('builtins.open', mock_open()):
            fish_tts(test_text, temp_audio_file)
            
            # Verify JSON payload structure
            call_args = mock_post.call_args
            payload_data = json.loads(call_args[1]['data'])
            
            expected_keys = {"text", "reference_id", "chunk_length", "normalize", "format", "latency"}
            assert set(payload_data.keys()) == expected_keys
            assert payload_data["text"] == test_text
            assert payload_data["reference_id"] == "char_12345abcde"
            assert payload_data["chunk_length"] == 200
            assert payload_data["normalize"] is True
            assert payload_data["format"] == "wav"
            assert payload_data["latency"] == "normal"
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_headers_structure(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                       mock_successful_response, mock_audio_response, mock_fish_config):
        """Test that request headers have correct structure."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        with patch('builtins.open', mock_open()):
            fish_tts("Testing headers", temp_audio_file)
            
            # Verify headers structure
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            
            assert headers['Authorization'] == 'Bearer test_fish_api_key_123'
            assert headers['Content-Type'] == 'application/json'
    
    def test_fish_tts_missing_api_key(self, temp_audio_file):
        """Test Fish TTS with missing API key."""
        with patch('core.tts_backend.fish_tts.load_key') as mock_load_key:
            mock_load_key.side_effect = KeyError("fish_tts.api_key not found")
            
            with pytest.raises(KeyError):
                fish_tts("Test missing key", temp_audio_file)
    
    def test_fish_tts_missing_character(self, temp_audio_file):
        """Test Fish TTS with missing character configuration."""
        with patch('core.tts_backend.fish_tts.load_key') as mock_load_key:
            def mock_load_key_func(key):
                if key == "fish_tts.api_key":
                    return "test_key"
                elif key == "fish_tts.character":
                    raise KeyError("fish_tts.character not found")
                
            mock_load_key.side_effect = mock_load_key_func
            
            with pytest.raises(KeyError):
                fish_tts("Test missing character", temp_audio_file)
    
    def test_fish_tts_missing_character_id_dict(self, temp_audio_file):
        """Test Fish TTS with missing character ID dictionary."""
        with patch('core.tts_backend.fish_tts.load_key') as mock_load_key:
            def mock_load_key_func(key):
                if key == "fish_tts.api_key":
                    return "test_key"
                elif key == "fish_tts.character":
                    return "test_char"
                elif key == "fish_tts.character_id_dict":
                    raise KeyError("fish_tts.character_id_dict not found")
                
            mock_load_key.side_effect = mock_load_key_func
            
            with pytest.raises(KeyError):
                fish_tts("Test missing dict", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    def test_fish_tts_character_not_in_dict(self, mock_load_key, temp_audio_file):
        """Test Fish TTS when character is not in character ID dictionary."""
        config = {
            "api_key": "test_key",
            "character": "nonexistent_char",
            "character_id_dict": {
                "char1": "id1",
                "char2": "id2"
            }
        }
        mock_load_key.side_effect = lambda key: config[key.split('.')[-1]]
        
        with pytest.raises(KeyError):
            fish_tts("Test missing char in dict", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_file_write_error(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                      mock_successful_response, mock_audio_response, mock_fish_config):
        """Test Fish TTS with file write error."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        mock_get.return_value = mock_audio_response
        
        # Mock file write error
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Cannot write to file")
            
            with pytest.raises(IOError):
                fish_tts("Test file error", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    def test_fish_tts_network_timeout(self, mock_post, mock_load_key, temp_audio_file, mock_fish_config):
        """Test Fish TTS with network timeout."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(requests.exceptions.Timeout):
            fish_tts("Test timeout", temp_audio_file)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_fish_tts_different_audio_formats(self, mock_get, mock_post, mock_load_key, temp_audio_file,
                                             mock_successful_response, mock_fish_config):
        """Test Fish TTS with different audio content types."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        mock_post.return_value = mock_successful_response
        
        # Test different audio content types
        test_contents = [
            b"RIFF" + b"\x00" * 100,  # WAV format
            b"\xFF\xFB" + b"\x00" * 100,  # MP3 format
            b"OggS" + b"\x00" * 100,  # OGG format
            b"\x00" * 100,  # Generic binary
        ]
        
        for content in test_contents:
            mock_audio_response = Mock()
            mock_audio_response.content = content
            mock_audio_response.raise_for_status = Mock()
            mock_get.return_value = mock_audio_response
            
            with patch('builtins.open', mock_open()) as mock_file:
                result = fish_tts("Test audio formats", temp_audio_file)
                
                assert result is True
                mock_file().write.assert_called_once_with(content)
                mock_file.reset_mock()


class TestFishTtsDecorator:
    """Test the @except_handler decorator functionality."""
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    def test_decorator_retry_logic(self, mock_post, mock_load_key, tmp_path, mock_fish_config):
        """Test that decorator retries failed requests."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        
        # First two calls fail, third succeeds
        mock_successful_response = Mock()
        mock_successful_response.json.return_value = {"url": "https://api.302.ai/download/audio.wav"}
        mock_successful_response.raise_for_status = Mock()
        
        mock_post.side_effect = [
            requests.exceptions.RequestException("First failure"),
            requests.exceptions.RequestException("Second failure"),
            mock_successful_response
        ]
        
        audio_file = str(tmp_path / "retry_test.wav")
        
        with patch('core.tts_backend.fish_tts.requests.get') as mock_get:
            mock_audio_response = Mock()
            mock_audio_response.content = b"success"
            mock_audio_response.raise_for_status = Mock()
            mock_get.return_value = mock_audio_response
            
            with patch('builtins.open', mock_open()):
                result = fish_tts("Test retry", audio_file)
                
                assert result is True
                # Should be called 3 times (2 failures + 1 success)
                assert mock_post.call_count == 3
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    def test_decorator_max_retries_exceeded(self, mock_post, mock_load_key, tmp_path, mock_fish_config):
        """Test that decorator raises exception after max retries."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        
        # All calls fail
        mock_post.side_effect = requests.exceptions.RequestException("Persistent failure")
        
        audio_file = str(tmp_path / "max_retry_test.wav")
        
        with pytest.raises(Exception, match="Failed to generate audio using 302.ai Fish TTS"):
            fish_tts("Test max retries", audio_file)
            
            # Should be called 3 times (max retries)
            assert mock_post.call_count == 3


class TestFishTtsIntegration:
    """Integration tests for Fish TTS functionality."""
    
    def test_fish_tts_function_signature(self):
        """Test that fish_tts function has correct signature."""
        import inspect
        
        sig = inspect.signature(fish_tts)
        params = list(sig.parameters.keys())
        
        assert params == ['text', 'save_as']
        assert sig.parameters['text'].annotation == str
        assert sig.parameters['save_as'].annotation == str
        assert sig.return_annotation == bool
    
    def test_fish_tts_main_function_exists(self):
        """Test that main function can be imported."""
        from core.tts_backend.fish_tts import fish_tts
        assert callable(fish_tts)
    
    def test_fish_tts_module_structure(self):
        """Test that main module has expected structure."""
        import core.tts_backend.fish_tts as fish_module
        
        # Check main function
        assert hasattr(fish_module, 'fish_tts')
        assert callable(fish_module.fish_tts)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_realistic_integration_workflow(self, mock_get, mock_post, mock_load_key, tmp_path):
        """Test realistic Fish TTS integration workflow."""
        # Setup realistic configuration
        realistic_config = {
            "api_key": "sk-302ai-fish-realistic-key-12345",
            "character": "xiaomai_voice",
            "character_id_dict": {
                "xiaomai_voice": "xiaomai_char_id_abcde12345",
                "xiaoming_voice": "xiaoming_char_id_67890fghij"
            }
        }
        mock_load_key.side_effect = lambda key: realistic_config[key.split('.')[-1]]
        
        # Mock realistic API responses
        mock_tts_response = Mock()
        mock_tts_response.json.return_value = {
            "url": "https://api.302.ai/fish-audio/downloads/generated_12345.wav",
            "duration": 2.5,
            "status": "success"
        }
        mock_tts_response.raise_for_status = Mock()
        mock_post.return_value = mock_tts_response
        
        mock_audio_response = Mock()
        mock_audio_response.content = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 2000  # Realistic WAV data
        mock_audio_response.raise_for_status = Mock()
        mock_get.return_value = mock_audio_response
        
        test_audio_file = str(tmp_path / "integration_test.wav")
        realistic_text = "欢迎使用VideoLingo！这是一个全面的视频翻译和配音应用程序。"
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = fish_tts(realistic_text, test_audio_file)
            
            assert result is True
            
            # Verify realistic TTS API call
            tts_call_args = mock_post.call_args
            assert tts_call_args[0][0] == "https://api.302.ai/fish-audio/v1/tts"
            
            headers = tts_call_args[1]['headers']
            assert headers['Authorization'] == 'Bearer sk-302ai-fish-realistic-key-12345'
            assert headers['Content-Type'] == 'application/json'
            
            payload = json.loads(tts_call_args[1]['data'])
            assert payload['text'] == realistic_text
            assert payload['reference_id'] == 'xiaomai_char_id_abcde12345'
            assert payload['chunk_length'] == 200
            assert payload['normalize'] is True
            assert payload['format'] == 'wav'
            assert payload['latency'] == 'normal'
            
            # Verify audio download
            mock_get.assert_called_once_with("https://api.302.ai/fish-audio/downloads/generated_12345.wav")
            
            # Verify file handling
            mock_file.assert_called_once_with(test_audio_file, "wb")
            mock_file().write.assert_called_once_with(mock_audio_response.content)
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_error_recovery_workflow(self, mock_get, mock_post, mock_load_key, tmp_path, mock_fish_config):
        """Test error recovery in realistic workflow."""
        mock_load_key.side_effect = lambda key: mock_fish_config[key.split('.')[-1]]
        
        # Simulate transient error followed by success
        mock_error_response = Mock()
        mock_error_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        
        mock_success_response = Mock()
        mock_success_response.json.return_value = {"url": "https://api.302.ai/recovered_audio.wav"}
        mock_success_response.raise_for_status = Mock()
        
        mock_post.side_effect = [mock_error_response, mock_success_response]
        
        mock_audio_response = Mock()
        mock_audio_response.content = b"recovered_audio_data"
        mock_audio_response.raise_for_status = Mock()
        mock_get.return_value = mock_audio_response
        
        test_audio_file = str(tmp_path / "recovery_test.wav")
        
        with patch('builtins.open', mock_open()) as mock_file:
            result = fish_tts("Test error recovery", test_audio_file)
            
            assert result is True
            # Should succeed after retry
            assert mock_post.call_count == 2
            mock_file().write.assert_called_once_with(b"recovered_audio_data")
    
    def test_fish_tts_main_module_execution(self):
        """Test that main module execution works."""
        import core.tts_backend.fish_tts as fish_module
        import inspect
        
        # Check if main execution block exists
        source = inspect.getsource(fish_module)
        assert '__main__' in source
        
        # The main block should call fish_tts function
        assert 'fish_tts(' in source
    
    @patch('core.tts_backend.fish_tts.load_key')
    @patch('core.tts_backend.fish_tts.requests.post')
    @patch('core.tts_backend.fish_tts.requests.get')
    def test_multiple_character_configurations(self, mock_get, mock_post, mock_load_key, tmp_path):
        """Test Fish TTS with multiple character configurations."""
        base_config = {
            "api_key": "test_key",
            "character_id_dict": {
                "char1": "id1",
                "char2": "id2",
                "char3": "id3"
            }
        }
        
        characters_to_test = ["char1", "char2", "char3"]
        
        for character in characters_to_test:
            config = base_config.copy()
            config["character"] = character
            
            mock_load_key.side_effect = lambda key: config[key.split('.')[-1]]
            
            mock_response = Mock()
            mock_response.json.return_value = {"url": f"https://api.302.ai/{character}_audio.wav"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            mock_audio = Mock()
            mock_audio.content = f"{character}_audio_data".encode()
            mock_audio.raise_for_status = Mock()
            mock_get.return_value = mock_audio
            
            audio_file = str(tmp_path / f"{character}_test.wav")
            
            with patch('builtins.open', mock_open()) as mock_file:
                result = fish_tts(f"Testing {character}", audio_file)
                
                assert result is True
                
                # Verify correct character ID is used
                call_args = mock_post.call_args
                payload = json.loads(call_args[1]['data'])
                expected_id = base_config["character_id_dict"][character]
                assert payload['reference_id'] == expected_id
                
                # Verify correct audio content
                mock_file().write.assert_called_once_with(f"{character}_audio_data".encode())
