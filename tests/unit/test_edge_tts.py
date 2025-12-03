import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess
from pathlib import Path

from core.tts_backend.edge_tts import edge_tts


class TestEdgeTts:
    """Test Edge TTS integration functionality."""
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "edge_test.wav")
    
    @pytest.fixture
    def mock_edge_config(self):
        """Create mock Edge TTS configuration."""
        return {
            "voice": "en-US-JennyNeural"
        }
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_success(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test successful Edge TTS generation."""
        # Setup mocks
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts("Hello, world!", temp_audio_file)
        
        # Verify subprocess call
        expected_cmd = [
            "edge-tts",
            "--voice", "en-US-JennyNeural",
            "--text", "Hello, world!",
            "--write-media", str(Path(temp_audio_file))
        ]
        mock_subprocess.assert_called_once_with(expected_cmd, check=True)
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_with_different_voice(self, mock_subprocess, mock_load_key, temp_audio_file):
        """Test Edge TTS with different voice configuration."""
        mock_edge_config = {"voice": "zh-CN-XiaoxiaoNeural"}
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts("你好世界", temp_audio_file)
        
        # Verify correct voice is used
        call_args = mock_subprocess.call_args[0][0]
        assert "--voice" in call_args
        voice_index = call_args.index("--voice") + 1
        assert call_args[voice_index] == "zh-CN-XiaoxiaoNeural"
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_default_voice_fallback(self, mock_subprocess, mock_load_key, temp_audio_file):
        """Test Edge TTS falls back to default voice when config is missing."""
        mock_edge_config = {}  # No voice specified
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts("Test default voice", temp_audio_file)
        
        # Should use default voice
        call_args = mock_subprocess.call_args[0][0]
        voice_index = call_args.index("--voice") + 1
        assert call_args[voice_index] == "en-US-JennyNeural"  # Default voice
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_with_special_characters(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with special characters in text."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        special_text = "Hello! How are you? I'm fine. Let's test @#$%^&*()_+"
        edge_tts(special_text, temp_audio_file)
        
        # Verify special characters are passed correctly
        call_args = mock_subprocess.call_args[0][0]
        text_index = call_args.index("--text") + 1
        assert call_args[text_index] == special_text
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_with_empty_text(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with empty text."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts("", temp_audio_file)
        
        # Verify empty text is handled
        call_args = mock_subprocess.call_args[0][0]
        text_index = call_args.index("--text") + 1
        assert call_args[text_index] == ""
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_with_long_text(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with very long text."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        long_text = "This is a very long text that should be handled properly by Edge TTS. " * 50
        edge_tts(long_text, temp_audio_file)
        
        # Verify long text is passed correctly
        call_args = mock_subprocess.call_args[0][0]
        text_index = call_args.index("--text") + 1
        assert call_args[text_index] == long_text
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_with_unicode_text(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with Unicode characters."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        unicode_text = "Hello 世界! Bonjour 🌍! Привет мир! こんにちは世界！"
        edge_tts(unicode_text, temp_audio_file)
        
        # Verify Unicode text is handled correctly
        call_args = mock_subprocess.call_args[0][0]
        text_index = call_args.index("--text") + 1
        assert call_args[text_index] == unicode_text
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_directory_creation(self, mock_subprocess, mock_load_key, mock_edge_config, tmp_path):
        """Test that parent directories are created if they don't exist."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Create nested path that doesn't exist
        nested_audio_file = str(tmp_path / "nested" / "dirs" / "audio.wav")
        
        edge_tts("Test directory creation", nested_audio_file)
        
        # Verify Path object is used correctly
        call_args = mock_subprocess.call_args[0][0]
        media_index = call_args.index("--write-media") + 1
        assert call_args[media_index] == nested_audio_file
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_subprocess_error(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with subprocess error."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "edge-tts", "Command failed")
        
        with pytest.raises(subprocess.CalledProcessError):
            edge_tts("Test subprocess error", temp_audio_file)
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_file_not_found_error(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS when edge-tts command is not found."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.side_effect = FileNotFoundError("edge-tts command not found")
        
        with pytest.raises(FileNotFoundError):
            edge_tts("Test command not found", temp_audio_file)
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_permission_error(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test Edge TTS with permission error."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            edge_tts("Test permission error", temp_audio_file)
    
    @patch('core.tts_backend.edge_tts.load_key')
    def test_edge_tts_config_loading_error(self, mock_load_key, temp_audio_file):
        """Test Edge TTS with configuration loading error."""
        mock_load_key.side_effect = KeyError("edge_tts config not found")
        
        with pytest.raises(KeyError):
            edge_tts("Test config error", temp_audio_file)
    
    @pytest.mark.parametrize("voice", [
        "en-US-JennyNeural",
        "en-US-GuyNeural",
        "en-GB-SoniaNeural",
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunxiNeural",
        "zh-CN-XiaoyiNeural"
    ])
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_common_voices(self, mock_subprocess, mock_load_key, voice, temp_audio_file):
        """Test Edge TTS with common voice configurations."""
        mock_edge_config = {"voice": voice}
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts(f"Testing {voice}", temp_audio_file)
        
        # Verify correct voice is used
        call_args = mock_subprocess.call_args[0][0]
        voice_index = call_args.index("--voice") + 1
        assert call_args[voice_index] == voice
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_command_structure(self, mock_subprocess, mock_load_key, temp_audio_file):
        """Test that Edge TTS command has correct structure."""
        mock_edge_config = {"voice": "en-US-AriaNeural"}
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        test_text = "Testing command structure"
        edge_tts(test_text, temp_audio_file)
        
        # Verify command structure
        call_args = mock_subprocess.call_args[0][0]
        
        assert call_args[0] == "edge-tts"
        assert "--voice" in call_args
        assert "--text" in call_args
        assert "--write-media" in call_args
        
        # Verify order and pairing
        voice_index = call_args.index("--voice")
        text_index = call_args.index("--text")
        media_index = call_args.index("--write-media")
        
        assert call_args[voice_index + 1] == "en-US-AriaNeural"
        assert call_args[text_index + 1] == test_text
        assert call_args[media_index + 1] == str(Path(temp_audio_file))
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_check_parameter(self, mock_subprocess, mock_load_key, temp_audio_file, mock_edge_config):
        """Test that subprocess is called with check=True."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        edge_tts("Test check parameter", temp_audio_file)
        
        # Verify check=True is passed
        call_kwargs = mock_subprocess.call_args[1]
        assert call_kwargs['check'] is True
    
    def test_edge_tts_voice_comments_exist(self):
        """Test that voice comments in the file are accurate."""
        # Read the source file to check comments
        import core.tts_backend.edge_tts as edge_module
        import inspect
        
        source = inspect.getsource(edge_module)
        
        # Check for common voice examples in comments
        expected_voices = [
            "en-US-JennyNeural",
            "en-US-GuyNeural",
            "en-GB-SoniaNeural",
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-XiaoyiNeural"
        ]
        
        for voice in expected_voices:
            assert voice in source
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_path_handling(self, mock_subprocess, mock_load_key, mock_edge_config):
        """Test Edge TTS with various file path formats."""
        mock_load_key.return_value = mock_edge_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        test_paths = [
            "/absolute/path/test.wav",
            "relative/path/test.wav",
            "./current/dir/test.wav",
            "../parent/dir/test.wav",
            "simple_filename.wav"
        ]
        
        for test_path in test_paths:
            edge_tts("Test path handling", test_path)
            
            # Verify path is converted to Path object
            call_args = mock_subprocess.call_args[0][0]
            media_index = call_args.index("--write-media") + 1
            assert call_args[media_index] == str(Path(test_path))


class TestEdgeTtsIntegration:
    """Integration tests for Edge TTS functionality."""
    
    def test_edge_tts_function_signature(self):
        """Test that edge_tts function has correct signature."""
        import inspect
        from core.tts_backend.edge_tts import edge_tts
        
        sig = inspect.signature(edge_tts)
        params = list(sig.parameters.keys())
        
        assert params == ['text', 'save_path']
        # Check that function exists and is callable
        assert callable(edge_tts)
    
    def test_edge_tts_module_imports(self):
        """Test that edge_tts module imports are correct."""
        import core.tts_backend.edge_tts as edge_module
        
        # Check required imports
        assert hasattr(edge_module, 'Path')
        assert hasattr(edge_module, 'edge_tts')
        assert hasattr(edge_module, 'subprocess')
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_realistic_edge_tts_workflow(self, mock_subprocess, mock_load_key, tmp_path):
        """Test realistic Edge TTS workflow."""
        # Setup realistic configuration
        realistic_config = {
            "voice": "en-US-AriaNeural"
        }
        mock_load_key.return_value = realistic_config
        mock_subprocess.return_value = Mock(returncode=0)
        
        test_audio_file = str(tmp_path / "realistic_test.wav")
        realistic_text = "Welcome to VideoLingo! This application provides comprehensive video translation and dubbing capabilities."
        
        edge_tts(realistic_text, test_audio_file)
        
        # Verify realistic command structure
        call_args = mock_subprocess.call_args[0][0]
        expected_cmd = [
            "edge-tts",
            "--voice", "en-US-AriaNeural",
            "--text", realistic_text,
            "--write-media", str(Path(test_audio_file))
        ]
        assert call_args == expected_cmd
        
        # Verify check=True parameter
        call_kwargs = mock_subprocess.call_args[1]
        assert call_kwargs['check'] is True
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_multilingual_edge_tts_workflow(self, mock_subprocess, mock_load_key, tmp_path):
        """Test Edge TTS with multilingual content."""
        # Test different language configurations
        language_configs = [
            ("zh-CN-XiaoxiaoNeural", "你好，欢迎使用VideoLingo！这是一个视频翻译应用程序。"),
            ("en-GB-SoniaNeural", "Hello, welcome to VideoLingo! This is a British English voice."),
            ("en-US-JennyNeural", "Hi there! Welcome to VideoLingo, your video translation solution.")
        ]
        
        for voice, text in language_configs:
            config = {"voice": voice}
            mock_load_key.return_value = config
            mock_subprocess.return_value = Mock(returncode=0)
            
            audio_file = str(tmp_path / f"{voice}_test.wav")
            edge_tts(text, audio_file)
            
            # Verify language-specific configuration
            call_args = mock_subprocess.call_args[0][0]
            voice_index = call_args.index("--voice") + 1
            text_index = call_args.index("--text") + 1
            
            assert call_args[voice_index] == voice
            assert call_args[text_index] == text
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_error_scenarios(self, mock_subprocess, mock_load_key, tmp_path):
        """Test Edge TTS error handling scenarios."""
        config = {"voice": "en-US-JennyNeural"}
        mock_load_key.return_value = config
        
        # Test various error scenarios
        error_scenarios = [
            subprocess.CalledProcessError(1, "edge-tts", "Voice not found"),
            subprocess.CalledProcessError(2, "edge-tts", "Network error"),
            FileNotFoundError("edge-tts command not found"),
            PermissionError("Permission denied"),
            OSError("System error")
        ]
        
        audio_file = str(tmp_path / "error_test.wav")
        
        for error in error_scenarios:
            mock_subprocess.side_effect = error
            
            with pytest.raises(type(error)):
                edge_tts("Test error handling", audio_file)
    
    def test_edge_tts_main_module_execution(self):
        """Test that main module execution works."""
        import core.tts_backend.edge_tts as edge_module
        
        # Check if main execution block exists
        source = inspect.getsource(edge_module)
        assert '__main__' in source
        
        # The main block should call edge_tts function
        assert 'edge_tts(' in source
    
    @patch('core.tts_backend.edge_tts.load_key')
    @patch('core.tts_backend.edge_tts.subprocess.run')
    def test_edge_tts_concurrent_calls(self, mock_subprocess, mock_load_key, tmp_path):
        """Test Edge TTS with concurrent calls simulation."""
        config = {"voice": "en-US-JennyNeural"}
        mock_load_key.return_value = config
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Simulate multiple concurrent calls
        texts_and_files = [
            ("First concurrent call", str(tmp_path / "concurrent_1.wav")),
            ("Second concurrent call", str(tmp_path / "concurrent_2.wav")),
            ("Third concurrent call", str(tmp_path / "concurrent_3.wav"))
        ]
        
        for text, audio_file in texts_and_files:
            edge_tts(text, audio_file)
            
            # Verify each call is handled independently
            call_args = mock_subprocess.call_args[0][0]
            text_index = call_args.index("--text") + 1
            media_index = call_args.index("--write-media") + 1
            
            assert call_args[text_index] == text
            assert call_args[media_index] == audio_file
