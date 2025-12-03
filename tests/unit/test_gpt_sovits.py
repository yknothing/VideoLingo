import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import requests
import socket
import time
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

from core.tts_backend.gpt_sovits_tts import (
    check_lang, gpt_sovits_tts, gpt_sovits_tts_for_videolingo,
    find_and_check_config_path, start_gpt_sovits_server
)


class TestCheckLang:
    """Test language checking functionality."""
    
    def test_check_lang_chinese_variants(self):
        """Test Chinese language detection variants."""
        chinese_variants = [
            ('zh', 'zh'),
            ('cn', 'zh'),
            ('中文', 'zh'),
            ('chinese', 'zh'),
            ('zh-CN', 'zh'),
            ('Chinese', 'zh'),
            ('CHINESE', 'zh')
        ]
        
        for input_lang, expected in chinese_variants:
            result_text, result_prompt = check_lang(input_lang, 'en')
            assert result_text == expected
            
    def test_check_lang_english_variants(self):
        """Test English language detection variants."""
        english_variants = [
            ('en', 'en'),
            ('english', 'en'),
            ('英文', 'en'),
            ('英语', 'en'),
            ('English', 'en'),
            ('ENGLISH', 'en')
        ]
        
        for input_lang, expected in english_variants:
            result_text, result_prompt = check_lang(input_lang, 'en')
            assert result_text == expected
    
    def test_check_lang_prompt_language_detection(self):
        """Test prompt language detection."""
        prompt_variants = [
            ('en', 'en'),
            ('english', 'en'),
            ('英文', 'en'),
            ('英语', 'en'),
            ('zh', 'zh'),
            ('cn', 'zh'),
            ('中文', 'zh'),
            ('chinese', 'zh')
        ]
        
        for input_lang, expected in prompt_variants:
            result_text, result_prompt = check_lang('en', input_lang)
            assert result_prompt == expected
    
    def test_check_lang_unsupported_text_language(self):
        """Test unsupported text language raises error."""
        unsupported_languages = ['fr', 'de', 'es', 'ja', 'ko', 'invalid']
        
        for lang in unsupported_languages:
            with pytest.raises(ValueError, match="Unsupported text language"):
                check_lang(lang, 'en')
    
    def test_check_lang_unsupported_prompt_language(self):
        """Test unsupported prompt language raises error."""
        unsupported_languages = ['fr', 'de', 'es', 'ja', 'ko', 'invalid']
        
        for lang in unsupported_languages:
            with pytest.raises(ValueError, match="Unsupported prompt language"):
                check_lang('en', lang)
    
    def test_check_lang_both_languages_processing(self):
        """Test both languages are processed correctly."""
        test_cases = [
            (('zh', 'en'), ('zh', 'en')),
            (('chinese', 'english'), ('zh', 'en')),
            (('中文', '英文'), ('zh', 'en')),
            (('en', 'zh'), ('en', 'zh')),
            (('english', 'chinese'), ('en', 'zh'))
        ]
        
        for (text_lang, prompt_lang), (expected_text, expected_prompt) in test_cases:
            result_text, result_prompt = check_lang(text_lang, prompt_lang)
            assert result_text == expected_text
            assert result_prompt == expected_prompt


class TestGptSovitsTts:
    """Test core GPT-SoVITS TTS functionality."""
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "gpt_sovits_test.wav")
    
    @pytest.fixture
    def mock_successful_response(self):
        """Create mock successful requests response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_gpt_sovits_audio_data"
        return mock_response
    
    @pytest.fixture
    def mock_error_response(self):
        """Create mock error requests response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.content = b""
        return mock_response
    
    @patch('core.tts_backend.gpt_sovits_tts.requests.post')
    @patch('core.tts_backend.gpt_sovits_tts.Path.cwd')
    def test_gpt_sovits_tts_success(self, mock_cwd, mock_post, temp_audio_file, mock_successful_response, tmp_path):
        """Test successful GPT-SoVITS TTS generation."""
        mock_cwd.return_value = tmp_path
        mock_post.return_value = mock_successful_response
        
        ref_audio = str(tmp_path / "reference.wav")
        
        result = gpt_sovits_tts(
            text="Hello world", 
            text_lang="en",
            save_path=temp_audio_file,
            ref_audio_path=ref_audio,
            prompt_lang="en",
            prompt_text="Hello"
        )
        
        assert result is True
        
        # Verify API call
        expected_payload = {
            'text': "Hello world",
            'text_lang': "en", 
            'ref_audio_path': ref_audio,
            'prompt_lang': "en",
            'prompt_text': "Hello",
            "speed_factor": 1.0,
        }
        mock_post.assert_called_once_with('http://127.0.0.1:9880/tts', json=expected_payload)
    
    @patch('core.tts_backend.gpt_sovits_tts.requests.post')
    @patch('core.tts_backend.gpt_sovits_tts.Path.cwd')
    def test_gpt_sovits_tts_error_response(self, mock_cwd, mock_post, temp_audio_file, mock_error_response, tmp_path):
        """Test GPT-SoVITS TTS with error response."""
        mock_cwd.return_value = tmp_path
        mock_post.return_value = mock_error_response
        
        ref_audio = str(tmp_path / "reference.wav")
        
        result = gpt_sovits_tts(
            text="Test error",
            text_lang="en", 
            save_path=temp_audio_file,
            ref_audio_path=ref_audio,
            prompt_lang="en",
            prompt_text="Test"
        )
        
        assert result is False
    
    @patch('core.tts_backend.gpt_sovits_tts.requests.post')
    def test_gpt_sovits_tts_language_validation(self, mock_post, temp_audio_file, mock_successful_response, tmp_path):
        """Test that languages are validated through check_lang."""
        mock_post.return_value = mock_successful_response
        ref_audio = str(tmp_path / "reference.wav")
        
        with patch('core.tts_backend.gpt_sovits_tts.Path.cwd', return_value=tmp_path):
            # Test Chinese language variants
            result = gpt_sovits_tts(
                text="测试",
                text_lang="chinese",  # Should be converted to 'zh'
                save_path=temp_audio_file,
                ref_audio_path=ref_audio,
                prompt_lang="中文",  # Should be converted to 'zh'
                prompt_text="测试"
            )
            
            assert result is True
            
            # Verify converted languages are used in API call
            call_args = mock_post.call_args[1]['json']
            assert call_args['text_lang'] == 'zh'
            assert call_args['prompt_lang'] == 'zh'
    
    def test_gpt_sovits_tts_unsupported_language(self, temp_audio_file, tmp_path):
        """Test GPT-SoVITS TTS with unsupported language."""
        ref_audio = str(tmp_path / "reference.wav")
        
        with pytest.raises(ValueError, match="Unsupported text language"):
            gpt_sovits_tts(
                text="Test",
                text_lang="french",  # Unsupported
                save_path=temp_audio_file,
                ref_audio_path=ref_audio,
                prompt_lang="en",
                prompt_text="Test"
            )
    
    @patch('core.tts_backend.gpt_sovits_tts.requests.post')
    def test_gpt_sovits_tts_file_saving(self, mock_post, temp_audio_file, mock_successful_response, tmp_path):
        """Test file saving functionality."""
        mock_post.return_value = mock_successful_response
        ref_audio = str(tmp_path / "reference.wav")
        
        with patch('core.tts_backend.gpt_sovits_tts.Path.cwd', return_value=tmp_path):
            with patch('builtins.open', mock_open()) as mock_file:
                result = gpt_sovits_tts(
                    text="Test file save",
                    text_lang="en",
                    save_path=temp_audio_file,
                    ref_audio_path=ref_audio,
                    prompt_lang="en",
                    prompt_text="Test"
                )
                
                assert result is True
                
                # Verify file is written with correct content
                expected_path = tmp_path / temp_audio_file
                mock_file().write.assert_called_once_with(b"fake_gpt_sovits_audio_data")


class TestFindAndCheckConfigPath:
    """Test configuration path finding and validation."""
    
    @patch('core.tts_backend.gpt_sovits_tts.Path.__file__')
    def test_find_and_check_config_path_success(self, mock_file, tmp_path):
        """Test successful config path finding."""
        # Setup mock file structure
        mock_file = tmp_path / "core" / "tts_backend" / "gpt_sovits_tts.py"
        
        # Create mock GPT-SoVITS directory
        gpt_sovits_dir = tmp_path / "GPT-SoVITS-v2-test"
        config_dir = gpt_sovits_dir / "GPT_SoVITS" / "configs"
        config_dir.mkdir(parents=True)
        
        # Create mock config file
        config_file = config_dir / "test_character.yaml"
        config_file.touch()
        
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_resolve.return_value.parent.parent.parent = tmp_path
            
            with patch('pathlib.Path.iterdir') as mock_iterdir:
                mock_dir = Mock()
                mock_dir.is_dir.return_value = True
                mock_dir.name = "GPT-SoVITS-v2-test"
                mock_iterdir.return_value = [mock_dir]
                
                with patch('pathlib.Path.__truediv__') as mock_div:
                    mock_div.return_value = gpt_sovits_dir
                    
                    gpt_dir, config_path = find_and_check_config_path("test_character")
                    
                    assert gpt_dir == gpt_sovits_dir
    
    @patch('core.tts_backend.gpt_sovits_tts.Path.__file__')
    def test_find_and_check_config_path_no_directory(self, mock_file, tmp_path):
        """Test config path finding when GPT-SoVITS directory doesn't exist."""
        mock_file = tmp_path / "core" / "tts_backend" / "gpt_sovits_tts.py"
        
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_resolve.return_value.parent.parent.parent = tmp_path
            
            with patch('pathlib.Path.iterdir') as mock_iterdir:
                mock_iterdir.return_value = []  # No directories
                
                with pytest.raises(FileNotFoundError, match="GPT-SoVITS-v2 directory not found"):
                    find_and_check_config_path("test_character")
    
    def test_find_and_check_config_path_no_config_file(self, tmp_path):
        """Test config path finding when config file doesn't exist."""
        # Create GPT-SoVITS directory without config file
        gpt_sovits_dir = tmp_path / "GPT-SoVITS-v2-test"
        config_dir = gpt_sovits_dir / "GPT_SoVITS" / "configs"
        config_dir.mkdir(parents=True)
        
        with patch('pathlib.Path.resolve') as mock_resolve:
            mock_resolve.return_value.parent.parent.parent = tmp_path
            
            with patch('pathlib.Path.iterdir') as mock_iterdir:
                mock_dir = Mock()
                mock_dir.is_dir.return_value = True
                mock_dir.name = "GPT-SoVITS-v2-test"
                mock_iterdir.return_value = [mock_dir]
                
                with patch('pathlib.Path.__truediv__') as mock_div:
                    mock_div.return_value = gpt_sovits_dir
                    
                    with pytest.raises(FileNotFoundError, match="Config file not found"):
                        find_and_check_config_path("nonexistent_character")


class TestStartGptSovitsServer:
    """Test GPT-SoVITS server startup functionality."""
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    def test_start_gpt_sovits_server_already_running(self, mock_socket):
        """Test server startup when server is already running."""
        # Mock socket to indicate port is in use
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0  # Port is in use
        mock_socket.return_value = mock_sock
        
        result = start_gpt_sovits_server()
        
        assert result is None
        mock_sock.close.assert_called_once()
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    @patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.os.chdir')
    @patch('core.tts_backend.gpt_sovits_tts.subprocess.Popen')
    @patch('core.tts_backend.gpt_sovits_tts.requests.get')
    @patch('core.tts_backend.gpt_sovits_tts.time.sleep')
    def test_start_gpt_sovits_server_windows(self, mock_sleep, mock_requests, mock_popen, 
                                           mock_chdir, mock_load_key, mock_find_config, 
                                           mock_socket, tmp_path):
        """Test server startup on Windows."""
        # Setup mocks
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1  # Port is not in use
        mock_socket.return_value = mock_sock
        
        mock_find_config.return_value = (tmp_path, tmp_path / "config.yaml")
        mock_load_key.return_value = "test_character"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        with patch('core.tts_backend.gpt_sovits_tts.sys.platform', 'win32'):
            result = start_gpt_sovits_server()
            
            # Verify subprocess is started with correct parameters
            expected_cmd = [
                "runtime\\python.exe",
                "api_v2.py",
                "-a", "127.0.0.1",
                "-p", "9880", 
                "-c", str(tmp_path / "config.yaml")
            ]
            mock_popen.assert_called_once_with(expected_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    @patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    def test_start_gpt_sovits_server_macos_manual_start(self, mock_load_key, mock_find_config, mock_socket, tmp_path):
        """Test server startup on macOS requiring manual start."""
        # Setup mocks
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1  # Port is not in use
        mock_socket.return_value = mock_sock
        
        mock_find_config.return_value = (tmp_path, tmp_path / "config.yaml")
        mock_load_key.return_value = "test_character"
        
        with patch('core.tts_backend.gpt_sovits_tts.sys.platform', 'darwin'):
            with patch('builtins.input', return_value='y'):  # User confirms server started
                result = start_gpt_sovits_server()
                
                assert result is None
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    @patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    def test_start_gpt_sovits_server_macos_user_declines(self, mock_load_key, mock_find_config, mock_socket, tmp_path):
        """Test server startup on macOS when user declines to start."""
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock
        
        mock_find_config.return_value = (tmp_path, tmp_path / "config.yaml")
        mock_load_key.return_value = "test_character"
        
        with patch('core.tts_backend.gpt_sovits_tts.sys.platform', 'darwin'):
            with patch('builtins.input', return_value='n'):  # User declines
                with pytest.raises(Exception, match="Please start the server before continuing"):
                    start_gpt_sovits_server()
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    def test_start_gpt_sovits_server_unsupported_os(self, mock_socket):
        """Test server startup on unsupported OS."""
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock
        
        with patch('core.tts_backend.gpt_sovits_tts.sys.platform', 'linux'):
            with pytest.raises(OSError, match="Unsupported operating system"):
                start_gpt_sovits_server()
    
    @patch('core.tts_backend.gpt_sovits_tts.socket.socket')
    @patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.os.chdir')
    @patch('core.tts_backend.gpt_sovits_tts.subprocess.Popen')
    @patch('core.tts_backend.gpt_sovits_tts.requests.get')
    @patch('core.tts_backend.gpt_sovits_tts.time.sleep')
    def test_start_gpt_sovits_server_timeout(self, mock_sleep, mock_requests, mock_popen,
                                           mock_chdir, mock_load_key, mock_find_config, mock_socket, tmp_path):
        """Test server startup timeout."""
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock
        
        mock_find_config.return_value = (tmp_path, tmp_path / "config.yaml")
        mock_load_key.return_value = "test_character"
        
        # Mock requests to always fail
        mock_requests.side_effect = requests.exceptions.RequestException("Connection failed")
        
        with patch('core.tts_backend.gpt_sovits_tts.sys.platform', 'win32'):
            with pytest.raises(Exception, match="GPT-SoVITS server failed to start within 50 seconds"):
                start_gpt_sovits_server()


class TestGptSovitsTtsForVideolingo:
    """Test VideoLingo-specific GPT-SoVITS TTS functionality."""
    
    @pytest.fixture
    def sample_task_df(self):
        """Create sample task DataFrame."""
        return pd.DataFrame({
            'number': [1, 2, 3],
            'origin': ['Original text 1', 'Original text 2', 'Original text 3']
        })
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "videolingo_test.wav")
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.gpt_sovits_tts')
    def test_gpt_sovits_tts_for_videolingo_mode_1(self, mock_gpt_tts, mock_load_key, 
                                                  mock_start_server, temp_audio_file, 
                                                  sample_task_df, tmp_path):
        """Test VideoLingo TTS with mode 1 (default reference)."""
        # Setup mocks
        mock_load_key_values = {
            "target_language": "en",
            "whisper.language": "en",
            "gpt_sovits": {"character": "test_char", "refer_mode": 1},
            "whisper.detected_language": "en"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        mock_gpt_tts.return_value = True
        
        # Mock config path finding
        config_dir = tmp_path / "GPT_SoVITS" / "configs"
        config_dir.mkdir(parents=True)
        ref_audio = config_dir / "test_char_reference.wav"
        ref_audio.touch()
        
        with patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path') as mock_find:
            mock_find.return_value = (tmp_path, tmp_path / "config.yaml")
            
            gpt_sovits_tts_for_videolingo("Test text", temp_audio_file, 1, sample_task_df)
            
            mock_gpt_tts.assert_called_once()
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.get_storage_paths')
    @patch('core.tts_backend.gpt_sovits_tts.gpt_sovits_tts')
    def test_gpt_sovits_tts_for_videolingo_mode_2(self, mock_gpt_tts, mock_storage, 
                                                  mock_load_key, mock_start_server, 
                                                  temp_audio_file, sample_task_df, tmp_path):
        """Test VideoLingo TTS with mode 2 (global reference)."""
        # Setup mocks
        mock_load_key_values = {
            "target_language": "zh",
            "whisper.language": "zh", 
            "gpt_sovits": {"character": "test_char", "refer_mode": 2},
            "whisper.detected_language": "zh"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        mock_storage.return_value = {"temp": str(tmp_path)}
        mock_gpt_tts.return_value = True
        
        # Create reference audio file
        ref_dir = tmp_path / "audio" / "refers"
        ref_dir.mkdir(parents=True)
        ref_audio = ref_dir / "1.wav"
        ref_audio.touch()
        
        gpt_sovits_tts_for_videolingo("测试文本", temp_audio_file, 1, sample_task_df)
        
        mock_gpt_tts.assert_called_once()
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.get_storage_paths')
    @patch('core.tts_backend.gpt_sovits_tts.gpt_sovits_tts')
    def test_gpt_sovits_tts_for_videolingo_mode_3_with_fallback(self, mock_gpt_tts, mock_storage,
                                                               mock_load_key, mock_start_server,
                                                               temp_audio_file, sample_task_df, tmp_path):
        """Test VideoLingo TTS with mode 3 and fallback to mode 2."""
        # Setup mocks
        mock_load_key_values = {
            "target_language": "en", 
            "whisper.language": "en",
            "gpt_sovits": {"character": "test_char", "refer_mode": 3},
            "whisper.detected_language": "en"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        mock_storage.return_value = {"temp": str(tmp_path)}
        
        # First call fails, second succeeds (fallback)
        mock_gpt_tts.side_effect = [False, True]
        
        # Create reference audio files
        ref_dir = tmp_path / "audio" / "refers"
        ref_dir.mkdir(parents=True)
        (ref_dir / "1.wav").touch()  # Global reference
        (ref_dir / "1.wav").touch()  # Per-segment reference
        
        gpt_sovits_tts_for_videolingo("Test fallback", temp_audio_file, 1, sample_task_df)
        
        # Should be called twice (original + fallback)
        assert mock_gpt_tts.call_count == 2
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.get_storage_paths')
    def test_gpt_sovits_tts_for_videolingo_missing_reference_extraction(self, mock_storage, mock_load_key, 
                                                                       mock_start_server, temp_audio_file, 
                                                                       sample_task_df, tmp_path):
        """Test reference audio extraction when file doesn't exist."""
        mock_load_key_values = {
            "target_language": "en",
            "whisper.language": "en",
            "gpt_sovits": {"character": "test_char", "refer_mode": 2},
            "whisper.detected_language": "en"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        mock_storage.return_value = {"temp": str(tmp_path)}
        
        # Reference file doesn't exist
        ref_dir = tmp_path / "audio" / "refers"
        ref_dir.mkdir(parents=True)
        # Don't create the reference file
        
        with patch('core.tts_backend.gpt_sovits_tts.extract_refer_audio_main') as mock_extract:
            with patch('core.tts_backend.gpt_sovits_tts.gpt_sovits_tts', return_value=True):
                gpt_sovits_tts_for_videolingo("Test extraction", temp_audio_file, 1, sample_task_df)
                
                mock_extract.assert_called_once()
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    def test_gpt_sovits_tts_for_videolingo_invalid_refer_mode(self, mock_load_key, mock_start_server,
                                                             temp_audio_file, sample_task_df):
        """Test invalid refer mode raises error."""
        mock_load_key_values = {
            "target_language": "en",
            "whisper.language": "en", 
            "gpt_sovits": {"character": "test_char", "refer_mode": 4},  # Invalid
            "whisper.detected_language": "en"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        with pytest.raises(ValueError, match="Invalid REFER_MODE"):
            gpt_sovits_tts_for_videolingo("Test invalid mode", temp_audio_file, 1, sample_task_df)
    
    def test_gpt_sovits_tts_for_videolingo_chinese_detection(self, tmp_path):
        """Test Chinese character detection in reference audio filename."""
        # This tests the language detection logic
        test_content = "测试中文字符"
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in test_content)
        assert has_chinese is True
        
        english_content = "English test content"
        has_chinese_english = any('\u4e00' <= char <= '\u9fff' for char in english_content)
        assert has_chinese_english is False


class TestGptSovitsTtsIntegration:
    """Integration tests for GPT-SoVITS TTS functionality."""
    
    def test_module_imports(self):
        """Test that all required modules are properly imported."""
        import core.tts_backend.gpt_sovits_tts as gpt_module
        
        required_functions = [
            'check_lang',
            'gpt_sovits_tts',
            'gpt_sovits_tts_for_videolingo', 
            'find_and_check_config_path',
            'start_gpt_sovits_server'
        ]
        
        for func_name in required_functions:
            assert hasattr(gpt_module, func_name)
            assert callable(getattr(gpt_module, func_name))
    
    @patch('core.tts_backend.gpt_sovits_tts.start_gpt_sovits_server')
    @patch('core.tts_backend.gpt_sovits_tts.load_key')
    @patch('core.tts_backend.gpt_sovits_tts.find_and_check_config_path')
    @patch('core.tts_backend.gpt_sovits_tts.gpt_sovits_tts')
    def test_realistic_workflow_integration(self, mock_gpt_tts, mock_find_config, mock_load_key,
                                          mock_start_server, tmp_path):
        """Test realistic GPT-SoVITS workflow integration."""
        # Setup realistic configuration
        mock_load_key_values = {
            "target_language": "zh",
            "whisper.language": "auto", 
            "gpt_sovits": {"character": "xiaomai", "refer_mode": 1},
            "whisper.detected_language": "zh"
        }
        mock_load_key.side_effect = lambda key: mock_load_key_values.get(key, mock_load_key_values.get(key.split('.')[0], {}))
        
        # Mock config and reference audio
        config_dir = tmp_path / "GPT_SoVITS" / "configs"
        config_dir.mkdir(parents=True)
        ref_audio = config_dir / "xiaomai_你好世界.wav"
        ref_audio.touch()
        
        mock_find_config.return_value = (tmp_path, config_dir / "xiaomai.yaml")
        mock_gpt_tts.return_value = True
        
        # Create realistic task DataFrame
        task_df = pd.DataFrame({
            'number': [1, 2],
            'origin': ['Hello world', 'How are you today?']
        })
        
        audio_file = str(tmp_path / "realistic_test.wav")
        realistic_text = "你好，欢迎使用VideoLingo！"
        
        gpt_sovits_tts_for_videolingo(realistic_text, audio_file, 1, task_df)
        
        # Verify server was started
        mock_start_server.assert_called_once()
        
        # Verify TTS was called with correct parameters
        mock_gpt_tts.assert_called_once()
        call_args = mock_gpt_tts.call_args
        
        assert call_args[0][0] == realistic_text  # text
        assert call_args[0][1] == "zh"  # target language
        assert call_args[0][2] == audio_file  # save path
        assert call_args[1]['prompt_lang'] == "zh"  # detected Chinese
        assert call_args[1]['prompt_text'] == "你好世界"  # extracted from filename
    
    def test_error_handling_robustness(self):
        """Test error handling robustness across the module."""
        # Test language validation errors
        with pytest.raises(ValueError):
            check_lang("unsupported", "en")
        
        with pytest.raises(ValueError):
            check_lang("en", "unsupported")
        
        # Test config path errors
        with patch('pathlib.Path.iterdir', return_value=[]):
            with pytest.raises(FileNotFoundError):
                find_and_check_config_path("test")
    
    @patch('core.tts_backend.gpt_sovits_tts.requests.post')
    def test_api_integration_realistic_payloads(self, mock_post, tmp_path):
        """Test API integration with realistic payloads."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"realistic_audio_data"
        mock_post.return_value = mock_response
        
        ref_audio_path = str(tmp_path / "reference.wav")
        save_path = str(tmp_path / "output.wav")
        
        with patch('core.tts_backend.gpt_sovits_tts.Path.cwd', return_value=tmp_path):
            result = gpt_sovits_tts(
                text="欢迎使用GPT-SoVITS进行语音合成！",
                text_lang="zh",
                save_path=save_path,
                ref_audio_path=ref_audio_path,
                prompt_lang="zh", 
                prompt_text="参考音频文本"
            )
            
            assert result is True
            
            # Verify realistic API payload
            call_args = mock_post.call_args[1]['json']
            expected_payload = {
                'text': "欢迎使用GPT-SoVITS进行语音合成！",
                'text_lang': "zh",
                'ref_audio_path': ref_audio_path,
                'prompt_lang': "zh",
                'prompt_text': "参考音频文本",
                'speed_factor': 1.0
            }
            
            assert call_args == expected_payload
