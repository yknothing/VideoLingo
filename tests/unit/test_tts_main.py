import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
from pydub import AudioSegment
from pathlib import Path

from core.tts_backend.tts_main import tts_main, clean_text_for_tts


class TestCleanTextForTts:
    """Test text cleaning functionality for TTS processing."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning removes problematic characters."""
        test_cases = [
            ("Hello & World", "Hello  World"),
            ("Company® Product™", "Company Product"),
            ("Copyright© Text", "Copyright Text"),
            ("Mixed &®™© chars", "Mixed  chars"),
            ("  Spaced text  ", "Spaced text"),
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            result = clean_text_for_tts(input_text)
            assert result == expected
    
    def test_clean_text_preserves_normal_chars(self):
        """Test that normal characters are preserved."""
        text = "Hello World 123 !@#$%^*()_+-=[]{}|;':\",./<>?"
        result = clean_text_for_tts(text)
        assert "Hello World 123 !@#$%^*()_+-=[]{}|;':\",./<>?" in result
    
    def test_clean_text_empty_input(self):
        """Test cleaning empty or whitespace-only input."""
        assert clean_text_for_tts("") == ""
        assert clean_text_for_tts("   ") == ""
        assert clean_text_for_tts("\t\n") == ""


class TestTtsMain:
    """Test main TTS orchestration functionality."""
    
    @pytest.fixture
    def sample_task_df(self):
        """Create sample task DataFrame for testing."""
        return pd.DataFrame({
            'number': [1, 2, 3],
            'origin': ['Original text 1', 'Original text 2', 'Original text 3'],
            'translated': ['Translated text 1', 'Translated text 2', 'Translated text 3']
        })
    
    @pytest.fixture
    def temp_audio_file(self, tmp_path):
        """Create temporary audio file path."""
        return str(tmp_path / "test_audio.wav")
    
    def test_empty_text_creates_silent_audio(self, temp_audio_file, sample_task_df):
        """Test that empty or single character text creates silent audio."""
        with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
            mock_silence = Mock()
            mock_audioseg.silent.return_value = mock_silence
            
            # Test empty text
            tts_main("", temp_audio_file, 1, sample_task_df)
            
            mock_audioseg.silent.assert_called_with(duration=100)
            mock_silence.export.assert_called_with(temp_audio_file, format="wav")
    
    def test_single_char_text_creates_silent_audio(self, temp_audio_file, sample_task_df):
        """Test that single character text creates silent audio."""
        with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
            mock_silence = Mock()
            mock_audioseg.silent.return_value = mock_silence
            
            # Test single character
            tts_main("a", temp_audio_file, 1, sample_task_df)
            
            mock_audioseg.silent.assert_called_with(duration=100)
            mock_silence.export.assert_called_with(temp_audio_file, format="wav")
    
    def test_punctuation_only_text_creates_silent_audio(self, temp_audio_file, sample_task_df):
        """Test that punctuation-only text creates silent audio."""
        with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
            mock_silence = Mock()
            mock_audioseg.silent.return_value = mock_silence
            
            # Test punctuation only
            tts_main("...!?", temp_audio_file, 1, sample_task_df)
            
            mock_audioseg.silent.assert_called_with(duration=100)
            mock_silence.export.assert_called_with(temp_audio_file, format="wav")
    
    @patch('core.tts_backend.tts_main.os.path.exists')
    def test_skip_if_file_exists(self, mock_exists, temp_audio_file, sample_task_df):
        """Test that TTS is skipped if audio file already exists."""
        mock_exists.return_value = True
        
        with patch('core.tts_backend.tts_main.load_key') as mock_load_key:
            mock_load_key.return_value = "openai_tts"
            
            # Should return early without processing
            tts_main("Hello world", temp_audio_file, 1, sample_task_df)
            
            # Verify no TTS method was called
            mock_exists.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_openai_tts_success(self, mock_duration, mock_openai, mock_load_key, mock_exists, 
                               temp_audio_file, sample_task_df):
        """Test successful OpenAI TTS generation."""
        mock_load_key.return_value = "openai_tts"
        mock_duration.return_value = 2.5  # Valid duration
        
        tts_main("Hello world", temp_audio_file, 1, sample_task_df)
        
        mock_openai.assert_called_once_with("Hello world", temp_audio_file)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.azure_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_azure_tts_success(self, mock_duration, mock_azure, mock_load_key, mock_exists,
                              temp_audio_file, sample_task_df):
        """Test successful Azure TTS generation."""
        mock_load_key.return_value = "azure_tts"
        mock_duration.return_value = 1.8
        
        tts_main("Test text", temp_audio_file, 1, sample_task_df)
        
        mock_azure.assert_called_once_with("Test text", temp_audio_file)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.edge_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_edge_tts_success(self, mock_duration, mock_edge, mock_load_key, mock_exists,
                             temp_audio_file, sample_task_df):
        """Test successful Edge TTS generation."""
        mock_load_key.return_value = "edge_tts"
        mock_duration.return_value = 3.2
        
        tts_main("Edge TTS test", temp_audio_file, 1, sample_task_df)
        
        mock_edge.assert_called_once_with("Edge TTS test", temp_audio_file)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.fish_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_fish_tts_success(self, mock_duration, mock_fish, mock_load_key, mock_exists,
                             temp_audio_file, sample_task_df):
        """Test successful Fish TTS generation."""
        mock_load_key.return_value = "fish_tts"
        mock_duration.return_value = 2.1
        
        tts_main("Fish TTS test", temp_audio_file, 1, sample_task_df)
        
        mock_fish.assert_called_once_with("Fish TTS test", temp_audio_file)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.custom_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_custom_tts_success(self, mock_duration, mock_custom, mock_load_key, mock_exists,
                               temp_audio_file, sample_task_df):
        """Test successful Custom TTS generation."""
        mock_load_key.return_value = "custom_tts"
        mock_duration.return_value = 1.5
        
        tts_main("Custom TTS test", temp_audio_file, 1, sample_task_df)
        
        mock_custom.assert_called_once_with("Custom TTS test", temp_audio_file)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.gpt_sovits_tts_for_videolingo')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_gpt_sovits_success(self, mock_duration, mock_sovits, mock_load_key, mock_exists,
                               temp_audio_file, sample_task_df):
        """Test successful GPT-SoVITS TTS generation."""
        mock_load_key.return_value = "gpt_sovits"
        mock_duration.return_value = 4.2
        
        tts_main("GPT-SoVITS test", temp_audio_file, 2, sample_task_df)
        
        mock_sovits.assert_called_once_with("GPT-SoVITS test", temp_audio_file, 2, sample_task_df)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.siliconflow_fish_tts_for_videolingo')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_sf_fish_tts_success(self, mock_duration, mock_sf_fish, mock_load_key, mock_exists,
                                temp_audio_file, sample_task_df):
        """Test successful SiliconFlow Fish TTS generation."""
        mock_load_key.return_value = "sf_fish_tts"
        mock_duration.return_value = 2.8
        
        tts_main("SiliconFlow Fish test", temp_audio_file, 3, sample_task_df)
        
        mock_sf_fish.assert_called_once_with("SiliconFlow Fish test", temp_audio_file, 3, sample_task_df)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.cosyvoice_tts_for_videolingo')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_cosyvoice_success(self, mock_duration, mock_cosy, mock_load_key, mock_exists,
                              temp_audio_file, sample_task_df):
        """Test successful CosyVoice TTS generation."""
        mock_load_key.return_value = "sf_cosyvoice2"
        mock_duration.return_value = 3.5
        
        tts_main("CosyVoice test", temp_audio_file, 1, sample_task_df)
        
        mock_cosy.assert_called_once_with("CosyVoice test", temp_audio_file, 1, sample_task_df)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.f5_tts_for_videolingo')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_f5_tts_success(self, mock_duration, mock_f5, mock_load_key, mock_exists,
                           temp_audio_file, sample_task_df):
        """Test successful F5-TTS generation."""
        mock_load_key.return_value = "f5tts"
        mock_duration.return_value = 2.3
        
        tts_main("F5-TTS test", temp_audio_file, 2, sample_task_df)
        
        mock_f5.assert_called_once_with("F5-TTS test", temp_audio_file, 2, sample_task_df)
        mock_duration.assert_called_once_with(temp_audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    @patch('core.tts_backend.tts_main.os.remove')
    @patch('core.tts_backend.tts_main.AudioSegment')
    def test_zero_duration_retry_logic(self, mock_audioseg, mock_remove, mock_duration, 
                                      mock_openai, mock_load_key, mock_exists,
                                      temp_audio_file, sample_task_df):
        """Test retry logic when generated audio has zero duration."""
        mock_load_key.return_value = "openai_tts"
        mock_duration.side_effect = [0, 0, 2.5]  # First two attempts fail, third succeeds
        mock_silence = Mock()
        mock_audioseg.silent.return_value = mock_silence
        
        tts_main("Test retry", temp_audio_file, 1, sample_task_df)
        
        # Should call TTS 3 times
        assert mock_openai.call_count == 3
        assert mock_duration.call_count == 3
        assert mock_remove.call_count == 2  # Remove failed audio files
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    @patch('core.tts_backend.tts_main.os.remove')
    @patch('core.tts_backend.tts_main.AudioSegment')
    def test_max_retries_creates_silence(self, mock_audioseg, mock_remove, mock_duration,
                                        mock_openai, mock_load_key, mock_exists,
                                        temp_audio_file, sample_task_df):
        """Test that max retries creates silent audio."""
        mock_load_key.return_value = "openai_tts"
        mock_duration.return_value = 0  # Always return 0 duration
        mock_silence = Mock()
        mock_audioseg.silent.return_value = mock_silence
        
        tts_main("Test max retry", temp_audio_file, 1, sample_task_df)
        
        # Should try 3 times then create silence
        assert mock_openai.call_count == 3
        assert mock_duration.call_count == 3
        mock_audioseg.silent.assert_called_with(duration=100)
        mock_silence.export.assert_called_with(temp_audio_file, format="wav")
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.ask_gpt')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_text_correction_on_final_retry(self, mock_duration, mock_ask_gpt, mock_openai,
                                           mock_load_key, mock_exists, temp_audio_file, sample_task_df):
        """Test that text correction is attempted on final retry."""
        mock_load_key.return_value = "openai_tts"
        mock_openai.side_effect = [Exception("Error 1"), Exception("Error 2"), None]
        mock_ask_gpt.return_value = {"text": "Corrected text"}
        mock_duration.return_value = 1.5
        
        tts_main("Problematic text", temp_audio_file, 1, sample_task_df)
        
        # Should call ask_gpt for text correction on final attempt
        mock_ask_gpt.assert_called_once()
        # Final call should use corrected text
        final_call = mock_openai.call_args_list[-1]
        assert final_call[0][0] == "Corrected text"
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    def test_exception_after_max_retries(self, mock_openai, mock_load_key, mock_exists,
                                        temp_audio_file, sample_task_df):
        """Test that exception is raised after max retries."""
        mock_load_key.return_value = "openai_tts"
        mock_openai.side_effect = Exception("TTS Error")
        
        with pytest.raises(Exception, match="Failed to generate audio after 3 attempts"):
            tts_main("Test error", temp_audio_file, 1, sample_task_df)
        
        assert mock_openai.call_count == 3
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_text_cleaning_before_tts(self, mock_duration, mock_openai, mock_load_key, mock_exists,
                                     temp_audio_file, sample_task_df):
        """Test that text is cleaned before TTS processing."""
        mock_load_key.return_value = "openai_tts"
        mock_duration.return_value = 1.5
        
        dirty_text = "Hello & World®™©"
        tts_main(dirty_text, temp_audio_file, 1, sample_task_df)
        
        # Verify cleaned text is passed to TTS
        mock_openai.assert_called_once_with("Hello  World", temp_audio_file)
    
    def test_unsupported_tts_method(self, temp_audio_file, sample_task_df):
        """Test behavior with unsupported TTS method."""
        with patch('core.tts_backend.tts_main.os.path.exists', return_value=False):
            with patch('core.tts_backend.tts_main.load_key') as mock_load_key:
                mock_load_key.return_value = "unsupported_method"
                
                # Should not call any TTS method and process should continue
                # This tests that the elif chain handles unknown methods gracefully
                with patch('core.tts_backend.tts_main.get_audio_duration') as mock_duration:
                    mock_duration.return_value = 0
                    
                    with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
                        mock_silence = Mock()
                        mock_audioseg.silent.return_value = mock_silence
                        
                        tts_main("Test unsupported", temp_audio_file, 1, sample_task_df)
                        
                        # Should create silence after failing to generate audio
                        mock_audioseg.silent.assert_called()
    
    def test_text_with_only_special_chars(self, temp_audio_file, sample_task_df):
        """Test text containing only special characters that get cleaned away."""
        with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
            mock_silence = Mock()
            mock_audioseg.silent.return_value = mock_silence
            
            tts_main("&®™©", temp_audio_file, 1, sample_task_df)
            
            mock_audioseg.silent.assert_called_with(duration=100)
            mock_silence.export.assert_called_with(temp_audio_file, format="wav")


class TestTtsMainIntegration:
    """Integration tests for TTS main functionality."""
    
    @pytest.fixture
    def integration_task_df(self):
        """Create realistic task DataFrame for integration tests."""
        return pd.DataFrame({
            'number': [1, 2, 3, 4, 5],
            'origin': [
                'Hello, how are you today?',
                'The weather is beautiful outside.',
                'Let me tell you a story.',
                'This is a test of the emergency broadcast system.',
                'Thank you for watching!'
            ],
            'translated': [
                'Hola, ¿cómo estás hoy?',
                'El clima está hermoso afuera.',
                'Déjame contarte una historia.',
                'Esta es una prueba del sistema de transmisión de emergencia.',
                '¡Gracias por mirar!'
            ]
        })
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    @patch('core.tts_backend.tts_main.openai_tts')
    @patch('core.tts_backend.tts_main.get_audio_duration')
    def test_realistic_workflow(self, mock_duration, mock_openai, mock_load_key, mock_exists,
                               integration_task_df, tmp_path):
        """Test realistic TTS generation workflow."""
        mock_load_key.return_value = "openai_tts"
        mock_duration.return_value = 2.5
        
        audio_file = str(tmp_path / "realistic_test.wav")
        
        tts_main("This is a realistic test sentence.", audio_file, 1, integration_task_df)
        
        mock_openai.assert_called_once_with("This is a realistic test sentence.", audio_file)
        mock_duration.assert_called_once_with(audio_file)
    
    @patch('core.tts_backend.tts_main.os.path.exists', return_value=False)
    @patch('core.tts_backend.tts_main.load_key')
    def test_multiple_tts_methods_coverage(self, mock_load_key, tmp_path, integration_task_df):
        """Test that all TTS methods can be configured and called."""
        methods_to_test = [
            ("openai_tts", "core.tts_backend.tts_main.openai_tts"),
            ("azure_tts", "core.tts_backend.tts_main.azure_tts"),
            ("edge_tts", "core.tts_backend.tts_main.edge_tts"),
            ("fish_tts", "core.tts_backend.tts_main.fish_tts"),
            ("custom_tts", "core.tts_backend.tts_main.custom_tts"),
        ]
        
        for method_name, method_path in methods_to_test:
            with patch(method_path) as mock_method:
                with patch('core.tts_backend.tts_main.get_audio_duration', return_value=1.5):
                    mock_load_key.return_value = method_name
                    audio_file = str(tmp_path / f"{method_name}_test.wav")
                    
                    tts_main(f"Test for {method_name}", audio_file, 1, integration_task_df)
                    
                    mock_method.assert_called_once()
    
    def test_error_handling_robustness(self, tmp_path, integration_task_df):
        """Test error handling robustness across different scenarios."""
        test_cases = [
            ("", "empty_text.wav"),  # Empty text
            ("a", "single_char.wav"),  # Single character
            ("!@#$%", "special_only.wav"),  # Special characters only
            ("   ", "whitespace_only.wav"),  # Whitespace only
        ]
        
        for text, filename in test_cases:
            audio_file = str(tmp_path / filename)
            
            with patch('core.tts_backend.tts_main.AudioSegment') as mock_audioseg:
                mock_silence = Mock()
                mock_audioseg.silent.return_value = mock_silence
                
                # Should not raise exception
                tts_main(text, audio_file, 1, integration_task_df)
                
                # Should create silent audio
                mock_audioseg.silent.assert_called_with(duration=100)
                mock_silence.export.assert_called_with(audio_file, format="wav")
