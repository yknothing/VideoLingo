"""
Comprehensive test suite for TTS backend - targeting 85% branch coverage
Tests TTS main function, text cleaning, retry logic, and all TTS methods
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from pydub import AudioSegment
from core.tts_backend.tts_main import tts_main, clean_text_for_tts


class TestCleanTextForTts:
    """Test text cleaning functionality for TTS"""

    def test_clean_text_removes_problematic_chars(self):
        """Test removal of problematic characters"""
        input_text = "Test text with & special ® chars ™ and © symbols"
        expected = "Test text with  special  chars  and  symbols"
        result = clean_text_for_tts(input_text)
        assert result == expected

    def test_clean_text_strips_whitespace(self):
        """Test whitespace stripping"""
        input_text = "  Test text with spaces  "
        expected = "Test text with spaces"
        result = clean_text_for_tts(input_text)
        assert result == expected

    def test_clean_text_empty_input(self):
        """Test empty input handling"""
        result = clean_text_for_tts("")
        assert result == ""

    def test_clean_text_only_problematic_chars(self):
        """Test text with only problematic characters"""
        input_text = "&®™©"
        result = clean_text_for_tts(input_text)
        assert result == ""


class TestTtsMainFunction:
    """Test main TTS function with different scenarios"""

    @pytest.fixture
    def mock_audio_segment(self):
        """Mock AudioSegment for silent audio generation"""
        with patch("core.tts_backend.tts_main.AudioSegment") as mock_segment:
            mock_silence = Mock()
            mock_segment.silent.return_value = mock_silence
            yield mock_segment, mock_silence

    @pytest.fixture
    def mock_dependencies(self):
        """Mock common dependencies"""
        with patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch("core.tts_backend.tts_main.load_key"), patch(
            "core.tts_backend.tts_main.ask_gpt"
        ), patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=1.5
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            yield

    def test_tts_main_empty_text_generates_silence(
        self, mock_dependencies, mock_audio_segment
    ):
        """Test that empty text generates silent audio"""
        mock_segment, mock_silence = mock_audio_segment

        tts_main("", "output.wav", 1, Mock())

        # Should create 100ms silent audio
        mock_segment.silent.assert_called_once_with(duration=100)
        mock_silence.export.assert_called_once_with("output.wav", format="wav")

    def test_tts_main_single_char_text_generates_silence(
        self, mock_dependencies, mock_audio_segment
    ):
        """Test that single character text generates silent audio"""
        mock_segment, mock_silence = mock_audio_segment

        tts_main("a", "output.wav", 1, Mock())

        # Should create 100ms silent audio
        mock_segment.silent.assert_called_once_with(duration=100)
        mock_silence.export.assert_called_once_with("output.wav", format="wav")

    def test_tts_main_punctuation_only_generates_silence(
        self, mock_dependencies, mock_audio_segment
    ):
        """Test that punctuation-only text generates silent audio"""
        mock_segment, mock_silence = mock_audio_segment

        tts_main("!@#$%", "output.wav", 1, Mock())

        # Should create 100ms silent audio
        mock_segment.silent.assert_called_once_with(duration=100)
        mock_silence.export.assert_called_once_with("output.wav", format="wav")

    def test_tts_main_file_exists_skips_generation(self, mock_dependencies):
        """Test that existing files are skipped"""
        with patch("core.tts_backend.tts_main.os.path.exists", return_value=True):
            with patch("core.tts_backend.tts_main.openai_tts") as mock_tts:
                tts_main("Test text", "existing_file.wav", 1, Mock())

                # Should not call TTS function
                mock_tts.assert_not_called()

    def test_tts_main_openai_tts_method(self, mock_dependencies):
        """Test OpenAI TTS method selection"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts:
            mock_load_key.return_value = "openai_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            mock_tts.assert_called_once_with("Test text", "output.wav")

    def test_tts_main_gpt_sovits_method(self, mock_dependencies):
        """Test GPT-SoVITS TTS method selection"""
        task_df = Mock()
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.gpt_sovits_tts_for_videolingo"
        ) as mock_tts:
            mock_load_key.return_value = "gpt_sovits"

            tts_main("Test text", "output.wav", 5, task_df)

            mock_tts.assert_called_once_with("Test text", "output.wav", 5, task_df)

    def test_tts_main_fish_tts_method(self, mock_dependencies):
        """Test Fish TTS method selection"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.fish_tts"
        ) as mock_tts:
            mock_load_key.return_value = "fish_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            mock_tts.assert_called_once_with("Test text", "output.wav")

    def test_tts_main_azure_tts_method(self, mock_dependencies):
        """Test Azure TTS method selection"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.azure_tts"
        ) as mock_tts:
            mock_load_key.return_value = "azure_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            mock_tts.assert_called_once_with("Test text", "output.wav")

    def test_tts_main_edge_tts_method(self, mock_dependencies):
        """Test Edge TTS method selection"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.edge_tts"
        ) as mock_tts:
            mock_load_key.return_value = "edge_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            mock_tts.assert_called_once_with("Test text", "output.wav")

    def test_tts_main_custom_tts_method(self, mock_dependencies):
        """Test Custom TTS method selection"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.custom_tts"
        ) as mock_tts:
            mock_load_key.return_value = "custom_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            mock_tts.assert_called_once_with("Test text", "output.wav")

    def test_tts_main_sf_cosyvoice2_method(self, mock_dependencies):
        """Test SiliconFlow CosyVoice2 TTS method selection"""
        task_df = Mock()
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.cosyvoice_tts_for_videolingo"
        ) as mock_tts:
            mock_load_key.return_value = "sf_cosyvoice2"

            tts_main("Test text", "output.wav", 3, task_df)

            mock_tts.assert_called_once_with("Test text", "output.wav", 3, task_df)

    def test_tts_main_f5tts_method(self, mock_dependencies):
        """Test F5-TTS method selection"""
        task_df = Mock()
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.f5_tts_for_videolingo"
        ) as mock_tts:
            mock_load_key.return_value = "f5tts"

            tts_main("Test text", "output.wav", 2, task_df)

            mock_tts.assert_called_once_with("Test text", "output.wav", 2, task_df)


class TestTtsRetryLogic:
    """Test retry logic and error handling in TTS"""

    def test_tts_main_retry_on_failure_with_gpt_correction(self, mock_file_operations):
        """Test retry logic with GPT text correction on final attempt"""
        mock_audio_segment = Mock()
        with patch("core.tts_backend.tts_main.AudioSegment") as mock_segment, patch(
            "core.tts_backend.tts_main.load_key"
        ) as mock_load_key, patch(
            "core.tts_backend.tts_main.ask_gpt"
        ) as mock_gpt, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=1.5
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "openai_tts"
            mock_gpt.return_value = {"text": "Corrected text"}

            # First two attempts fail, third succeeds
            mock_tts.side_effect = [Exception("Error 1"), Exception("Error 2"), None]

            tts_main("Original text", "output.wav", 1, Mock())

            # Should call GPT for text correction on final attempt
            mock_gpt.assert_called_once()
            # Should call TTS 3 times (2 failures + 1 success)
            assert mock_tts.call_count == 3
            # Final call should use corrected text
            final_call_args = mock_tts.call_args_list[-1]
            assert final_call_args[0][0] == "Corrected text"

    def test_tts_main_zero_duration_audio_retry(self, mock_file_operations):
        """Test retry when generated audio has zero duration"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration"
        ) as mock_duration, patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.os.remove"
        ) as mock_remove, patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "openai_tts"
            # First attempt generates zero duration, second succeeds
            mock_duration.side_effect = [0, 1.5]

            tts_main("Test text", "output.wav", 1, Mock())

            # Should try twice
            assert mock_tts.call_count == 2
            # Should remove the zero-duration file
            mock_remove.assert_called_once_with("output.wav")

    def test_tts_main_zero_duration_final_attempt_generates_silence(
        self, mock_file_operations
    ):
        """Test fallback to silence when all attempts produce zero duration"""
        mock_audio_segment = Mock()
        with patch("core.tts_backend.tts_main.AudioSegment") as mock_segment, patch(
            "core.tts_backend.tts_main.load_key"
        ) as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=0
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.os.remove"
        ), patch(
            "core.tts_backend.tts_main.ask_gpt"
        ) as mock_gpt, patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "openai_tts"
            mock_gpt.return_value = {"text": "Corrected text"}
            mock_segment.silent.return_value = mock_audio_segment

            tts_main("Test text", "output.wav", 1, Mock())

            # Should create silent audio as fallback
            mock_segment.silent.assert_called_once_with(duration=100)
            mock_audio_segment.export.assert_called_once_with(
                "output.wav", format="wav"
            )

    def test_tts_main_max_retries_exceeded_raises_exception(self, mock_file_operations):
        """Test that max retries raises exception"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts",
            side_effect=Exception("Persistent error"),
        ), patch("core.tts_backend.tts_main.os.path.exists", return_value=False), patch(
            "core.tts_backend.tts_main.ask_gpt"
        ) as mock_gpt, patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ):
            mock_load_key.return_value = "openai_tts"
            mock_gpt.return_value = {"text": "Corrected text"}

            with pytest.raises(
                Exception, match="Failed to generate audio after 3 attempts"
            ):
                tts_main("Test text", "output.wav", 1, Mock())


class TestTtsSpecificMethods:
    """Test TTS method-specific branches and scenarios"""

    def test_tts_main_siliconflow_fish_tts_method(self, mock_file_operations):
        """Test SiliconFlow Fish TTS method"""
        task_df = Mock()
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.siliconflow_fish_tts_for_videolingo"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=1.5
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "sf_fish_tts"

            tts_main("Test text", "output.wav", 4, task_df)

            mock_tts.assert_called_once_with("Test text", "output.wav", 4, task_df)

    def test_tts_main_successful_audio_generation_logs_metrics(
        self, mock_file_operations
    ):
        """Test that successful audio generation logs proper metrics"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=2.5
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ) as mock_log, patch(
            "core.tts_backend.tts_main.inc_counter"
        ) as mock_counter, patch(
            "core.tts_backend.tts_main.observe_histogram"
        ) as mock_histogram:
            mock_load_key.return_value = "openai_tts"

            tts_main("Test text", "output.wav", 1, Mock())

            # Should log success event with duration
            mock_log.assert_any_call(
                "info",
                "tts generate success",
                stage="tts",
                op="generate",
                attempt=0,
                duration_ms=2500,
            )

            # Should increment success counter
            mock_counter.assert_called_with("tts.success", 1, stage="tts")

            # Should observe duration histogram
            mock_histogram.assert_called_with("tts.audio_duration_s", 2.5, stage="tts")


@pytest.mark.coverage
class TestTtsEdgeCases:
    """Test edge cases for comprehensive branch coverage"""

    def test_tts_main_text_cleaning_integration(self, mock_file_operations):
        """Test integration of text cleaning with TTS generation"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=1.5
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "openai_tts"

            # Text with problematic characters
            input_text = "Test & text ® with © symbols"
            expected_cleaned = "Test  text  with  symbols"

            tts_main(input_text, "output.wav", 1, Mock())

            # Should call TTS with cleaned text
            mock_tts.assert_called_once_with(expected_cleaned, "output.wav")

    def test_tts_main_gpt_correction_error_handling(self, mock_file_operations):
        """Test handling of GPT correction errors"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts", side_effect=Exception("TTS Error")
        ), patch(
            "core.tts_backend.tts_main.ask_gpt", side_effect=Exception("GPT Error")
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ):
            mock_load_key.return_value = "openai_tts"

            with pytest.raises(
                Exception, match="Failed to generate audio after 3 attempts"
            ):
                tts_main("Test text", "output.wav", 1, Mock())

    def test_tts_main_observability_error_handling(self, mock_file_operations):
        """Test that observability errors don't break TTS flow"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration", return_value=1.5
        ), patch(
            "core.tts_backend.tts_main.os.path.exists", return_value=False
        ), patch(
            "core.tts_backend.tts_main.log_event", side_effect=Exception("Log error")
        ), patch(
            "core.tts_backend.tts_main.inc_counter",
            side_effect=Exception("Counter error"),
        ), patch(
            "core.tts_backend.tts_main.observe_histogram",
            side_effect=Exception("Histogram error"),
        ):
            mock_load_key.return_value = "openai_tts"

            # Should not raise exception despite observability errors
            tts_main("Test text", "output.wav", 1, Mock())

            # TTS should still be called
            mock_tts.assert_called_once()

    def test_tts_main_file_exists_check_on_retry(self, mock_file_operations):
        """Test file existence check behavior during retries"""
        with patch("core.tts_backend.tts_main.load_key") as mock_load_key, patch(
            "core.tts_backend.tts_main.openai_tts"
        ) as mock_tts, patch(
            "core.tts_backend.tts_main.get_audio_duration"
        ) as mock_duration, patch(
            "core.tts_backend.tts_main.os.path.exists"
        ) as mock_exists, patch(
            "core.tts_backend.tts_main.os.remove"
        ) as mock_remove, patch(
            "core.tts_backend.tts_main.log_event"
        ), patch(
            "core.tts_backend.tts_main.inc_counter"
        ), patch(
            "core.tts_backend.tts_main.observe_histogram"
        ):
            mock_load_key.return_value = "openai_tts"
            # File doesn't exist initially, then exists after first attempt
            mock_exists.side_effect = [False, True]
            # First attempt creates zero duration, file exists for removal
            mock_duration.side_effect = [0, 1.5]

            tts_main("Test text", "output.wav", 1, Mock())

            # Should call TTS twice (retry after zero duration)
            assert mock_tts.call_count == 2
            # Should remove zero-duration file
            mock_remove.assert_called_once()
