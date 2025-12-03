"""
Comprehensive test coverage for core.tts_backend.tts_main module.
Tests main TTS orchestration, text cleaning, retry logic, and provider routing.
Covers edge cases, error handling, and duration validation.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call, MagicMock
from pydub import AudioSegment

# Import the module under test
from core.tts_backend.tts_main import clean_text_for_tts, tts_main


class TestCleanTextForTTS:
    """Test the clean_text_for_tts function."""

    def test_clean_problematic_characters(self):
        """Test removal of problematic characters for TTS."""
        test_cases = [
            ("Hello & World!", "Hello  World!"),
            ("Company® Product™", "Company Product"),
            ("Copyright© Text", "Copyright Text"),
            ("Mix &®™© All", "Mix  All"),
            ("Normal text", "Normal text"),
            ("", ""),
            ("   spaces   ", "spaces"),
        ]

        for input_text, expected in test_cases:
            result = clean_text_for_tts(input_text)
            assert result == expected

    def test_clean_text_strips_whitespace(self):
        """Test that clean_text_for_tts strips leading/trailing whitespace."""
        assert clean_text_for_tts("  hello  ") == "hello"
        assert clean_text_for_tts("\t\nworld\t\n") == "world"


class TestTTSMain:
    """Comprehensive test suite for tts_main function."""

    @pytest.fixture
    def mock_audio_segment(self):
        """Mock AudioSegment for silent audio generation."""
        with patch("core.tts_backend.tts_main.AudioSegment") as mock_segment:
            mock_silent = Mock()
            mock_segment.silent.return_value = mock_silent
            yield mock_segment

    @pytest.fixture
    def mock_duration_check(self):
        """Mock get_audio_duration function."""
        with patch("core.tts_backend.tts_main.get_audio_duration") as mock_duration:
            mock_duration.return_value = 2.5  # Default to positive duration
            yield mock_duration

    @pytest.fixture
    def mock_load_key(self):
        """Mock config loading."""
        with patch("core.tts_backend.tts_main.load_key") as mock_key:
            mock_key.return_value = "openai_tts"  # Default TTS method
            yield mock_key

    @pytest.fixture
    def mock_ask_gpt(self):
        """Mock GPT text correction."""
        with patch("core.tts_backend.tts_main.ask_gpt") as mock_gpt:
            mock_gpt.return_value = {"text": "corrected text"}
            yield mock_gpt

    @pytest.fixture
    def mock_file_operations(self):
        """Mock file operations."""
        with patch("os.path.exists") as mock_exists, patch("os.remove") as mock_remove:
            mock_exists.return_value = False  # File doesn't exist by default
            yield {"exists": mock_exists, "remove": mock_remove}

    @pytest.fixture
    def mock_tts_providers(self):
        """Mock all TTS provider functions."""
        with patch("core.tts_backend.tts_main.openai_tts") as mock_openai, patch(
            "core.tts_backend.tts_main.gpt_sovits_tts_for_videolingo"
        ) as mock_sovits, patch(
            "core.tts_backend.tts_main.fish_tts"
        ) as mock_fish, patch(
            "core.tts_backend.tts_main.azure_tts"
        ) as mock_azure, patch(
            "core.tts_backend.tts_main.edge_tts"
        ) as mock_edge, patch(
            "core.tts_backend.tts_main.custom_tts"
        ) as mock_custom, patch(
            "core.tts_backend.tts_main.siliconflow_fish_tts_for_videolingo"
        ) as mock_sf_fish, patch(
            "core.tts_backend.tts_main.cosyvoice_tts_for_videolingo"
        ) as mock_cosyvoice, patch(
            "core.tts_backend.tts_main.f5_tts_for_videolingo"
        ) as mock_f5tts:
            yield {
                "openai": mock_openai,
                "sovits": mock_sovits,
                "fish": mock_fish,
                "azure": mock_azure,
                "edge": mock_edge,
                "custom": mock_custom,
                "sf_fish": mock_sf_fish,
                "cosyvoice": mock_cosyvoice,
                "f5tts": mock_f5tts,
            }

    def test_empty_text_creates_silent_audio(
        self, mock_audio_segment, mock_file_operations, mock_duration_check
    ):
        """Test that empty text creates silent audio."""
        with patch("builtins.print"):
            tts_main("", "output.wav", 1, Mock())

        # Should create silent audio
        mock_audio_segment.silent.assert_called_once_with(duration=100)
        mock_audio_segment.silent.return_value.export.assert_called_once_with(
            "output.wav", format="wav"
        )

    def test_single_character_creates_silent_audio(
        self, mock_audio_segment, mock_file_operations, mock_duration_check
    ):
        """Test that single character text creates silent audio."""
        with patch("builtins.print"):
            tts_main(".", "output.wav", 1, Mock())

        # Should create silent audio
        mock_audio_segment.silent.assert_called_once_with(duration=100)
        mock_audio_segment.silent.return_value.export.assert_called_once_with(
            "output.wav", format="wav"
        )

    def test_whitespace_only_creates_silent_audio(
        self, mock_audio_segment, mock_file_operations, mock_duration_check
    ):
        """Test that whitespace-only text creates silent audio."""
        with patch("builtins.print"):
            tts_main("   !@#$%   ", "output.wav", 1, Mock())

        # Should create silent audio
        mock_audio_segment.silent.assert_called_once_with(duration=100)
        mock_audio_segment.silent.return_value.export.assert_called_once_with(
            "output.wav", format="wav"
        )

    def test_skip_if_file_exists(
        self, mock_file_operations, mock_tts_providers, mock_load_key
    ):
        """Test that function returns early if output file exists."""
        mock_file_operations["exists"].return_value = True

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should not call any TTS provider
        for provider in mock_tts_providers.values():
            provider.assert_not_called()

    def test_openai_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful OpenAI TTS generation."""
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        mock_tts_providers["openai"].assert_called_once_with(
            "Hello world", "output.wav"
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_azure_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful Azure TTS generation."""
        mock_load_key.return_value = "azure_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        mock_tts_providers["azure"].assert_called_once_with("Hello world", "output.wav")
        mock_duration_check.assert_called_once_with("output.wav")

    def test_edge_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful Edge TTS generation."""
        mock_load_key.return_value = "edge_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        mock_tts_providers["edge"].assert_called_once_with("Hello world", "output.wav")
        mock_duration_check.assert_called_once_with("output.wav")

    def test_fish_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful Fish TTS generation."""
        mock_load_key.return_value = "fish_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        mock_tts_providers["fish"].assert_called_once_with("Hello world", "output.wav")
        mock_duration_check.assert_called_once_with("output.wav")

    def test_custom_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful Custom TTS generation."""
        mock_load_key.return_value = "custom_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        mock_tts_providers["custom"].assert_called_once_with(
            "Hello world", "output.wav"
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_gpt_sovits_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful GPT-SoVITS generation."""
        mock_load_key.return_value = "gpt_sovits"
        task_df = Mock()

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, task_df)

        mock_tts_providers["sovits"].assert_called_once_with(
            "Hello world", "output.wav", 1, task_df
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_sf_fish_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful SiliconFlow Fish TTS generation."""
        mock_load_key.return_value = "sf_fish_tts"
        task_df = Mock()

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, task_df)

        mock_tts_providers["sf_fish"].assert_called_once_with(
            "Hello world", "output.wav", 1, task_df
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_cosyvoice_tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful CosyVoice TTS generation."""
        mock_load_key.return_value = "sf_cosyvoice2"
        task_df = Mock()

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, task_df)

        mock_tts_providers["cosyvoice"].assert_called_once_with(
            "Hello world", "output.wav", 1, task_df
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_f5tts_success(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test successful F5-TTS generation."""
        mock_load_key.return_value = "f5tts"
        task_df = Mock()

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, task_df)

        mock_tts_providers["f5tts"].assert_called_once_with(
            "Hello world", "output.wav", 1, task_df
        )
        mock_duration_check.assert_called_once_with("output.wav")

    def test_retry_on_zero_duration(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
        mock_audio_segment,
    ):
        """Test retry mechanism when generated audio has zero duration."""
        # First attempt returns zero duration, second attempt succeeds
        mock_duration_check.side_effect = [0, 2.5]
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should be called twice due to retry
        assert mock_tts_providers["openai"].call_count == 2
        assert mock_duration_check.call_count == 2
        mock_file_operations["remove"].assert_called_once_with("output.wav")

    def test_fallback_to_silent_audio_after_max_retries(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
        mock_audio_segment,
    ):
        """Test fallback to silent audio after max retries with zero duration."""
        # Always return zero duration
        mock_duration_check.return_value = 0
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should be called 3 times (max retries)
        assert mock_tts_providers["openai"].call_count == 3
        # Should create fallback silent audio
        mock_audio_segment.silent.assert_called_once_with(duration=100)
        mock_audio_segment.silent.return_value.export.assert_called_once_with(
            "output.wav", format="wav"
        )

    def test_retry_with_gpt_correction_on_final_attempt(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
        mock_ask_gpt,
    ):
        """Test GPT text correction on final retry attempt."""
        # Fail first two attempts, succeed on third with corrected text
        mock_tts_providers["openai"].side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            None,
        ]
        mock_load_key.return_value = "openai_tts"
        mock_ask_gpt.return_value = {"text": "corrected text"}

        with patch("builtins.print"), patch(
            "core.tts_backend.tts_main.get_correct_text_prompt"
        ) as mock_prompt:
            mock_prompt.return_value = "correct this text"

            tts_main("problematic text", "output.wav", 1, Mock())

        # Should call ask_gpt for text correction
        mock_ask_gpt.assert_called_once_with(
            "correct this text", resp_type="json", log_title="tts_correct_text"
        )

        # Final call should use corrected text
        final_call = mock_tts_providers["openai"].call_args_list[-1]
        assert final_call[0][0] == "corrected text"

    def test_exception_propagation_after_max_retries(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that exceptions are properly propagated after max retries."""
        mock_tts_providers["openai"].side_effect = Exception("TTS Error")
        mock_load_key.return_value = "openai_tts"

        with pytest.raises(
            Exception, match="Failed to generate audio after 3 attempts: TTS Error"
        ), patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

    def test_text_cleaning_before_processing(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that text is cleaned before TTS processing."""
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print"):
            tts_main("Hello & World®!", "output.wav", 1, Mock())

        # Should call TTS with cleaned text
        mock_tts_providers["openai"].assert_called_once_with(
            "Hello  World!", "output.wav"
        )

    def test_print_generation_message(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that generation message is printed."""
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print") as mock_print:
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should print generation message
        mock_print.assert_any_call("Generating <Hello world...>")

    def test_unknown_tts_method_no_generation(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test behavior with unknown TTS method."""
        mock_load_key.return_value = "unknown_method"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        # No TTS provider should be called
        for provider in mock_tts_providers.values():
            provider.assert_not_called()

        # Should still check duration
        mock_duration_check.assert_called_once_with("output.wav")

    def test_retry_count_tracking(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that retry attempts are properly tracked."""
        # Fail first attempt, succeed on second
        mock_tts_providers["openai"].side_effect = [Exception("Error"), None]
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print") as mock_print:
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should print retry message
        mock_print.assert_any_call("Attempt 1 failed, retrying...")
        assert mock_tts_providers["openai"].call_count == 2

    def test_file_cleanup_on_zero_duration(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that files are cleaned up when duration is zero."""
        # Return zero duration, then success
        mock_duration_check.side_effect = [0, 2.5]
        mock_file_operations["exists"].side_effect = lambda path: path == "output.wav"
        mock_load_key.return_value = "openai_tts"

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 1, Mock())

        # Should remove file with zero duration
        mock_file_operations["remove"].assert_called_once_with("output.wav")

    def test_task_df_parameter_passed_correctly(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test that task_df parameter is passed correctly to providers that need it."""
        mock_load_key.return_value = "gpt_sovits"
        task_df = Mock()

        with patch("builtins.print"):
            tts_main("Hello world", "output.wav", 5, task_df)

        # Should pass both number and task_df parameters
        mock_tts_providers["sovits"].assert_called_once_with(
            "Hello world", "output.wav", 5, task_df
        )

    def test_integration_text_processing_pipeline(
        self,
        mock_tts_providers,
        mock_file_operations,
        mock_load_key,
        mock_duration_check,
    ):
        """Test complete text processing pipeline from input to TTS call."""
        mock_load_key.return_value = "openai_tts"
        input_text = "  Hello & World®!  "
        expected_cleaned = "Hello  World!"

        with patch("builtins.print"):
            tts_main(input_text, "output.wav", 1, Mock())

        # Verify text was properly cleaned and passed to TTS
        mock_tts_providers["openai"].assert_called_once_with(
            expected_cleaned, "output.wav"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
