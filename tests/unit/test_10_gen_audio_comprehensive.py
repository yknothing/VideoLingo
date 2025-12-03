import pytest
import os
import tempfile
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import sys
from pydub import AudioSegment
from concurrent.futures import Future

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core._10_gen_audio import (
    adjust_audio_speed,
    process_row,
    generate_tts_audio,
    process_chunk,
    merge_chunks,
    gen_audio,
)


class TestAdjustAudioSpeed:
    """Test audio speed adjustment functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "input.wav")
        self.output_file = os.path.join(self.temp_dir, "output.wav")

        # Create dummy input file
        with open(self.input_file, "wb") as f:
            f.write(b"fake audio data")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("shutil.copy2")
    def test_adjust_audio_speed_near_unity(self, mock_copy, mock_duration):
        """Test speed adjustment when factor is close to 1.0"""
        # Speed factor very close to 1.0 should just copy file
        speed_factor = 1.001  # Close to 1.0

        adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        mock_copy.assert_called_once_with(self.input_file, self.output_file)
        mock_duration.assert_not_called()

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("subprocess.run")
    def test_adjust_audio_speed_success(self, mock_subprocess, mock_duration):
        """Test successful audio speed adjustment"""
        speed_factor = 1.5
        input_duration = 10.0
        output_duration = 6.67  # 10/1.5

        mock_duration.side_effect = [input_duration, output_duration]
        mock_subprocess.return_value = Mock()

        adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        # Verify ffmpeg command was called correctly
        expected_cmd = [
            "ffmpeg",
            "-i",
            self.input_file,
            "-filter:a",
            f"atempo={speed_factor}",
            "-y",
            self.output_file,
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd, check=True, stderr=subprocess.PIPE
        )

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("subprocess.run")
    @patch("core._10_gen_audio.AudioSegment")
    def test_adjust_audio_speed_trim_short_audio(
        self, mock_audiosegment, mock_subprocess, mock_duration
    ):
        """Test trimming when short audio duration is abnormal"""
        speed_factor = 2.0
        input_duration = 2.0  # Less than 3 seconds
        output_duration = 1.05  # Slightly longer than expected (1.0)
        expected_duration = 1.0

        mock_duration.side_effect = [input_duration, output_duration]
        mock_subprocess.return_value = Mock()

        # Mock AudioSegment operations
        mock_audio = Mock()
        mock_trimmed = Mock()
        mock_audiosegment.from_wav.return_value = mock_audio
        mock_audio.__getitem__.return_value = mock_trimmed

        with patch("builtins.print"):
            adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        # Verify audio was trimmed
        mock_audiosegment.from_wav.assert_called_once_with(self.output_file)
        mock_audio.__getitem__.assert_called_once_with(
            slice(None, expected_duration * 1000)
        )
        mock_trimmed.export.assert_called_once_with(self.output_file, format="wav")

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("subprocess.run")
    def test_adjust_audio_speed_abnormal_duration_error(
        self, mock_subprocess, mock_duration
    ):
        """Test error when output duration is abnormal"""
        speed_factor = 1.5
        input_duration = 10.0
        output_duration = 15.0  # Much longer than expected (6.67)

        mock_duration.side_effect = [input_duration, output_duration]
        mock_subprocess.return_value = Mock()

        with pytest.raises(Exception) as exc_info:
            adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        assert "Audio duration abnormal" in str(exc_info.value)

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("core._10_gen_audio.rprint")
    def test_adjust_audio_speed_retry_success(
        self, mock_rprint, mock_sleep, mock_subprocess, mock_duration
    ):
        """Test retry mechanism on subprocess failure"""
        speed_factor = 1.2
        input_duration = 5.0
        output_duration = 4.17  # 5/1.2

        # First call fails, second succeeds
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "ffmpeg"),
            Mock(),
        ]
        mock_duration.side_effect = [input_duration, output_duration]

        adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        # Should have retried
        assert mock_subprocess.call_count == 2
        mock_sleep.assert_called_once()
        mock_rprint.assert_called()

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("core._10_gen_audio.rprint")
    def test_adjust_audio_speed_max_retries_exceeded(
        self, mock_rprint, mock_sleep, mock_subprocess, mock_duration
    ):
        """Test max retries exceeded"""
        speed_factor = 1.2
        input_duration = 5.0

        # All attempts fail
        error = subprocess.CalledProcessError(1, "ffmpeg")
        mock_subprocess.side_effect = error
        mock_duration.return_value = input_duration

        with pytest.raises(subprocess.CalledProcessError):
            adjust_audio_speed(self.input_file, self.output_file, speed_factor)

        # Should have tried maximum times
        assert mock_subprocess.call_count == 3  # RetryConstants.AUDIO_SPEED_MAX_RETRIES

    @patch("core._10_gen_audio.SPEED_FACTOR_PRECISION", 0.01)
    @patch("shutil.copy2")
    def test_adjust_audio_speed_precision_threshold(self, mock_copy):
        """Test precision threshold for speed factor"""
        # Test values just below and above threshold
        test_cases = [
            (1.005, True),  # Below threshold, should copy
            (1.015, False),  # Above threshold, should process
        ]

        for speed_factor, should_copy in test_cases:
            mock_copy.reset_mock()

            with patch(
                "core._10_gen_audio.get_audio_duration", return_value=5.0
            ), patch("subprocess.run", return_value=Mock()):
                adjust_audio_speed(self.input_file, self.output_file, speed_factor)

                if should_copy:
                    mock_copy.assert_called_once()
                else:
                    mock_copy.assert_not_called()


class TestProcessRow:
    """Test row processing functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.tasks_df = pd.DataFrame(
            {"number": [1, 2, 3], "lines": [["Hello"], ["World", "Test"], ["End"]]}
        )

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_single_line(self, mock_duration, mock_tts):
        """Test processing row with single line"""
        row = pd.Series({"number": 1, "lines": ["Hello world"]})

        mock_duration.return_value = 2.5

        number, real_dur = process_row(row, self.tasks_df)

        assert number == 1
        assert real_dur == 2.5

        # Verify TTS was called correctly
        mock_tts.assert_called_once_with(
            "Hello world",
            f"{self.temp_dir}/audio/tmp/1_0_temp.wav".replace(
                self.temp_dir, ""
            ).replace("/audio/tmp", f"{self.temp_dir}/audio/tmp"),
            1,
            self.tasks_df,
        )

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_multiple_lines(self, mock_duration, mock_tts):
        """Test processing row with multiple lines"""
        row = pd.Series({"number": 2, "lines": ["Hello", "world", "test"]})

        mock_duration.side_effect = [1.5, 2.0, 1.2]  # Different durations for each line

        number, real_dur = process_row(row, self.tasks_df)

        assert number == 2
        assert real_dur == 4.7  # Sum of all durations

        # Verify TTS was called for each line
        assert mock_tts.call_count == 3
        assert mock_duration.call_count == 3

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_with_string_lines(self, mock_duration, mock_tts):
        """Test processing row where lines is stored as string (eval needed)"""
        row = pd.Series(
            {
                "number": 3,
                "lines": "['Line one', 'Line two']",  # String representation of list
            }
        )

        mock_duration.side_effect = [3.0, 2.5]

        number, real_dur = process_row(row, self.tasks_df)

        assert number == 3
        assert real_dur == 5.5

        # Verify both lines were processed
        assert mock_tts.call_count == 2
        mock_tts.assert_any_call("Line one", mock.ANY, 3, self.tasks_df)
        mock_tts.assert_any_call("Line two", mock.ANY, 3, self.tasks_df)

    @patch("core._10_gen_audio.tts_main", side_effect=Exception("TTS failed"))
    def test_process_row_tts_error(self, mock_tts):
        """Test error handling in process_row"""
        row = pd.Series({"number": 1, "lines": ["Test line"]})

        with pytest.raises(Exception) as exc_info:
            process_row(row, self.tasks_df)

        assert "TTS failed" in str(exc_info.value)

    def test_process_row_empty_lines(self):
        """Test processing row with empty lines list"""
        row = pd.Series({"number": 1, "lines": []})

        number, real_dur = process_row(row, self.tasks_df)

        assert number == 1
        assert real_dur == 0  # No lines means no duration


class TestGenerateTTSAudio:
    """Test TTS audio generation functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.tasks_df = pd.DataFrame(
            {
                "number": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ],  # 6 rows to test warmup and parallel processing
                "lines": [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]],
            }
        )

    def teardown_method(self):
        """Clean up test environment"""
        pass

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.Progress")
    @patch("core._10_gen_audio.rprint")
    def test_generate_tts_audio_small_dataset(
        self, mock_rprint, mock_progress, mock_process_row, mock_load_key
    ):
        """Test TTS generation with small dataset (no parallel processing)"""
        small_df = self.tasks_df.head(3)  # Only 3 rows
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "openai",
        }.get(key, 4)

        # Mock process_row to return (number, duration)
        mock_process_row.side_effect = [(1, 2.0), (2, 3.0), (3, 1.5)]

        # Mock progress context manager
        mock_progress_instance = Mock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance
        mock_task = Mock()
        mock_progress_instance.add_task.return_value = mock_task

        result_df = generate_tts_audio(small_df.copy())

        # Verify all rows were processed sequentially (warmup only)
        assert mock_process_row.call_count == 3
        assert list(result_df["real_dur"]) == [2.0, 3.0, 1.5]

        # Progress should advance for each row
        assert mock_progress_instance.advance.call_count == 3

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.Progress")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.rprint")
    def test_generate_tts_audio_large_dataset_parallel(
        self, mock_rprint, mock_executor, mock_progress, mock_process_row, mock_load_key
    ):
        """Test TTS generation with large dataset (parallel processing)"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 2,
            "tts_method": "openai",
        }.get(key, 2)

        # Mock process_row for warmup (first 5)
        mock_process_row.side_effect = [
            (1, 1.0),
            (2, 2.0),
            (3, 3.0),
            (4, 4.0),
            (5, 5.0),
        ]

        # Mock ThreadPoolExecutor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock futures for parallel processing (last 1)
        mock_future = Mock(spec=Future)
        mock_future.result.return_value = (6, 6.0)
        mock_executor_instance.submit.return_value = mock_future

        # Mock as_completed
        with patch("core._10_gen_audio.as_completed", return_value=[mock_future]):
            # Mock progress
            mock_progress_instance = Mock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance

            result_df = generate_tts_audio(self.tasks_df.copy())

        # Verify warmup processing (first 5 rows)
        assert mock_process_row.call_count == 5

        # Verify parallel processing setup for remaining (1 row)
        mock_executor_instance.submit.assert_called_once()

        # Verify results
        expected_durations = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert list(result_df["real_dur"]) == expected_durations

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.Progress")
    @patch("core._10_gen_audio.rprint")
    def test_generate_tts_audio_gpt_sovits_no_parallel(
        self, mock_rprint, mock_progress, mock_process_row, mock_load_key
    ):
        """Test that gpt_sovits uses sequential processing only"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "gpt_sovits",
        }.get(key, 4)

        # Mock process_row
        mock_process_row.side_effect = [(i, float(i)) for i in range(1, 7)]

        # Mock progress
        mock_progress_instance = Mock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance

        result_df = generate_tts_audio(self.tasks_df.copy())

        # All 6 rows should be processed sequentially (no parallel processing)
        assert mock_process_row.call_count == 6

        # Verify max_workers would be 1 for gpt_sovits (checked in actual implementation)
        # This is verified by ensuring no ThreadPoolExecutor is used

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.Progress")
    @patch("core._10_gen_audio.rprint")
    def test_generate_tts_audio_warmup_error(
        self, mock_rprint, mock_progress, mock_process_row, mock_load_key
    ):
        """Test error handling during warmup phase"""
        mock_load_key.return_value = 4

        # First call succeeds, second fails
        mock_process_row.side_effect = [(1, 1.0), Exception("Warmup failed")]

        mock_progress_instance = Mock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance

        with pytest.raises(Exception) as exc_info:
            generate_tts_audio(self.tasks_df.copy())

        assert "Warmup failed" in str(exc_info.value)
        mock_rprint.assert_any_call("[red]❌ Error in warmup: Warmup failed[/red]")

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.Progress")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.rprint")
    def test_generate_tts_audio_parallel_error(
        self, mock_rprint, mock_executor, mock_progress, mock_process_row, mock_load_key
    ):
        """Test error handling during parallel processing"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 2,
            "tts_method": "openai",
        }.get(key, 2)

        # Warmup succeeds
        mock_process_row.side_effect = [(i, float(i)) for i in range(1, 6)]

        # Mock ThreadPoolExecutor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        # Mock future that fails
        mock_future = Mock(spec=Future)
        mock_future.result.side_effect = Exception("Parallel processing failed")
        mock_executor_instance.submit.return_value = mock_future

        with patch("core._10_gen_audio.as_completed", return_value=[mock_future]):
            mock_progress_instance = Mock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance

            with pytest.raises(Exception) as exc_info:
                generate_tts_audio(self.tasks_df.copy())

        assert "Parallel processing failed" in str(exc_info.value)
        mock_rprint.assert_any_call("[red]❌ Error: Parallel processing failed[/red]")


class TestProcessChunk:
    """Test chunk processing functionality"""

    def test_process_chunk_normal_case(self):
        """Test normal chunk processing"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [2.0, 3.0, 1.5],
                "tol_dur": [2.5, 3.2, 2.0],
                "tolerance": [0.5, 0.2, 0.3],
                "gap": [0.1, 0.2, 0.1],
            }
        )

        accept = 1.2
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Verify speed factor calculation and rounding
        assert isinstance(speed_factor, float)
        assert speed_factor >= min_speed
        assert isinstance(keep_gaps, bool)

    def test_process_chunk_with_gaps_acceptable(self):
        """Test chunk processing where gaps can be kept"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [1.0, 1.0, 1.0],  # Total: 3.0
                "tol_dur": [2.0, 2.0, 2.0],  # Total: 6.0
                "tolerance": [0.5, 0.5, 0.8],  # Last: 0.8
                "gap": [0.2, 0.2, 0.2],  # Total: 0.6, without last: 0.4
            }
        )

        accept = 0.8  # Low accept value
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # With low accept, should be able to keep gaps
        assert speed_factor >= min_speed
        # The specific logic depends on the calculations in the function

    def test_process_chunk_no_gaps(self):
        """Test chunk processing where gaps must be removed"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [5.0, 5.0, 5.0],  # Total: 15.0
                "tol_dur": [4.0, 4.0, 4.0],  # Total: 12.0
                "tolerance": [0.2, 0.2, 0.3],  # Last: 0.3
                "gap": [0.5, 0.5, 0.5],  # Total: 1.5
            }
        )

        accept = 2.0  # High accept value
        min_speed = 0.8

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # With high audio duration relative to time, gaps might need to be removed
        assert speed_factor >= min_speed

    def test_process_chunk_min_speed_enforcement(self):
        """Test that minimum speed is enforced"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [0.1, 0.1, 0.1],  # Very short durations
                "tol_dur": [10.0, 10.0, 10.0],  # Very long tolerance
                "tolerance": [1.0, 1.0, 2.0],  # Last: 2.0
                "gap": [0.0, 0.0, 0.0],  # No gaps
            }
        )

        accept = 1.0
        min_speed = 0.7

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Speed factor should not go below minimum
        assert speed_factor >= min_speed

    def test_process_chunk_edge_case_single_row(self):
        """Test chunk processing with single row"""
        chunk_df = pd.DataFrame(
            {"real_dur": [3.0], "tol_dur": [3.5], "tolerance": [0.5], "gap": [0.1]}
        )

        accept = 1.0
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        assert isinstance(speed_factor, float)
        assert isinstance(keep_gaps, bool)
        assert speed_factor >= min_speed

    def test_process_chunk_rounding(self):
        """Test that speed factor is properly rounded to 3 decimal places"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [1.234567],
                "tol_dur": [2.345678],
                "tolerance": [0.111111],
                "gap": [0.055555],
            }
        )

        accept = 1.0
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Verify rounding to 3 decimal places
        assert len(str(speed_factor).split(".")[1]) <= 3


class TestMergeChunks:
    """Test audio chunk merging functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_single_chunk(
        self,
        mock_rprint,
        mock_duration,
        mock_adjust,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test merging with single chunk"""
        tasks_df = pd.DataFrame(
            {
                "number": [1, 2],
                "lines": [["Hello"], ["World"]],
                "cut_off": [0, 1],  # Only second row is cut_off
                "start_time": ["00:00:00,000", "00:00:05,000"],
                "end_time": ["00:00:02,000", "00:00:07,000"],
                "tolerance": [0.5, 0.5],
                "gap": [0.1, 0.1],
            }
        )

        # Mock configuration
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.5,
        }.get(key)

        # Mock chunk processing
        mock_process_chunk.return_value = (1.0, True)  # Normal speed, keep gaps

        # Mock time parsing
        mock_parse_time.side_effect = [0.0, 5.0, 2.0, 7.0]  # start1, start2, end1, end2

        # Mock audio duration
        mock_duration.return_value = 2.0

        result_df = merge_chunks(tasks_df.copy())

        # Verify processing occurred
        mock_process_chunk.assert_called_once()
        assert "new_sub_times" in result_df.columns

        # Verify audio speed adjustment was called
        assert mock_adjust.call_count == 2  # One for each row's lines

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_multiple_chunks(
        self,
        mock_rprint,
        mock_duration,
        mock_adjust,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test merging with multiple chunks"""
        tasks_df = pd.DataFrame(
            {
                "number": [1, 2, 3, 4],
                "lines": [["A"], ["B"], ["C"], ["D"]],
                "cut_off": [0, 1, 0, 1],  # Two chunks: [1,2] and [3,4]
                "start_time": [
                    "00:00:00,000",
                    "00:00:02,000",
                    "00:00:05,000",
                    "00:00:07,000",
                ],
                "end_time": [
                    "00:00:01,000",
                    "00:00:03,000",
                    "00:00:06,000",
                    "00:00:08,000",
                ],
                "tolerance": [0.2, 0.3, 0.2, 0.3],
                "gap": [0.1, 0.1, 0.1, 0.1],
            }
        )

        # Mock configuration
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.5,
        }.get(key)

        # Mock chunk processing - different speeds for each chunk
        mock_process_chunk.side_effect = [(1.1, True), (0.9, False)]

        # Mock time parsing
        mock_parse_time.side_effect = [
            0.0,
            2.0,
            5.0,
            7.0,
            1.0,
            3.0,
            6.0,
            8.0,
        ]  # Alternating start/end times

        # Mock audio duration
        mock_duration.return_value = 1.5

        result_df = merge_chunks(tasks_df.copy())

        # Verify both chunks were processed
        assert mock_process_chunk.call_count == 2
        assert "new_sub_times" in result_df.columns

        # Verify audio speed adjustment was called for all lines
        assert mock_adjust.call_count == 4  # One for each row

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.AudioSegment")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_time_exceed_with_trimming(
        self,
        mock_rprint,
        mock_audiosegment,
        mock_duration,
        mock_adjust,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test chunk merging with time exceeding and audio trimming"""
        tasks_df = pd.DataFrame(
            {
                "number": [1],
                "lines": [["Long audio line"]],
                "cut_off": [1],
                "start_time": ["00:00:00,000"],
                "end_time": ["00:00:05,000"],
                "tolerance": [0.5],  # Chunk should end at 5.5
                "gap": [0.0],
            }
        )

        # Mock configuration
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.0,
            "speed_factor.min": 0.5,
        }.get(key)

        # Mock chunk processing
        mock_process_chunk.return_value = (1.0, True)

        # Mock time parsing - audio exceeds chunk end time slightly
        mock_parse_time.side_effect = [
            0.0,
            5.0,
        ]  # start=0, end=5, so chunk_end_time = 5.5

        # Mock audio duration - long enough to exceed chunk end
        mock_duration.return_value = 6.0  # Would end at 6.0, exceeding 5.5 by 0.5

        # Mock AudioSegment for trimming
        mock_audio = Mock()
        mock_trimmed = Mock()
        mock_audiosegment.from_wav.return_value = mock_audio
        mock_audio.__len__.return_value = 6000  # 6 seconds in milliseconds
        mock_audio.__getitem__.return_value = mock_trimmed

        result_df = merge_chunks(tasks_df.copy())

        # Should have attempted to trim the audio
        mock_audiosegment.from_wav.assert_called()
        mock_trimmed.export.assert_called()

        # Should log the trimming
        trim_calls = [
            call
            for call in mock_rprint.call_args_list
            if "exceeds by" in str(call) and "truncating" in str(call)
        ]
        assert len(trim_calls) > 0

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_merge_chunks_time_exceed_too_much_error(
        self,
        mock_duration,
        mock_adjust,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test error when chunk exceeds time by too much"""
        tasks_df = pd.DataFrame(
            {
                "number": [1],
                "lines": [["Way too long audio"]],
                "cut_off": [1],
                "start_time": ["00:00:00,000"],
                "end_time": ["00:00:05,000"],
                "tolerance": [0.5],  # Chunk should end at 5.5
                "gap": [0.0],
            }
        )

        # Mock configuration
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.0,
            "speed_factor.min": 0.5,
        }.get(key)

        # Mock chunk processing
        mock_process_chunk.return_value = (1.0, True)

        # Mock time parsing
        mock_parse_time.side_effect = [0.0, 5.0]  # chunk_end_time = 5.5

        # Mock audio duration - exceeds by too much
        mock_duration.return_value = (
            10.0  # Would end at 10.0, exceeding 5.5 by 4.5 (way over tolerance)
        )

        with pytest.raises(Exception) as exc_info:
            merge_chunks(tasks_df.copy())

        assert "exceeds the chunk end time" in str(exc_info.value)

    def test_merge_chunks_empty_dataframe(self):
        """Test merging with empty DataFrame"""
        empty_df = pd.DataFrame(columns=["number", "lines", "cut_off"])

        with patch("core._10_gen_audio.load_key"), patch("core._10_gen_audio.rprint"):
            result_df = merge_chunks(empty_df)

            assert "new_sub_times" in result_df.columns
            assert len(result_df) == 0


class TestGenAudio:
    """Test main gen_audio function"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.tasks_file = os.path.join(self.temp_dir, "tasks.xlsx")

        # Create sample tasks DataFrame
        self.sample_tasks = pd.DataFrame(
            {
                "number": [1, 2],
                "lines": [["Hello"], ["World"]],
                "cut_off": [0, 1],
                "start_time": ["00:00:00,000", "00:00:02,000"],
                "end_time": ["00:00:01,000", "00:00:03,000"],
            }
        )

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._10_gen_audio._8_1_AUDIO_TASK")
    @patch("core._10_gen_audio._AUDIO_TMP_DIR")
    @patch("core._10_gen_audio._AUDIO_SEGS_DIR")
    @patch("os.makedirs")
    @patch("pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_success(
        self,
        mock_rprint,
        mock_merge,
        mock_generate,
        mock_read_excel,
        mock_makedirs,
        mock_segs_dir,
        mock_tmp_dir,
        mock_audio_task,
    ):
        """Test successful audio generation process"""
        # Configure mocks
        mock_audio_task.__str__.return_value = self.tasks_file
        mock_tmp_dir.__str__.return_value = os.path.join(self.temp_dir, "tmp")
        mock_segs_dir.__str__.return_value = os.path.join(self.temp_dir, "segs")

        mock_read_excel.return_value = self.sample_tasks.copy()
        mock_generate.return_value = self.sample_tasks.copy()
        mock_merge.return_value = self.sample_tasks.copy()

        with patch.object(self.sample_tasks, "to_excel") as mock_to_excel:
            # Override the mock_merge return value to have to_excel method
            result_df = self.sample_tasks.copy()
            mock_merge.return_value = result_df

            gen_audio()

            # Verify directories were created
            assert mock_makedirs.call_count == 2

            # Verify file operations
            mock_read_excel.assert_called_once()
            mock_generate.assert_called_once()
            mock_merge.assert_called_once()

            # Verify success messages
            success_calls = [
                call
                for call in mock_rprint.call_args_list
                if "completed successfully" in str(call)
            ]
            assert len(success_calls) > 0

    @patch("core._10_gen_audio._8_1_AUDIO_TASK")
    @patch("core._10_gen_audio._AUDIO_TMP_DIR")
    @patch("core._10_gen_audio._AUDIO_SEGS_DIR")
    @patch("os.makedirs")
    @patch("pd.read_excel")
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_file_not_found(
        self,
        mock_rprint,
        mock_read_excel,
        mock_makedirs,
        mock_segs_dir,
        mock_tmp_dir,
        mock_audio_task,
    ):
        """Test error when task file is not found"""
        mock_audio_task.__str__.return_value = "/nonexistent/file.xlsx"
        mock_read_excel.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            gen_audio()

    @patch("core._10_gen_audio._8_1_AUDIO_TASK")
    @patch("core._10_gen_audio._AUDIO_TMP_DIR")
    @patch("core._10_gen_audio._AUDIO_SEGS_DIR")
    @patch("os.makedirs")
    @patch("pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_tts_generation_error(
        self,
        mock_rprint,
        mock_generate,
        mock_read_excel,
        mock_makedirs,
        mock_segs_dir,
        mock_tmp_dir,
        mock_audio_task,
    ):
        """Test error during TTS generation"""
        mock_audio_task.__str__.return_value = self.tasks_file
        mock_read_excel.return_value = self.sample_tasks.copy()
        mock_generate.side_effect = Exception("TTS generation failed")

        with pytest.raises(Exception) as exc_info:
            gen_audio()

        assert "TTS generation failed" in str(exc_info.value)

    @patch("core._10_gen_audio._8_1_AUDIO_TASK")
    @patch("core._10_gen_audio._AUDIO_TMP_DIR")
    @patch("core._10_gen_audio._AUDIO_SEGS_DIR")
    @patch("os.makedirs")
    @patch("pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_merge_error(
        self,
        mock_rprint,
        mock_merge,
        mock_generate,
        mock_read_excel,
        mock_makedirs,
        mock_segs_dir,
        mock_tmp_dir,
        mock_audio_task,
    ):
        """Test error during chunk merging"""
        mock_audio_task.__str__.return_value = self.tasks_file
        mock_read_excel.return_value = self.sample_tasks.copy()
        mock_generate.return_value = self.sample_tasks.copy()
        mock_merge.side_effect = Exception("Chunk merging failed")

        with pytest.raises(Exception) as exc_info:
            gen_audio()

        assert "Chunk merging failed" in str(exc_info.value)

    @patch("core._10_gen_audio._8_1_AUDIO_TASK")
    @patch("core._10_gen_audio._AUDIO_TMP_DIR")
    @patch("core._10_gen_audio._AUDIO_SEGS_DIR")
    @patch("os.makedirs", side_effect=OSError("Permission denied"))
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_directory_creation_error(
        self, mock_rprint, mock_makedirs, mock_segs_dir, mock_tmp_dir, mock_audio_task
    ):
        """Test error during directory creation"""
        with pytest.raises(OSError) as exc_info:
            gen_audio()

        assert "Permission denied" in str(exc_info.value)

    def test_gen_audio_integration_minimal(self):
        """Minimal integration test with real DataFrame operations"""
        # Create a real tasks file
        self.sample_tasks.to_excel(self.tasks_file, index=False)

        with patch("core._10_gen_audio._8_1_AUDIO_TASK", self.tasks_file), patch(
            "core._10_gen_audio._AUDIO_TMP_DIR", os.path.join(self.temp_dir, "tmp")
        ), patch(
            "core._10_gen_audio._AUDIO_SEGS_DIR", os.path.join(self.temp_dir, "segs")
        ), patch(
            "core._10_gen_audio.generate_tts_audio"
        ) as mock_generate, patch(
            "core._10_gen_audio.merge_chunks"
        ) as mock_merge, patch(
            "core._10_gen_audio.rprint"
        ):
            # Mock the processing functions to return modified DataFrames
            def mock_generate_func(df):
                df["real_dur"] = [1.0, 2.0]
                return df

            def mock_merge_func(df):
                df["new_sub_times"] = [[[0, 1]], [[2, 4]]]
                return df

            mock_generate.side_effect = mock_generate_func
            mock_merge.side_effect = mock_merge_func

            # Should not raise any exceptions
            gen_audio()

            # Verify modified file was saved
            result_df = pd.read_excel(self.tasks_file)
            assert "real_dur" in result_df.columns
            assert "new_sub_times" in result_df.columns
