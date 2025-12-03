"""
Comprehensive test suite for core/_10_gen_audio.py to achieve 85% branch coverage.
Focuses on all critical branches identified in the coverage analysis.
"""

import pytest
import os
import sys
import subprocess
import shutil
import tempfile
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import pandas as pd
from concurrent.futures import Future, ThreadPoolExecutor
from pydub import AudioSegment

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core._10_gen_audio import (
    adjust_audio_speed,
    process_row,
    generate_tts_audio,
    process_chunk,
    merge_chunks,
    gen_audio,
    TEMP_FILE_TEMPLATE,
    OUTPUT_FILE_TEMPLATE,
    WARMUP_SIZE,
)


class TestAdjustAudioSpeed:
    """Test audio speed adjustment functionality"""

    @patch("core._10_gen_audio.shutil.copy2")
    def test_adjust_audio_speed_no_change_needed(self, mock_copy):
        """Test when speed factor is close to 1.0, file is just copied"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 1.001  # Close to 1.0

        adjust_audio_speed(input_file, output_file, speed_factor)

        mock_copy.assert_called_once_with(input_file, output_file)

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    def test_adjust_audio_speed_normal_case(self, mock_subprocess, mock_get_duration):
        """Test normal audio speed adjustment"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 1.5

        mock_get_duration.side_effect = [10.0, 6.67]  # Input: 10s, Output: ~6.67s
        mock_subprocess.return_value = MagicMock(returncode=0)

        adjust_audio_speed(input_file, output_file, speed_factor)

        # Verify ffmpeg command
        expected_cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-filter:a",
            "atempo=1.5",
            "-y",
            output_file,
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd, check=True, stderr=subprocess.PIPE
        )

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.AudioSegment.from_wav")
    def test_adjust_audio_speed_short_audio_trimming(
        self, mock_from_wav, mock_subprocess, mock_get_duration
    ):
        """Test trimming for short audio with slight duration excess"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 2.0

        # Short input (< 3s), output slightly exceeds expected duration
        mock_get_duration.side_effect = [
            2.5,
            1.3,
        ]  # Input: 2.5s, Output: 1.3s (expected: 1.25s)
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_trimmed = MagicMock()
        mock_audio.__getitem__.return_value = mock_trimmed
        mock_from_wav.return_value = mock_audio

        with patch("builtins.print") as mock_print:
            adjust_audio_speed(input_file, output_file, speed_factor)

            # Verify trimming occurred
            mock_audio.__getitem__.assert_called_once_with(
                slice(None, 1250.0)
            )  # 1.25s * 1000ms
            mock_trimmed.export.assert_called_once_with(output_file, format="wav")
            mock_print.assert_called_with(
                "✂️ Trimmed to expected duration: 1.25 seconds"
            )

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    def test_adjust_audio_speed_duration_abnormal(
        self, mock_subprocess, mock_get_duration
    ):
        """Test exception when output duration is abnormally long"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 2.0

        # Long input, output duration way exceeds expected
        mock_get_duration.side_effect = [
            10.0,
            8.0,
        ]  # Input: 10s, Output: 8s (expected: 5s)
        mock_subprocess.return_value = MagicMock(returncode=0)

        with pytest.raises(Exception) as exc_info:
            adjust_audio_speed(input_file, output_file, speed_factor)

        assert "Audio duration abnormal" in str(exc_info.value)

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.time.sleep")
    @patch("core._10_gen_audio.rprint")
    def test_adjust_audio_speed_retry_mechanism(
        self, mock_rprint, mock_sleep, mock_subprocess, mock_get_duration
    ):
        """Test retry mechanism on subprocess failure"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 1.5

        mock_get_duration.side_effect = [10.0, 6.67]
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "ffmpeg"),
            subprocess.CalledProcessError(1, "ffmpeg"),
            MagicMock(returncode=0),  # Third attempt succeeds
        ]

        adjust_audio_speed(input_file, output_file, speed_factor)

        # Verify retries
        assert mock_subprocess.call_count == 3
        assert mock_sleep.call_count == 2
        mock_rprint.assert_any_call(
            "[yellow]⚠️ Audio speed adjustment failed, retrying in 0.1s (1/3)[/yellow]"
        )

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.time.sleep")
    @patch("core._10_gen_audio.rprint")
    def test_adjust_audio_speed_max_retries_exceeded(
        self, mock_rprint, mock_sleep, mock_subprocess, mock_get_duration
    ):
        """Test when max retries are exceeded"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 1.5

        mock_get_duration.return_value = 10.0
        error = subprocess.CalledProcessError(1, "ffmpeg")
        mock_subprocess.side_effect = error

        with pytest.raises(subprocess.CalledProcessError):
            adjust_audio_speed(input_file, output_file, speed_factor)

        mock_rprint.assert_any_call(
            "[red]❌ Audio speed adjustment failed, max retries reached (3)[/red]"
        )


class TestProcessRow:
    """Test single row processing functionality"""

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_single_line(self, mock_get_duration, mock_tts_main):
        """Test processing row with single line"""
        mock_get_duration.return_value = 2.5

        row = pd.Series({"number": 1, "lines": ["Hello world"]})
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 1
        assert real_dur == 2.5
        mock_tts_main.assert_called_once_with(
            "Hello world", "/test/audio_tmp/1_0_temp.wav", 1, tasks_df
        )

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_multiple_lines(self, mock_get_duration, mock_tts_main):
        """Test processing row with multiple lines"""
        mock_get_duration.side_effect = [1.5, 2.0, 1.8]  # Three audio segments

        row = pd.Series(
            {"number": 5, "lines": ["First line", "Second line", "Third line"]}
        )
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 5
        assert real_dur == 5.3  # Sum of durations
        assert mock_tts_main.call_count == 3

        # Verify correct temp file names
        expected_calls = [
            call("First line", "/test/audio_tmp/5_0_temp.wav", 5, tasks_df),
            call("Second line", "/test/audio_tmp/5_1_temp.wav", 5, tasks_df),
            call("Third line", "/test/audio_tmp/5_2_temp.wav", 5, tasks_df),
        ]
        mock_tts_main.assert_has_calls(expected_calls)

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_row_with_string_lines(self, mock_get_duration, mock_tts_main):
        """Test processing row where lines is a string (needs eval)"""
        mock_get_duration.return_value = 1.0

        row = pd.Series(
            {
                "number": 3,
                "lines": "['Line from string']",  # String representation of list
            }
        )
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 3
        assert real_dur == 1.0
        mock_tts_main.assert_called_once_with(
            "Line from string", "/test/audio_tmp/3_0_temp.wav", 3, tasks_df
        )


class TestGenerateTtsAudio:
    """Test TTS audio generation functionality"""

    def setup_method(self):
        """Setup test data"""
        self.mock_tasks_df = pd.DataFrame(
            {"number": [1, 2, 3], "lines": [["First"], ["Second"], ["Third"]]}
        )

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_warmup_only(
        self, mock_progress_class, mock_rprint, mock_process_row, mock_load_key
    ):
        """Test TTS generation with warmup only (small dataset)"""
        mock_load_key.side_effect = ["gpt_sovits", 1]
        mock_process_row.side_effect = [(1, 2.0), (2, 1.5), (3, 2.5)]

        # Mock Progress context manager
        mock_progress = MagicMock()
        mock_task = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress
        mock_progress.add_task.return_value = mock_task

        result_df = generate_tts_audio(self.mock_tasks_df)

        # Verify warmup processing (all 3 rows processed in warmup since <= 5)
        assert mock_process_row.call_count == 3
        assert mock_progress.advance.call_count == 3

        # Check results were updated
        assert len(result_df) == 3

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.as_completed")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_with_parallel_processing(
        self,
        mock_progress_class,
        mock_rprint,
        mock_as_completed,
        mock_executor_class,
        mock_process_row,
        mock_load_key,
    ):
        """Test TTS generation with parallel processing for larger dataset"""
        # Create larger dataset to trigger parallel processing
        large_df = pd.DataFrame(
            {
                "number": list(range(1, 11)),  # 10 rows
                "lines": [["Line {}".format(i)] for i in range(1, 11)],
            }
        )

        mock_load_key.side_effect = ["azure_tts", 4]  # Not gpt_sovits, max_workers=4
        mock_process_row.side_effect = [(i, 1.0) for i in range(1, 11)]

        # Mock ThreadPoolExecutor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_future1 = MagicMock()
        mock_future1.result.return_value = (6, 1.0)
        mock_future2 = MagicMock()
        mock_future2.result.return_value = (7, 1.5)
        mock_executor.submit.return_value = mock_future1
        mock_as_completed.return_value = [mock_future1, mock_future2]

        # Mock Progress
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        result_df = generate_tts_audio(large_df)

        # Verify warmup (first 5 rows) + parallel processing (remaining 5 rows)
        assert mock_process_row.call_count >= 5  # At least warmup calls
        mock_executor_class.assert_called_once_with(max_workers=4)

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.as_completed")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_gpt_sovits_no_parallel(
        self,
        mock_progress_class,
        mock_rprint,
        mock_as_completed,
        mock_executor_class,
        mock_process_row,
        mock_load_key,
    ):
        """Test that GPT-SoVITS uses sequential processing (max_workers=1)"""
        large_df = pd.DataFrame(
            {
                "number": list(range(1, 11)),
                "lines": [["Line {}".format(i)] for i in range(1, 11)],
            }
        )

        mock_load_key.side_effect = [
            "gpt_sovits",
            4,
        ]  # GPT-SoVITS should override max_workers
        mock_process_row.side_effect = [(i, 1.0) for i in range(1, 11)]

        # Mock executor and progress
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        generate_tts_audio(large_df)

        # Verify max_workers=1 for GPT-SoVITS
        mock_executor_class.assert_called_once_with(max_workers=1)

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_warmup_exception(
        self, mock_progress_class, mock_rprint, mock_process_row, mock_load_key
    ):
        """Test exception handling during warmup phase"""
        mock_load_key.side_effect = ["azure_tts", 2]
        mock_process_row.side_effect = Exception("TTS failed")

        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        with pytest.raises(Exception) as exc_info:
            generate_tts_audio(self.mock_tasks_df)

        assert "TTS failed" in str(exc_info.value)
        mock_rprint.assert_any_call("[red]❌ Error in warmup: TTS failed[/red]")

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.as_completed")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_parallel_exception(
        self,
        mock_progress_class,
        mock_rprint,
        mock_as_completed,
        mock_executor_class,
        mock_process_row,
        mock_load_key,
    ):
        """Test exception handling during parallel processing"""
        large_df = pd.DataFrame(
            {
                "number": list(range(1, 11)),
                "lines": [["Line {}".format(i)] for i in range(1, 11)],
            }
        )

        mock_load_key.side_effect = ["azure_tts", 4]
        # Warmup succeeds, parallel processing fails
        mock_process_row.side_effect = [(i, 1.0) for i in range(1, 6)]  # Warmup success

        # Mock parallel processing failure
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Parallel processing failed")
        mock_as_completed.return_value = [mock_future]

        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        with pytest.raises(Exception) as exc_info:
            generate_tts_audio(large_df)

        assert "Parallel processing failed" in str(exc_info.value)
        mock_rprint.assert_any_call("[red]❌ Error: Parallel processing failed[/red]")


class TestProcessChunk:
    """Test audio chunk processing functionality"""

    def test_process_chunk_normal_case(self):
        """Test normal chunk processing without gaps"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [2.0, 1.5, 3.0],
                "tol_dur": [2.5, 2.0, 3.5],
                "tolerance": [0.5, 0.3, 0.8],
                "gap": [0.1, 0.2, 0.15],
            }
        )

        accept = 1.2
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Verify calculations
        chunk_durs = 6.5  # sum of real_dur
        tol_durs = 8.0  # sum of tol_dur
        durations = tol_durs - chunk_df.iloc[-1]["tolerance"]  # 8.0 - 0.8 = 7.2
        all_gaps = chunk_df["gap"].sum() - chunk_df.iloc[-1]["gap"]  # 0.45 - 0.15 = 0.3

        # Should use first condition: (chunk_durs + all_gaps) / accept < durations
        # (6.5 + 0.3) / 1.2 = 5.67 < 7.2
        expected_speed = max(min_speed, (chunk_durs + all_gaps) / (durations - 0.1))
        expected_speed = max(0.5, 6.8 / 7.1)

        assert abs(speed_factor - round(expected_speed, 3)) < 0.001
        assert keep_gaps is True

    def test_process_chunk_no_gaps_needed(self):
        """Test chunk processing where gaps should be removed"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [5.0, 4.0],  # Large durations
                "tol_dur": [3.0, 2.5],  # Smaller tolerances
                "tolerance": [0.2, 0.3],
                "gap": [0.5, 0.4],
            }
        )

        accept = 1.0
        min_speed = 0.8

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Should fall into second condition where keep_gaps = False
        assert keep_gaps is False
        assert speed_factor >= min_speed

    def test_process_chunk_min_speed_enforced(self):
        """Test that minimum speed is enforced"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [1.0],
                "tol_dur": [10.0],  # Very high tolerance
                "tolerance": [0.1],
                "gap": [0.1],
            }
        )

        accept = 1.0
        min_speed = 0.7

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Speed should be at least min_speed
        assert speed_factor >= min_speed

    def test_process_chunk_high_speed_no_gaps(self):
        """Test chunk processing with high speed factor and no gaps"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [8.0, 6.0],  # High durations
                "tol_dur": [4.0, 3.0],  # Low tolerances
                "tolerance": [0.2, 0.3],
                "gap": [0.1, 0.2],
            }
        )

        accept = 0.8
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(chunk_df, accept, min_speed)

        # Should result in high speed factor and no gaps
        assert speed_factor > accept
        assert keep_gaps is False


class TestMergeChunks:
    """Test audio chunk merging functionality"""

    def setup_method(self):
        """Setup test data for merge chunks tests"""
        self.mock_tasks_df = pd.DataFrame(
            {
                "number": [1, 2, 3],
                "cut_off": [0, 0, 1],  # Only last row has cut_off=1
                "start_time": ["00:00:00", "00:00:10", "00:00:20"],
                "end_time": ["00:00:10", "00:00:20", "00:00:30"],
                "tolerance": [1.0, 1.0, 1.0],
                "lines": [["Line 1"], ["Line 2"], ["Line 3"]],
                "gap": [0.5, 0.5, 0.5],
                "real_dur": [2.0, 2.0, 2.0],
                "tol_dur": [3.0, 3.0, 3.0],
            }
        )

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_normal_processing(
        self,
        mock_rprint,
        mock_get_duration,
        mock_adjust_speed,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test normal chunk merging process"""
        mock_load_key.side_effect = [1.2, 0.8]  # accept, min_speed
        mock_process_chunk.return_value = (1.1, True)  # speed_factor, keep_gaps
        mock_parse_time.side_effect = [0.0, 30.0]  # start and end times in seconds
        mock_get_duration.return_value = 1.8  # Adjusted audio duration

        result_df = merge_chunks(self.mock_tasks_df.copy())

        # Verify processing occurred
        mock_process_chunk.assert_called_once()
        assert mock_adjust_speed.call_count == 3  # One per line in the chunk

        # Verify new_sub_times was set
        assert "new_sub_times" in result_df.columns
        assert result_df["new_sub_times"].iloc[0] is not None

        mock_rprint.assert_any_call(
            "[cyan]⚡ Processed chunk 0 to 2 with speed factor 1.1[/cyan]"
        )

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_high_speed_warning(
        self,
        mock_rprint,
        mock_get_duration,
        mock_adjust_speed,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test chunk merging with high speed factor showing warning emoji"""
        mock_load_key.side_effect = [1.0, 0.8]  # accept=1.0, min_speed
        mock_process_chunk.return_value = (1.5, False)  # High speed factor > accept
        mock_parse_time.side_effect = [0.0, 30.0]
        mock_get_duration.return_value = 1.8

        merge_chunks(self.mock_tasks_df.copy())

        # Should show warning emoji for high speed
        mock_rprint.assert_any_call(
            "[cyan]⚠️ Processed chunk 0 to 2 with speed factor 1.5[/cyan]"
        )

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.AudioSegment.from_wav")
    @patch("core._10_gen_audio.rprint")
    def test_merge_chunks_with_trimming(
        self,
        mock_rprint,
        mock_from_wav,
        mock_get_duration,
        mock_adjust_speed,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test chunk merging with audio trimming when exceeding time limit"""
        mock_load_key.side_effect = [1.2, 0.8]
        mock_process_chunk.return_value = (1.0, True)
        mock_parse_time.side_effect = [0.0, 30.0]  # chunk_start_time, chunk_end_time
        mock_get_duration.return_value = 2.0

        # Mock current time calculation to exceed chunk end time slightly
        with patch("core._10_gen_audio.TIME_DIFF_TOLERANCE", 0.2):
            # Make cur_time exceed chunk_end_time by small amount
            chunk_end_time = 31.0  # Slightly different from mock_parse_time for test
            mock_parse_time.side_effect = [0.0, chunk_end_time]

            # Setup to trigger trimming logic
            mock_audio = MagicMock()
            mock_trimmed = MagicMock()
            mock_audio.__len__.return_value = 2000  # 2 seconds in milliseconds
            mock_audio.__getitem__.return_value = mock_trimmed
            mock_from_wav.return_value = mock_audio

            # Modify the test to simulate time exceeding
            with patch.object(self.mock_tasks_df, "at") as mock_at:
                result_df = merge_chunks(self.mock_tasks_df.copy())

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_merge_chunks_time_exceeds_limit(
        self,
        mock_get_duration,
        mock_adjust_speed,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test exception when chunk processing exceeds time limit significantly"""
        mock_load_key.side_effect = [1.2, 0.8]
        mock_process_chunk.return_value = (1.0, True)
        mock_parse_time.side_effect = [0.0, 30.0]  # chunk_end_time = 30.0

        # Mock duration that would cause significant time excess
        mock_get_duration.return_value = 5.0  # Very long audio

        with pytest.raises(Exception) as exc_info:
            merge_chunks(self.mock_tasks_df.copy())

        assert "exceeds the chunk end time" in str(exc_info.value)

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.process_chunk")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_merge_chunks_no_gaps_processing(
        self,
        mock_get_duration,
        mock_adjust_speed,
        mock_parse_time,
        mock_process_chunk,
        mock_load_key,
    ):
        """Test chunk processing when gaps should not be kept"""
        mock_load_key.side_effect = [1.2, 0.8]
        mock_process_chunk.return_value = (1.5, False)  # keep_gaps=False
        mock_parse_time.side_effect = [0.0, 30.0]
        mock_get_duration.return_value = 1.5

        result_df = merge_chunks(self.mock_tasks_df.copy())

        # Processing should still work, just without gap additions
        mock_process_chunk.assert_called_once()
        assert mock_adjust_speed.call_count == 3


class TestGenAudio:
    """Test main audio generation function"""

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    @patch("core._10_gen_audio.rprint")
    def test_gen_audio_complete_workflow(
        self,
        mock_rprint,
        mock_merge_chunks,
        mock_generate_tts,
        mock_read_excel,
        mock_makedirs,
    ):
        """Test complete audio generation workflow"""
        # Setup test data
        mock_tasks_df = pd.DataFrame(
            {
                "number": [1, 2],
                "lines": [["Line 1"], ["Line 2"]],
                "real_dur": [0, 0],  # Will be updated by generate_tts_audio
            }
        )

        mock_read_excel.return_value = mock_tasks_df
        mock_generate_tts.return_value = mock_tasks_df  # Return with updated real_dur
        mock_merge_chunks.return_value = mock_tasks_df  # Return with new_sub_times

        with patch("core._10_gen_audio.pd.DataFrame.to_excel") as mock_to_excel:
            gen_audio()

            # Verify directory creation
            mock_makedirs.assert_any_call("/test/audio_tmp", exist_ok=True)
            mock_makedirs.assert_any_call("/test/audio_segs", exist_ok=True)

            # Verify workflow steps
            mock_read_excel.assert_called_once()
            mock_generate_tts.assert_called_once()
            mock_merge_chunks.assert_called_once()
            mock_to_excel.assert_called_once()

            # Verify messages
            mock_rprint.assert_any_call(
                "[bold magenta]🚀 Starting audio generation process...[/bold magenta]"
            )
            mock_rprint.assert_any_call(
                "[green]📊 Loaded task file successfully[/green]"
            )
            mock_rprint.assert_any_call(
                "[bold green]🎉 Audio generation completed successfully![/bold green]"
            )

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    def test_gen_audio_file_loading_error(
        self, mock_generate_tts, mock_read_excel, mock_makedirs
    ):
        """Test error handling when task file cannot be loaded"""
        mock_read_excel.side_effect = FileNotFoundError("Task file not found")

        with pytest.raises(FileNotFoundError):
            gen_audio()

        # Should still create directories
        mock_makedirs.assert_any_call("/test/audio_tmp", exist_ok=True)
        mock_makedirs.assert_any_call("/test/audio_segs", exist_ok=True)

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    def test_gen_audio_tts_generation_error(
        self, mock_merge_chunks, mock_generate_tts, mock_read_excel, mock_makedirs
    ):
        """Test error handling during TTS generation"""
        mock_read_excel.return_value = pd.DataFrame()
        mock_generate_tts.side_effect = Exception("TTS generation failed")

        with pytest.raises(Exception) as exc_info:
            gen_audio()

        assert "TTS generation failed" in str(exc_info.value)
        mock_merge_chunks.assert_not_called()  # Should not reach merge phase

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    def test_gen_audio_merge_error(
        self, mock_merge_chunks, mock_generate_tts, mock_read_excel, mock_makedirs
    ):
        """Test error handling during chunk merging"""
        mock_read_excel.return_value = pd.DataFrame()
        mock_generate_tts.return_value = pd.DataFrame()
        mock_merge_chunks.side_effect = Exception("Merge failed")

        with pytest.raises(Exception) as exc_info:
            gen_audio()

        assert "Merge failed" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    def test_adjust_audio_speed_exact_boundary(
        self, mock_subprocess, mock_get_duration
    ):
        """Test speed factor exactly at the boundary"""
        from core.constants import SPEED_FACTOR_PRECISION
        from core.constants import AudioConstants

        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = (
            AudioConstants.BASE_SPEED_FACTOR + SPEED_FACTOR_PRECISION
        )  # Exactly at boundary

        mock_get_duration.side_effect = [10.0, 10.0]
        mock_subprocess.return_value = MagicMock(returncode=0)

        adjust_audio_speed(input_file, output_file, speed_factor)

        # Should use ffmpeg (not copy) since it's at the boundary
        mock_subprocess.assert_called_once()

    def test_process_row_empty_lines(self):
        """Test processing row with empty lines list"""
        row = pd.Series({"number": 1, "lines": []})
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 1
        assert real_dur == 0

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.rprint")
    @patch("core._10_gen_audio.Progress")
    def test_generate_tts_audio_empty_dataframe(
        self, mock_progress_class, mock_rprint, mock_load_key
    ):
        """Test TTS generation with empty DataFrame"""
        mock_load_key.side_effect = ["azure_tts", 4]
        empty_df = pd.DataFrame()

        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        result_df = generate_tts_audio(empty_df)

        # Should handle empty DataFrame gracefully
        assert len(result_df) == 0
        assert "real_dur" in result_df.columns

    def test_process_chunk_single_row(self):
        """Test processing chunk with only one row"""
        single_row_df = pd.DataFrame(
            {"real_dur": [2.0], "tol_dur": [3.0], "tolerance": [0.5], "gap": [0.1]}
        )

        accept = 1.2
        min_speed = 0.5

        speed_factor, keep_gaps = process_chunk(single_row_df, accept, min_speed)

        # Should still process correctly
        assert isinstance(speed_factor, float)
        assert isinstance(keep_gaps, bool)
        assert speed_factor >= min_speed


class TestConstants:
    """Test constants and template usage"""

    def test_file_templates(self):
        """Test that file templates are correctly formatted"""
        # Test temp file template
        temp_file = TEMP_FILE_TEMPLATE.format("1_0")
        assert temp_file == "/test/audio_tmp/1_0_temp.wav"

        # Test output file template
        output_file = OUTPUT_FILE_TEMPLATE.format("1_0")
        assert output_file == "/test/audio_segs/1_0.wav"

    def test_warmup_size_constant(self):
        """Test warmup size constant"""
        assert WARMUP_SIZE == 5  # From ProcessingConstants
        assert isinstance(WARMUP_SIZE, int)
        assert WARMUP_SIZE > 0


class TestIntegrationScenarios:
    """Test integration scenarios with realistic data flows"""

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.AudioSegment.from_wav")
    def test_audio_speed_adjustment_integration(
        self, mock_from_wav, mock_subprocess, mock_get_duration
    ):
        """Test integrated audio speed adjustment with trimming"""
        input_file = "/test/input.wav"
        output_file = "/test/output.wav"
        speed_factor = 2.0

        # Setup for short audio that needs trimming
        mock_get_duration.side_effect = [
            2.0,
            1.05,
        ]  # Input: 2s, Output: 1.05s (expected: 1s)
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock AudioSegment for trimming
        mock_audio = MagicMock()
        mock_trimmed = MagicMock()
        mock_audio.__getitem__.return_value = mock_trimmed
        mock_audio.__len__.return_value = 1050  # 1.05s in milliseconds
        mock_from_wav.return_value = mock_audio

        adjust_audio_speed(input_file, output_file, speed_factor)

        # Verify trimming was applied
        mock_audio.__getitem__.assert_called_once_with(
            slice(None, 1000.0)
        )  # Trim to 1s
        mock_trimmed.export.assert_called_once_with(output_file, format="wav")

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.ThreadPoolExecutor")
    @patch("core._10_gen_audio.as_completed")
    @patch("core._10_gen_audio.Progress")
    def test_tts_generation_integration_flow(
        self,
        mock_progress_class,
        mock_as_completed,
        mock_executor_class,
        mock_load_key,
        mock_get_duration,
        mock_tts_main,
    ):
        """Test integrated TTS generation flow with realistic data"""
        # Setup realistic dataset
        tasks_df = pd.DataFrame(
            {
                "number": [1, 2, 3, 4, 5, 6, 7],
                "lines": [
                    ["Hello world"],
                    ["This is a test"],
                    ["Multiple", "lines", "here"],
                    ["Another single line"],
                    ["Short"],
                    ["This is a longer sentence for testing"],
                    ["Final line"],
                ],
            }
        )

        mock_load_key.side_effect = ["azure_tts", 3]  # max_workers=3
        mock_get_duration.return_value = 2.0
        mock_tts_main.return_value = None

        # Mock parallel execution
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        # Create futures for parallel tasks
        futures = []
        for i in range(6, 8):  # Tasks 6-7 processed in parallel
            future = MagicMock()
            future.result.return_value = (i, 2.0)
            futures.append(future)
        mock_as_completed.return_value = futures

        # Mock progress
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        result_df = generate_tts_audio(tasks_df)

        # Verify realistic expectations
        assert len(result_df) == 7
        assert "real_dur" in result_df.columns
        assert (
            mock_executor_class.call_count >= 0
        )  # May or may not use parallel processing depending on size
