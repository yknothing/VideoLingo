"""
Comprehensive test suite for core/_10_gen_audio.py
Tests TTS audio generation, speed adjustment, timeline processing, and chunk merging functionality.
"""
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import pytest
from pydub import AudioSegment

# Import the module under test
import sys

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
    """Test cases for adjust_audio_speed function"""

    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.shutil.copy2")
    def test_speed_factor_near_one_copies_file(self, mock_copy, mock_duration):
        """Test that speed factor close to 1.0 just copies the file"""
        adjust_audio_speed("input.wav", "output.wav", 1.001)
        mock_copy.assert_called_once_with("input.wav", "output.wav")
        mock_duration.assert_not_called()

    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_speed_adjustment_success(self, mock_duration, mock_subprocess):
        """Test successful audio speed adjustment"""
        mock_duration.side_effect = [10.0, 5.0]  # input: 10s, output: 5s (2x speed)
        mock_subprocess.return_value.returncode = 0

        adjust_audio_speed("input.wav", "output.wav", 2.0)

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "atempo=2.0" in call_args

    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.AudioSegment.from_wav")
    def test_duration_trimming_for_short_audio(
        self, mock_audio_segment, mock_duration, mock_subprocess
    ):
        """Test duration trimming for short audio files with acceptable error"""
        # Setup: short input audio (2.5s) with slightly long output
        mock_duration.side_effect = [
            2.5,
            5.2,
        ]  # input: 2.5s, output: 5.2s (expected: 5.0s)
        mock_subprocess.return_value.returncode = 0

        # Mock AudioSegment operations
        mock_audio = Mock()
        mock_audio_segment.return_value = mock_audio
        mock_audio.__getitem__.return_value = mock_audio

        adjust_audio_speed("input.wav", "output.wav", 0.5)

        # Should trim to expected duration (5.0 seconds)
        mock_audio.__getitem__.assert_called_once_with(
            slice(None, 5000)
        )  # 5s in milliseconds
        mock_audio.export.assert_called_once_with("output.wav", format="wav")

    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_abnormal_duration_raises_exception(self, mock_duration, mock_subprocess):
        """Test that abnormal duration differences raise an exception"""
        mock_duration.side_effect = [10.0, 20.0]  # output much longer than expected
        mock_subprocess.return_value.returncode = 0

        with pytest.raises(Exception, match="Audio duration abnormal"):
            adjust_audio_speed("input.wav", "output.wav", 2.0)

    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.time.sleep")
    @patch("core._10_gen_audio.rprint")
    def test_retry_mechanism(
        self, mock_rprint, mock_sleep, mock_duration, mock_subprocess
    ):
        """Test retry mechanism for failed subprocess calls"""
        mock_duration.return_value = 10.0
        # First two calls fail, third succeeds
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "ffmpeg"),
            subprocess.CalledProcessError(1, "ffmpeg"),
            Mock(returncode=0),
        ]

        adjust_audio_speed("input.wav", "output.wav", 2.0)

        assert mock_subprocess.call_count == 3
        assert mock_sleep.call_count == 2
        assert mock_rprint.call_count >= 2  # Warning messages

    @patch("core._10_gen_audio.subprocess.run")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_max_retries_exceeded_raises_error(self, mock_duration, mock_subprocess):
        """Test that exceeding max retries raises the original error"""
        mock_duration.return_value = 10.0
        error = subprocess.CalledProcessError(1, "ffmpeg")
        mock_subprocess.side_effect = error

        with pytest.raises(subprocess.CalledProcessError):
            adjust_audio_speed("input.wav", "output.wav", 2.0)


class TestProcessRow:
    """Test cases for process_row function"""

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_single_line(self, mock_duration, mock_tts):
        """Test processing a single line in a row"""
        mock_duration.return_value = 5.0

        row = pd.Series({"number": 1, "lines": ["Hello world"]})
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 1
        assert real_dur == 5.0
        mock_tts.assert_called_once_with(
            "Hello world", "/tmp/audio_tmp/1_0_temp.wav", 1, tasks_df
        )

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_multiple_lines(self, mock_duration, mock_tts):
        """Test processing multiple lines in a row"""
        mock_duration.side_effect = [3.0, 4.0, 2.0]

        row = pd.Series({"number": 2, "lines": ["Line 1", "Line 2", "Line 3"]})
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 2
        assert real_dur == 9.0  # 3 + 4 + 2
        assert mock_tts.call_count == 3

    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    def test_process_string_lines(self, mock_duration, mock_tts):
        """Test processing lines stored as string (eval format)"""
        mock_duration.return_value = 5.0

        row = pd.Series({"number": 3, "lines": "['Hello', 'World']"})
        tasks_df = pd.DataFrame()

        number, real_dur = process_row(row, tasks_df)

        assert number == 3
        assert real_dur == 10.0  # 2 lines × 5.0s each
        assert mock_tts.call_count == 2


class TestGenerateTtsAudio:
    """Test cases for generate_tts_audio function"""

    def setup_method(self):
        """Setup test data"""
        self.tasks_df = pd.DataFrame(
            {
                "number": [1, 2, 3, 4, 5],
                "lines": [["Line 1"], ["Line 2"], ["Line 3"], ["Line 4"], ["Line 5"]],
            }
        )

    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.load_key")
    def test_warmup_processing(self, mock_load_key, mock_process_row):
        """Test warmup processing for first few rows"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "openai",
        }.get(key, None)
        mock_process_row.side_effect = [
            (1, 5.0),
            (2, 4.0),
            (3, 6.0),
            (4, 3.0),
            (5, 7.0),
        ]

        result_df = generate_tts_audio(self.tasks_df.copy())

        # Check that all rows were processed
        assert len(result_df) == 5
        assert "real_dur" in result_df.columns
        assert mock_process_row.call_count == 5

    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.load_key")
    def test_gpt_sovits_single_worker(self, mock_load_key, mock_process_row):
        """Test that GPT-SoVITS uses single worker mode"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "gpt_sovits",
        }.get(key, None)
        mock_process_row.side_effect = [(1, 5.0), (2, 4.0), (3, 6.0)]

        small_df = self.tasks_df.head(3).copy()
        generate_tts_audio(small_df)

        # GPT-SoVITS should still process but with max_workers=1
        assert mock_process_row.call_count == 3

    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.load_key")
    def test_parallel_processing_after_warmup(self, mock_load_key, mock_process_row):
        """Test parallel processing for tasks after warmup"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "openai",
        }.get(key, None)
        mock_process_row.side_effect = [(i, float(i)) for i in range(1, 11)]

        large_df = pd.DataFrame(
            {"number": range(1, 11), "lines": [["Line"] for _ in range(10)]}
        )

        result_df = generate_tts_audio(large_df)

        assert len(result_df) == 10
        assert mock_process_row.call_count == 10

    @patch("core._10_gen_audio.process_row")
    def test_error_handling_in_warmup(self, mock_process_row):
        """Test error handling during warmup phase"""
        mock_process_row.side_effect = Exception("TTS generation failed")

        with pytest.raises(Exception, match="TTS generation failed"):
            generate_tts_audio(self.tasks_df.copy())

    @patch("core._10_gen_audio.process_row")
    @patch("core._10_gen_audio.load_key")
    def test_error_handling_in_parallel(self, mock_load_key, mock_process_row):
        """Test error handling during parallel processing"""
        mock_load_key.side_effect = lambda key: {
            "max_workers": 4,
            "tts_method": "openai",
        }.get(key, None)
        # First few succeed (warmup), then one fails
        mock_process_row.side_effect = [(1, 5.0), (2, 4.0), Exception("Parallel error")]

        large_df = pd.DataFrame(
            {"number": range(1, 8), "lines": [["Line"] for _ in range(7)]}
        )

        with pytest.raises(Exception, match="Parallel error"):
            generate_tts_audio(large_df)


class TestProcessChunk:
    """Test cases for process_chunk function"""

    def test_normal_timing_with_gaps(self):
        """Test normal timing calculation with gaps preserved"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [5.0, 4.0, 6.0],
                "tol_dur": [6.0, 5.0, 7.0],
                "tolerance": [0.5, 0.5, 1.0],  # Last row tolerance used
                "gap": [1.0, 0.5, 0.8],  # Last row gap excluded
            }
        )

        speed_factor, keep_gaps = process_chunk(chunk_df, accept=1.2, min_speed=0.5)

        # Should preserve gaps and use normal speed
        assert keep_gaps == True
        assert speed_factor > 0

    def test_timing_requires_speed_up_with_gaps(self):
        """Test timing that requires speed up but can keep gaps"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [8.0, 7.0, 9.0],  # Long durations
                "tol_dur": [6.0, 5.0, 7.0],  # Shorter tolerance
                "tolerance": [0.5, 0.5, 1.0],
                "gap": [1.0, 0.5, 0.8],
            }
        )

        speed_factor, keep_gaps = process_chunk(chunk_df, accept=1.5, min_speed=0.8)

        # Should calculate appropriate speed factor
        assert speed_factor >= 0.8
        assert isinstance(keep_gaps, bool)

    def test_timing_requires_removing_gaps(self):
        """Test timing that requires removing gaps"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [10.0, 9.0, 11.0],  # Very long durations
                "tol_dur": [6.0, 5.0, 7.0],  # Much shorter tolerance
                "tolerance": [0.5, 0.5, 1.0],
                "gap": [1.0, 0.5, 0.8],
            }
        )

        speed_factor, keep_gaps = process_chunk(chunk_df, accept=1.2, min_speed=0.5)

        # Should remove gaps due to timing constraints
        assert speed_factor >= 0.5
        assert keep_gaps == False

    def test_min_speed_enforcement(self):
        """Test that minimum speed is enforced"""
        chunk_df = pd.DataFrame(
            {
                "real_dur": [20.0, 18.0, 22.0],  # Extremely long durations
                "tol_dur": [2.0, 3.0, 4.0],  # Very short tolerance
                "tolerance": [0.5, 0.5, 1.0],
                "gap": [1.0, 0.5, 0.8],
            }
        )

        speed_factor, keep_gaps = process_chunk(chunk_df, accept=1.2, min_speed=2.0)

        assert speed_factor == 2.0  # Should be clamped to min_speed

    def test_empty_chunk_handling(self):
        """Test handling of empty or minimal chunks"""
        chunk_df = pd.DataFrame(
            {"real_dur": [1.0], "tol_dur": [2.0], "tolerance": [0.5], "gap": [0.1]}
        )

        speed_factor, keep_gaps = process_chunk(chunk_df, accept=1.2, min_speed=0.5)

        assert speed_factor > 0
        assert isinstance(keep_gaps, bool)


class TestMergeChunks:
    """Test cases for merge_chunks function"""

    def setup_method(self):
        """Setup test data for merge_chunks tests"""
        self.tasks_df = pd.DataFrame(
            {
                "number": [1, 2, 3],
                "lines": [["Line 1"], ["Line 2"], ["Line 3"]],
                "cut_off": [0, 0, 1],  # Last row marks end of chunk
                "start_time": ["00:00:00,000", "00:00:05,000", "00:00:10,000"],
                "end_time": ["00:00:05,000", "00:00:10,000", "00:00:15,000"],
                "tolerance": [1.0, 1.0, 2.0],
                "gap": [0.5, 0.3, 0.7],
                "real_dur": [4.5, 4.8, 5.2],
                "tol_dur": [5.0, 5.0, 6.0],
            }
        )

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.parse_df_srt_time")
    def test_successful_chunk_merge(
        self, mock_parse_time, mock_duration, mock_adjust, mock_load_key
    ):
        """Test successful chunk merging with timeline calculation"""
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.8,
        }.get(key, 1.0)
        mock_parse_time.side_effect = [
            0.0,
            5.0,
            10.0,
            15.0,
        ]  # Start/end times in seconds
        mock_duration.return_value = 4.0  # Adjusted duration

        result_df = merge_chunks(self.tasks_df.copy())

        # Should add new_sub_times column
        assert "new_sub_times" in result_df.columns
        assert mock_adjust.call_count == 3  # One for each line

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.parse_df_srt_time")
    @patch("core._10_gen_audio.AudioSegment.from_wav")
    def test_chunk_exceeds_timeline_trimming(
        self,
        mock_audio_segment,
        mock_parse_time,
        mock_duration,
        mock_adjust,
        mock_load_key,
    ):
        """Test handling when chunk exceeds timeline and requires trimming"""
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.8,
        }.get(key, 1.0)
        mock_parse_time.side_effect = [0.0, 5.0, 10.0, 17.0]  # Timeline: 0-17s
        mock_duration.return_value = 6.0  # Each audio is 6s, total will exceed

        # Mock AudioSegment for trimming
        mock_audio = Mock()
        mock_audio_segment.return_value = mock_audio
        mock_audio.__len__.return_value = 6000  # 6 seconds in milliseconds

        result_df = merge_chunks(self.tasks_df.copy())

        # Should have attempted trimming
        assert "new_sub_times" in result_df.columns

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.parse_df_srt_time")
    def test_chunk_exceeds_tolerance_raises_error(
        self, mock_parse_time, mock_duration, mock_adjust, mock_load_key
    ):
        """Test that exceeding tolerance raises an exception"""
        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.8,
        }.get(key, 1.0)
        mock_parse_time.side_effect = [0.0, 5.0, 10.0, 15.0]  # Timeline: 0-15s
        mock_duration.return_value = 10.0  # Each audio is 10s, will greatly exceed

        with pytest.raises(Exception, match="exceeds the chunk end time"):
            merge_chunks(self.tasks_df.copy())

    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.parse_df_srt_time")
    def test_multiple_chunks_processing(
        self, mock_parse_time, mock_duration, mock_adjust, mock_load_key
    ):
        """Test processing multiple chunks in sequence"""
        # Create data with two chunks
        multi_chunk_df = pd.DataFrame(
            {
                "number": [1, 2, 3, 4],
                "lines": [["Line 1"], ["Line 2"], ["Line 3"], ["Line 4"]],
                "cut_off": [0, 1, 0, 1],  # Two chunks: [1,2] and [3,4]
                "start_time": [
                    "00:00:00,000",
                    "00:00:05,000",
                    "00:00:15,000",
                    "00:00:20,000",
                ],
                "end_time": [
                    "00:00:05,000",
                    "00:00:10,000",
                    "00:00:20,000",
                    "00:00:25,000",
                ],
                "tolerance": [1.0, 1.0, 1.0, 1.0],
                "gap": [0.5, 0.3, 0.4, 0.6],
                "real_dur": [4.0, 4.0, 4.0, 4.0],
                "tol_dur": [5.0, 5.0, 5.0, 5.0],
            }
        )

        mock_load_key.side_effect = lambda key: {
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.8,
        }.get(key, 1.0)
        mock_parse_time.side_effect = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
        mock_duration.return_value = 4.0

        result_df = merge_chunks(multi_chunk_df)

        # Should process both chunks
        assert len(result_df) == 4
        assert "new_sub_times" in result_df.columns


class TestGenAudio:
    """Test cases for main gen_audio function"""

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    @patch("core._10_gen_audio.pd.DataFrame.to_excel")
    def test_complete_audio_generation_flow(
        self, mock_to_excel, mock_merge, mock_generate, mock_read_excel, mock_makedirs
    ):
        """Test complete audio generation workflow"""
        # Setup mock data
        mock_df = pd.DataFrame(
            {"number": [1, 2], "lines": [["Hello"], ["World"]], "cut_off": [0, 1]}
        )
        mock_read_excel.return_value = mock_df
        mock_generate.return_value = mock_df
        mock_merge.return_value = mock_df

        gen_audio()

        # Verify the complete workflow
        mock_makedirs.assert_any_call("/tmp/audio_tmp", exist_ok=True)
        mock_makedirs.assert_any_call("/tmp/audio_segs", exist_ok=True)
        mock_read_excel.assert_called_once()
        mock_generate.assert_called_once_with(mock_df)
        mock_merge.assert_called_once_with(mock_df)
        mock_to_excel.assert_called_once()

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    def test_file_loading_error_handling(self, mock_read_excel, mock_makedirs):
        """Test error handling when task file cannot be loaded"""
        mock_read_excel.side_effect = FileNotFoundError("Task file not found")

        with pytest.raises(FileNotFoundError, match="Task file not found"):
            gen_audio()

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    def test_tts_generation_error_handling(
        self, mock_generate, mock_read_excel, mock_makedirs
    ):
        """Test error handling during TTS generation"""
        mock_read_excel.return_value = pd.DataFrame()
        mock_generate.side_effect = Exception("TTS generation failed")

        with pytest.raises(Exception, match="TTS generation failed"):
            gen_audio()

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.generate_tts_audio")
    @patch("core._10_gen_audio.merge_chunks")
    def test_chunk_merging_error_handling(
        self, mock_merge, mock_generate, mock_read_excel, mock_makedirs
    ):
        """Test error handling during chunk merging"""
        mock_df = pd.DataFrame()
        mock_read_excel.return_value = mock_df
        mock_generate.return_value = mock_df
        mock_merge.side_effect = Exception("Chunk merging failed")

        with pytest.raises(Exception, match="Chunk merging failed"):
            gen_audio()


class TestIntegrationScenarios:
    """Integration test scenarios for realistic workflows"""

    @patch("core._10_gen_audio.os.makedirs")
    @patch("core._10_gen_audio.pd.read_excel")
    @patch("core._10_gen_audio.tts_main")
    @patch("core._10_gen_audio.get_audio_duration")
    @patch("core._10_gen_audio.adjust_audio_speed")
    @patch("core._10_gen_audio.load_key")
    @patch("core._10_gen_audio.parse_df_srt_time")
    def test_realistic_audio_generation_scenario(
        self,
        mock_parse_time,
        mock_load_key,
        mock_adjust,
        mock_duration,
        mock_tts,
        mock_read_excel,
        mock_makedirs,
    ):
        """Test a realistic complete audio generation scenario"""
        # Setup realistic test data
        realistic_df = pd.DataFrame(
            {
                "number": [1, 2, 3],
                "lines": [["Hello world!"], ["How are you?"], ["Goodbye!"]],
                "cut_off": [0, 0, 1],
                "start_time": ["00:00:00,000", "00:00:05,000", "00:00:10,000"],
                "end_time": ["00:00:04,500", "00:00:09,200", "00:00:13,800"],
                "tolerance": [0.5, 0.8, 1.0],
                "gap": [0.2, 0.3, 0.4],
                "tol_dur": [5.0, 4.7, 4.8],
            }
        )

        # Configure mocks for realistic behavior
        mock_read_excel.return_value = realistic_df.copy()
        mock_load_key.side_effect = lambda key: {
            "max_workers": 2,
            "tts_method": "openai",
            "speed_factor.accept": 1.2,
            "speed_factor.min": 0.7,
        }.get(key, 1.0)
        mock_duration.side_effect = [
            4.2,
            3.8,
            4.1,
            4.0,
            3.6,
            3.9,
        ]  # TTS + adjusted durations
        mock_parse_time.side_effect = [0.0, 4.5, 5.0, 9.2, 10.0, 13.8]

        # Mock Excel save
        with patch.object(realistic_df, "to_excel") as mock_save:
            gen_audio()
            mock_save.assert_called()

        # Verify key operations were called
        assert mock_tts.call_count == 3  # One per line
        assert mock_adjust.call_count == 3  # Speed adjustment for each
        mock_makedirs.assert_any_call("/tmp/audio_tmp", exist_ok=True)

    def test_empty_dataframe_handling(self):
        """Test handling of empty task dataframes"""
        with patch("core._10_gen_audio.pd.read_excel", return_value=pd.DataFrame()):
            with patch("core._10_gen_audio.os.makedirs"):
                # Should handle empty DataFrame gracefully
                result_df = generate_tts_audio(pd.DataFrame())
                assert len(result_df) == 0
                assert "real_dur" in result_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
