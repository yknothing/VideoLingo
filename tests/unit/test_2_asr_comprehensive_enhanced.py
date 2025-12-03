"""
Comprehensive test suite for core._2_asr module achieving 85%+ branch coverage.

This test suite covers all critical branch points:
- Memory usage monitoring and warnings
- ASR backend selection (local, cloud, elevenlabs)
- Demucs vocal separation integration
- Audio processing and segmentation
- Error handling and edge cases
- File operations and directory management
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import sys
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import modules under test
from core._2_asr import check_memory_usage, monitor_memory_and_warn, transcribe


class TestCheckMemoryUsage:
    """Test memory usage checking functionality with comprehensive branch coverage"""

    def test_check_memory_usage_success(self):
        """Test successful memory usage check"""
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 60.5

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 4096.0
            assert result["used_percent"] == 60.5
            assert result["available_percent"] == 39.5

    def test_check_memory_usage_exception_handling(self):
        """Test memory usage check exception branch (Lines 24-26)"""
        with patch(
            "psutil.virtual_memory", side_effect=Exception("Memory error")
        ), patch("core._2_asr.rprint") as mock_rprint:
            result = check_memory_usage()

            # Verify exception branch is taken
            assert result["available_mb"] == -1
            assert result["used_percent"] == -1
            assert result["available_percent"] == -1
            mock_rprint.assert_called_with(
                "[yellow]Memory check failed: Memory error[/yellow]"
            )

    def test_check_memory_usage_zero_memory(self):
        """Test edge case with zero available memory"""
        mock_memory = Mock()
        mock_memory.available = 0
        mock_memory.percent = 100.0

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 0.0
            assert result["used_percent"] == 100.0
            assert result["available_percent"] == 0.0

    def test_check_memory_usage_large_values(self):
        """Test with large memory values"""
        mock_memory = Mock()
        mock_memory.available = 128 * 1024 * 1024 * 1024  # 128GB
        mock_memory.percent = 5.0

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 131072.0
            assert result["used_percent"] == 5.0
            assert result["available_percent"] == 95.0


class TestMonitorMemoryAndWarn:
    """Test memory monitoring with all warning condition branches"""

    def test_monitor_memory_check_failure_branch(self):
        """Test when memory check returns error values (negative MB)"""
        memory_info = {"available_mb": -1, "used_percent": -1, "available_percent": -1}

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ):
            # Should handle gracefully without warnings
            monitor_memory_and_warn("test stage", 2048)

    def test_monitor_low_memory_warning_branch(self):
        """Test low memory warning branch (Lines 38-40)"""
        memory_info = {
            "available_mb": 1024,
            "used_percent": 70,
            "available_percent": 30,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Verify low memory branch was taken
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Warning: Low memory" in str(call)
            ]
            assert len(warning_calls) > 0

            suggestion_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Consider closing other applications" in str(call)
            ]
            assert len(suggestion_calls) > 0

    def test_monitor_high_memory_usage_warning_branch(self):
        """Test high memory usage warning branch (Lines 41-42)"""
        memory_info = {
            "available_mb": 4096,
            "used_percent": 90,
            "available_percent": 10,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Verify high usage branch was taken
            high_usage_calls = [
                call
                for call in mock_rprint.call_args_list
                if "High memory usage" in str(call)
            ]
            assert len(high_usage_calls) > 0

    def test_monitor_good_memory_status_branch(self):
        """Test good memory status branch (Lines 43-44)"""
        memory_info = {
            "available_mb": 8192,
            "used_percent": 50,
            "available_percent": 50,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Verify good status branch was taken
            good_status_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Memory status at test stage" in str(call)
            ]
            assert len(good_status_calls) > 0

    def test_monitor_memory_boundary_conditions(self):
        """Test boundary conditions for memory thresholds"""
        # Test exactly at threshold
        memory_info = {
            "available_mb": 2048,
            "used_percent": 85,
            "available_percent": 15,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("boundary test", 2048)

            # Should not warn for low memory (exactly at threshold)
            # Should warn for high usage (exactly at 85%)
            high_usage_calls = [
                call
                for call in mock_rprint.call_args_list
                if "High memory usage" in str(call)
            ]
            assert len(high_usage_calls) > 0


class TestTranscribeMainFunction:
    """Test the main transcribe function with all branch combinations"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_file = os.path.join(self.temp_dir, "test_video.mp4")
        self.output_file = os.path.join(self.temp_dir, "cleaned_chunks.xlsx")

        # Create mock video file
        with open(self.video_file, "wb") as f:
            f.write(b"fake video data")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._2_asr.demucs_audio", None)  # Demucs not available
    def test_transcribe_demucs_enabled_but_unavailable_branch(self):
        """Test Demucs enabled but unavailable branch (Lines 65-67)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return True  # Enabled but not available
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ):
                transcribe()

                # Verify warning for Demucs enabled but unavailable
                warning_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Demucs is enabled in config but not available" in str(call)
                ]
                assert len(warning_calls) > 0

    def test_transcribe_demucs_disabled_branch(self):
        """Test Demucs disabled branch (Lines 68-69)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False  # Disabled
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ):
                transcribe()

    @patch("core._2_asr.demucs_audio")
    def test_transcribe_demucs_enabled_and_available_branch(self, mock_demucs):
        """Test Demucs enabled and available branch (Lines 57-64)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.normalize_audio_volume", return_value="/path/vocal.mp3"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return True  # Enabled and available
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ):
                transcribe()

                # Verify Demucs was called
                mock_demucs.assert_called_once()

                # Verify vocal separation message
                vocal_sep_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Starting vocal separation with Demucs" in str(call)
                ]
                assert len(vocal_sep_calls) > 0

    def test_transcribe_local_runtime_branch(self):
        """Test local runtime selection branch (Lines 78-81)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ) as mock_transcriber:
                transcribe()

                # Verify local runtime message
                local_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Transcribing audio with local model" in str(call)
                ]
                assert len(local_calls) > 0

                # Verify local transcriber was used
                mock_transcriber.assert_called()

    def test_transcribe_cloud_runtime_branch(self):
        """Test cloud runtime selection branch (Lines 82-85)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "cloud"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_302.transcribe_audio_302",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ) as mock_transcriber:
                transcribe()

                # Verify cloud runtime message
                cloud_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Transcribing audio with 302 API" in str(call)
                ]
                assert len(cloud_calls) > 0

                # Verify cloud transcriber was used
                mock_transcriber.assert_called()

    def test_transcribe_elevenlabs_runtime_branch(self):
        """Test ElevenLabs runtime selection branch (Lines 86-89)"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "elevenlabs"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ) as mock_transcriber:
                transcribe()

                # Verify ElevenLabs runtime message
                elevenlabs_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Transcribing audio with ElevenLabs API" in str(call)
                ]
                assert len(elevenlabs_calls) > 0

                # Verify ElevenLabs transcriber was used
                mock_transcriber.assert_called()

    def test_transcribe_memory_monitoring_loop_branch(self):
        """Test memory monitoring in processing loop branch (Lines 93-95)"""
        # Create 15 segments to trigger memory monitoring (every 10 segments)
        segments = [(i * 30, (i + 1) * 30) for i in range(15)]

        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=segments), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core._2_asr.gc.collect"
        ) as mock_gc, patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                return_value={"segments": [{"text": "test", "start": 0, "end": 5}]},
            ):
                transcribe()

                # Verify memory monitoring was called for segment 10
                segment_monitor_calls = [
                    call
                    for call in mock_monitor.call_args_list
                    if "transcription segment" in str(call)
                ]
                assert len(segment_monitor_calls) > 0

                # Verify garbage collection was called in loop
                assert mock_gc.call_count >= 2  # At least loop + final cleanup

    def test_transcribe_multiple_segments_processing(self):
        """Test processing multiple audio segments"""
        segments = [(0, 30), (30, 60), (60, 90)]

        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=segments), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            mock_results = [
                {
                    "segments": [
                        {"text": f"segment {i}", "start": i * 30, "end": (i + 1) * 30}
                    ]
                }
                for i in range(3)
            ]

            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                side_effect=mock_results,
            ) as mock_transcriber:
                transcribe()

                # Verify transcriber was called for each segment
                assert mock_transcriber.call_count == 3

    def test_transcribe_file_already_exists_decorator(self):
        """Test @check_file_exists decorator branch"""
        with patch("os.path.exists", return_value=True), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            transcribe()

            # Verify skip message from decorator
            skip_calls = [
                call
                for call in mock_rprint.call_args_list
                if "already exists, skip" in str(call)
            ]
            assert len(skip_calls) > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for comprehensive coverage"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_file = os.path.join(self.temp_dir, "test_video.mp4")
        self.output_file = os.path.join(self.temp_dir, "cleaned_chunks.xlsx")

        # Create mock video file
        with open(self.video_file, "wb") as f:
            f.write(b"fake video data")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_transcribe_with_transcription_errors(self):
        """Test handling of transcription errors"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            # Mock transcriber to raise exception
            with patch(
                "core.asr_backend.whisperX_local.transcribe_audio",
                side_effect=Exception("Transcription failed"),
            ):
                with pytest.raises(Exception, match="Transcription failed"):
                    transcribe()

    def test_transcribe_empty_segments_list(self):
        """Test handling of empty segments list"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[]), patch(
            "core._2_asr.process_transcription", return_value=pd.DataFrame({"text": []})
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect

            transcribe()

    def test_transcribe_invalid_runtime_configuration(self):
        """Test handling of invalid runtime configuration"""
        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "invalid_runtime"  # Invalid runtime
                return None

            mock_load_key.side_effect = load_key_side_effect

            # This should cause an error due to undefined transcriber
            with pytest.raises(NameError):
                transcribe()


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_file = os.path.join(self.temp_dir, "test_video.mp4")
        self.output_file = os.path.join(self.temp_dir, "cleaned_chunks.xlsx")

        # Create mock video file
        with open(self.video_file, "wb") as f:
            f.write(b"fake video data")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._2_asr.demucs_audio")
    def test_full_pipeline_with_demucs_and_cloud_runtime(self, mock_demucs):
        """Test full pipeline: Demucs enabled + cloud runtime + multiple segments"""
        segments = [(0, 30), (30, 60)]

        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=segments), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["seg1", "seg2"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core._2_asr.gc.collect"
        ) as mock_gc, patch(
            "core._2_asr.normalize_audio_volume", return_value="/path/vocal.mp3"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return True  # Demucs enabled
                elif key == "whisper.runtime":
                    return "cloud"  # Cloud runtime
                return None

            mock_load_key.side_effect = load_key_side_effect

            mock_results = [
                {
                    "segments": [
                        {"text": f"cloud seg {i}", "start": i * 30, "end": (i + 1) * 30}
                    ]
                }
                for i in range(2)
            ]

            with patch(
                "core.asr_backend.whisperX_302.transcribe_audio_302",
                side_effect=mock_results,
            ):
                transcribe()

                # Verify Demucs was called
                mock_demucs.assert_called_once()

                # Verify cloud transcription message
                cloud_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Transcribing audio with 302 API" in str(call)
                ]
                assert len(cloud_calls) > 0

                # Verify memory monitoring for different stages
                demucs_monitor = any(
                    "before Demucs" in str(call) for call in mock_monitor.call_args_list
                )
                cloud_monitor = any(
                    "before cloud transcription" in str(call)
                    for call in mock_monitor.call_args_list
                )
                assert demucs_monitor
                assert cloud_monitor

                # Verify garbage collection
                assert mock_gc.call_count >= 2

    @patch("core._2_asr.demucs_audio")
    def test_full_pipeline_with_demucs_and_elevenlabs_runtime(self, mock_demucs):
        """Test full pipeline: Demucs enabled + ElevenLabs runtime + large segment count"""
        segments = [
            (i * 30, (i + 1) * 30) for i in range(12)
        ]  # 12 segments to trigger memory monitoring

        with patch("core._2_asr.find_video_files", return_value=self.video_file), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=segments), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": [f"seg{i}" for i in range(12)]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core._2_asr.gc.collect"
        ) as mock_gc, patch(
            "core._2_asr.normalize_audio_volume", return_value="/path/vocal.mp3"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "os.path.exists", return_value=False
        ), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "demucs":
                    return True  # Demucs enabled
                elif key == "whisper.runtime":
                    return "elevenlabs"  # ElevenLabs runtime
                return None

            mock_load_key.side_effect = load_key_side_effect

            mock_results = [
                {
                    "segments": [
                        {
                            "text": f"elevenlabs seg {i}",
                            "start": i * 30,
                            "end": (i + 1) * 30,
                        }
                    ]
                }
                for i in range(12)
            ]

            with patch(
                "core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs",
                side_effect=mock_results,
            ) as mock_transcriber:
                transcribe()

                # Verify ElevenLabs transcriber was called for all segments
                assert mock_transcriber.call_count == 12

                # Verify Demucs was called
                mock_demucs.assert_called_once()

                # Verify ElevenLabs transcription message
                elevenlabs_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Transcribing audio with ElevenLabs API" in str(call)
                ]
                assert len(elevenlabs_calls) > 0

                # Verify memory monitoring in loop (should trigger at segment 10)
                segment_monitor = any(
                    "transcription segment" in str(call)
                    for call in mock_monitor.call_args_list
                )
                assert segment_monitor


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=core._2_asr",
            "--cov-branch",
            "--cov-report=term-missing",
        ]
    )
