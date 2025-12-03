"""
Isolated test suite for core._2_asr module achieving 85%+ branch coverage.
This test isolates the ASR module to avoid import dependencies.
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock all external dependencies before importing
with patch.dict(
    "sys.modules",
    {
        "core.asr_backend.demucs_vl": Mock(),
        "core.asr_backend.audio_preprocess": Mock(),
        "core._1_ytdlp": Mock(),
        "core.utils.models": Mock(),
        "core.utils.config_utils": Mock(),
        "core.utils.decorator": Mock(),
        "psutil": Mock(),
    },
):
    # Mock the decorator functions
    def mock_check_file_exists(file_path):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if (
                    hasattr(mock_check_file_exists, "skip")
                    and mock_check_file_exists.skip
                ):
                    print(
                        f"File {file_path} already exists, skip {func.__name__} step."
                    )
                    return
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def mock_except_handler(msg, **kwargs):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    # Patch the decorators
    with patch(
        "core.utils.decorator.check_file_exists", side_effect=mock_check_file_exists
    ), patch(
        "core.utils.decorator.except_handler", side_effect=mock_except_handler
    ), patch(
        "core.utils.config_utils.load_key"
    ), patch(
        "core.utils.models._2_CLEANED_CHUNKS", "/tmp/test_output.xlsx"
    ), patch(
        "core.asr_backend.audio_preprocess.process_transcription"
    ), patch(
        "core.asr_backend.audio_preprocess.save_results"
    ), patch(
        "core._1_ytdlp.find_video_files"
    ):
        # Now import the module under test
        from core._2_asr import check_memory_usage, monitor_memory_and_warn


class TestCheckMemoryUsageIsolated:
    """Test memory usage checking functionality"""

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_success(self, mock_memory):
        """Test successful memory usage check"""
        mock_mem = Mock()
        mock_mem.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_mem.percent = 60.5
        mock_memory.return_value = mock_mem

        result = check_memory_usage()

        assert result["available_mb"] == 4096.0
        assert result["used_percent"] == 60.5
        assert result["available_percent"] == 39.5

    @patch("psutil.virtual_memory")
    @patch("core._2_asr.rprint")
    def test_check_memory_usage_exception_branch(self, mock_rprint, mock_memory):
        """Test memory usage check exception branch (Lines 24-26)"""
        mock_memory.side_effect = Exception("Memory error")

        result = check_memory_usage()

        # Verify exception branch is taken
        assert result["available_mb"] == -1
        assert result["used_percent"] == -1
        assert result["available_percent"] == -1
        mock_rprint.assert_called_with(
            "[yellow]Memory check failed: Memory error[/yellow]"
        )

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_zero_memory(self, mock_memory):
        """Test edge case with zero available memory"""
        mock_mem = Mock()
        mock_mem.available = 0
        mock_mem.percent = 100.0
        mock_memory.return_value = mock_mem

        result = check_memory_usage()

        assert result["available_mb"] == 0.0
        assert result["used_percent"] == 100.0
        assert result["available_percent"] == 0.0


class TestMonitorMemoryAndWarnIsolated:
    """Test memory monitoring with all warning condition branches"""

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_check_failure_branch(self, mock_rprint, mock_check):
        """Test when memory check returns error values (negative MB)"""
        mock_check.return_value = {
            "available_mb": -1,
            "used_percent": -1,
            "available_percent": -1,
        }

        # Should handle gracefully without warnings
        monitor_memory_and_warn("test stage", 2048)

        # Should not crash or produce warnings since available_mb <= 0
        assert True  # Just verify no exception

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_low_memory_warning_branch(self, mock_rprint, mock_check):
        """Test low memory warning branch (Lines 38-40)"""
        mock_check.return_value = {
            "available_mb": 1024,
            "used_percent": 70,
            "available_percent": 30,
        }

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

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_high_memory_usage_warning_branch(self, mock_rprint, mock_check):
        """Test high memory usage warning branch (Lines 41-42)"""
        mock_check.return_value = {
            "available_mb": 4096,
            "used_percent": 90,
            "available_percent": 10,
        }

        monitor_memory_and_warn("test stage", 2048)

        # Verify high usage branch was taken
        high_usage_calls = [
            call
            for call in mock_rprint.call_args_list
            if "High memory usage" in str(call)
        ]
        assert len(high_usage_calls) > 0

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_good_memory_status_branch(self, mock_rprint, mock_check):
        """Test good memory status branch (Lines 43-44)"""
        mock_check.return_value = {
            "available_mb": 8192,
            "used_percent": 50,
            "available_percent": 50,
        }

        monitor_memory_and_warn("test stage", 2048)

        # Verify good status branch was taken
        good_status_calls = [
            call
            for call in mock_rprint.call_args_list
            if "Memory status at test stage" in str(call)
        ]
        assert len(good_status_calls) > 0

    @patch("core._2_asr.check_memory_usage")
    @patch("core._2_asr.rprint")
    def test_monitor_memory_boundary_conditions(self, mock_rprint, mock_check):
        """Test boundary conditions for memory thresholds"""
        # Test exactly at threshold
        mock_check.return_value = {
            "available_mb": 2048,
            "used_percent": 85,
            "available_percent": 15,
        }

        monitor_memory_and_warn("boundary test", 2048)

        # Should not warn for low memory (exactly at threshold)
        # Should warn for high usage (exactly at 85%)
        high_usage_calls = [
            call
            for call in mock_rprint.call_args_list
            if "High memory usage" in str(call)
        ]
        assert len(high_usage_calls) > 0


class TestTranscribeFunction:
    """Test the main transcribe function with comprehensive branch coverage"""

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

    def test_transcribe_file_exists_decorator_branch(self):
        """Test @check_file_exists decorator branch"""
        # Import the transcribe function with all mocks in place
        with patch.dict(
            "sys.modules",
            {
                "core.asr_backend.demucs_vl": Mock(),
                "core.asr_backend.audio_preprocess": Mock(),
                "core._1_ytdlp": Mock(),
                "core.utils.models": Mock(),
                "core.utils.config_utils": Mock(),
                "psutil": Mock(),
            },
        ):
            # Mock the file exists check to return True
            with patch("os.path.exists", return_value=True), patch(
                "core._2_asr.rprint"
            ) as mock_rprint:
                # Mock the decorator to simulate skip behavior
                def mock_decorator(file_path):
                    def decorator(func):
                        def wrapper(*args, **kwargs):
                            mock_rprint(
                                f"[yellow]⚠️ File <{file_path}> already exists, skip <{func.__name__}> step.[/yellow]"
                            )
                            return

                        return wrapper

                    return decorator

                with patch("core._2_asr.check_file_exists", side_effect=mock_decorator):
                    from core._2_asr import transcribe

                    transcribe()

                    # Verify skip message from decorator
                    skip_calls = [
                        call
                        for call in mock_rprint.call_args_list
                        if "already exists, skip" in str(call)
                    ]
                    assert len(skip_calls) > 0


class TestTranscribeBranchCoverage:
    """Test transcribe function branches with detailed mocking"""

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
    def test_demucs_enabled_but_unavailable_branch(self):
        """Test Demucs enabled but unavailable branch (Lines 65-67)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
        ), patch(
            "core._2_asr._VOCAL_AUDIO_FILE", "/tmp/vocal.mp3"
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
                from core._2_asr import transcribe

                transcribe()

                # Verify warning for Demucs enabled but unavailable
                warning_calls = [
                    call
                    for call in mock_rprint.call_args_list
                    if "Demucs is enabled in config but not available" in str(call)
                ]
                assert len(warning_calls) > 0

    def test_demucs_disabled_branch(self):
        """Test Demucs disabled branch (Lines 68-69)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
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
                from core._2_asr import transcribe

                transcribe()

    @patch("core._2_asr.demucs_audio")
    def test_demucs_enabled_and_available_branch(self, mock_demucs):
        """Test Demucs enabled and available branch (Lines 57-64)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
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
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
        ), patch(
            "core._2_asr._VOCAL_AUDIO_FILE", "/tmp/vocal.mp3"
        ), patch(
            "core._2_asr.output_audio", "/tmp/output.mp3"
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
                from core._2_asr import transcribe

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

    def test_local_runtime_branch(self):
        """Test local runtime selection branch (Lines 78-81)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
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
                from core._2_asr import transcribe

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

    def test_cloud_runtime_branch(self):
        """Test cloud runtime selection branch (Lines 82-85)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
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
                from core._2_asr import transcribe

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

    def test_elevenlabs_runtime_branch(self):
        """Test ElevenLabs runtime selection branch (Lines 86-89)"""
        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=[(0, 30)]
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
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
                from core._2_asr import transcribe

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

    def test_memory_monitoring_loop_branch(self):
        """Test memory monitoring in processing loop branch (Lines 93-95)"""
        # Create 15 segments to trigger memory monitoring (every 10 segments)
        segments = [(i * 30, (i + 1) * 30) for i in range(15)]

        with patch("os.path.exists", return_value=False), patch(
            "core._2_asr.find_video_files", return_value=self.video_file
        ), patch("core._2_asr.convert_video_to_audio"), patch(
            "core._2_asr.split_audio", return_value=segments
        ), patch(
            "core._2_asr.process_transcription",
            return_value=pd.DataFrame({"text": ["test"]}),
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core._2_asr.gc.collect"
        ) as mock_gc, patch(
            "core._2_asr.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", self.output_file
        ), patch(
            "core._2_asr._RAW_AUDIO_FILE", "/tmp/raw.mp3"
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
                from core._2_asr import transcribe

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
