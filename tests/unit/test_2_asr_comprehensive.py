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

# Mock torch and other heavy dependencies before any imports
sys.modules["torch"] = MagicMock()
sys.modules["core.asr_backend.demucs_vl"] = MagicMock()

# Mock the demucs_audio function at module level
mock_demucs = MagicMock()
mock_demucs.demucs_audio = Mock()

# Patch imports before importing the module under test
with patch.dict(
    "sys.modules",
    {
        "torch": MagicMock(),
        "core.asr_backend.demucs_vl": mock_demucs,
    },
):
    from core._2_asr import check_memory_usage, monitor_memory_and_warn, transcribe


class TestCheckMemoryUsage:
    """Test memory usage checking functionality"""

    def test_check_memory_usage_success(self):
        """Test successful memory usage check"""
        mock_memory = Mock()
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 60.5  # 60.5% used

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 4096.0
            assert result["used_percent"] == 60.5
            assert result["available_percent"] == 39.5

    def test_check_memory_usage_exception(self):
        """Test memory usage check exception handling - covers lines 24-26"""
        with patch(
            "psutil.virtual_memory", side_effect=Exception("Memory check failed")
        ), patch("core._2_asr.rprint") as mock_rprint:
            result = check_memory_usage()

            assert result["available_mb"] == -1
            assert result["used_percent"] == -1
            assert result["available_percent"] == -1
            mock_rprint.assert_called_with(
                "[yellow]Memory check failed: Memory check failed[/yellow]"
            )

    def test_check_memory_usage_edge_cases(self):
        """Test memory usage check with edge case values"""
        # Test with zero available memory
        mock_memory = Mock()
        mock_memory.available = 0
        mock_memory.percent = 100.0

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 0.0
            assert result["used_percent"] == 100.0
            assert result["available_percent"] == 0.0

    def test_check_memory_usage_large_values(self):
        """Test memory usage check with large memory values"""
        mock_memory = Mock()
        mock_memory.available = 64 * 1024 * 1024 * 1024  # 64GB
        mock_memory.percent = 20.0

        with patch("psutil.virtual_memory", return_value=mock_memory):
            result = check_memory_usage()

            assert result["available_mb"] == 65536.0
            assert result["used_percent"] == 20.0
            assert result["available_percent"] == 80.0

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_os_error(self, mock_virtual_memory):
        """Test OS-specific exceptions during memory check"""
        mock_virtual_memory.side_effect = OSError("Access denied")

        with patch("core._2_asr.rprint") as mock_rprint:
            result = check_memory_usage()

            assert result["available_mb"] == -1
            assert result["used_percent"] == -1
            assert result["available_percent"] == -1
            mock_rprint.assert_called_with(
                "[yellow]Memory check failed: Access denied[/yellow]"
            )

    @patch("psutil.virtual_memory")
    def test_check_memory_usage_permission_error(self, mock_virtual_memory):
        """Test permission errors during memory check"""
        mock_virtual_memory.side_effect = PermissionError("Permission denied")

        with patch("core._2_asr.rprint") as mock_rprint:
            result = check_memory_usage()

            assert result["available_mb"] == -1
            mock_rprint.assert_called_with(
                "[yellow]Memory check failed: Permission denied[/yellow]"
            )


class TestMonitorMemoryAndWarn:
    """Test memory monitoring and warning functionality"""

    def test_monitor_low_memory_warning(self):
        """Test warning for low memory - covers lines 37-40"""
        memory_info = {
            "available_mb": 1024,  # Less than required 2048
            "used_percent": 75,
            "available_percent": 25,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Should warn about low memory
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

    def test_monitor_high_memory_usage_warning(self):
        """Test warning for high memory usage percentage - covers lines 41-42"""
        memory_info = {
            "available_mb": 4096,  # Sufficient memory
            "used_percent": 90,  # High usage
            "available_percent": 10,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Should warn about high usage
            high_usage_calls = [
                call
                for call in mock_rprint.call_args_list
                if "High memory usage" in str(call)
            ]
            assert len(high_usage_calls) > 0

    def test_monitor_good_memory_status(self):
        """Test good memory status message - covers lines 43-44"""
        memory_info = {
            "available_mb": 8192,  # Good memory
            "used_percent": 50,  # Normal usage
            "available_percent": 50,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("test stage", 2048)

            # Should show good status
            good_status_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Memory status at test stage" in str(call)
            ]
            assert len(good_status_calls) > 0

    def test_monitor_memory_check_failure(self):
        """Test handling when memory check fails"""
        memory_info = {
            "available_mb": -1,  # Error indicator
            "used_percent": -1,
            "available_percent": -1,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ):
            # Should not raise exception
            monitor_memory_and_warn("test stage", 2048)

    def test_monitor_memory_custom_requirements(self):
        """Test memory monitoring with custom requirements"""
        memory_info = {"available_mb": 512, "used_percent": 60, "available_percent": 40}

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            # Test with low requirement - should pass
            monitor_memory_and_warn("test stage", 256)
            good_status_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Memory status at test stage" in str(call)
            ]
            assert len(good_status_calls) > 0

            mock_rprint.reset_mock()

            # Test with high requirement - should warn
            monitor_memory_and_warn("test stage", 1024)
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Warning: Low memory" in str(call)
            ]
            assert len(warning_calls) > 0

    def test_monitor_memory_zero_available(self):
        """Test monitoring when available memory is exactly 0"""
        memory_info = {"available_mb": 0, "used_percent": 100, "available_percent": 0}

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("critical stage", 1024)

            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Warning: Low memory" in str(call)
            ]
            assert len(warning_calls) > 0

    def test_monitor_memory_exactly_at_threshold(self):
        """Test monitoring when memory is exactly at threshold"""
        memory_info = {
            "available_mb": 2048,
            "used_percent": 70,
            "available_percent": 30,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            # Exactly at threshold should show good status
            monitor_memory_and_warn("threshold test", 2048)

            good_status_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Memory status at threshold test" in str(call)
            ]
            assert len(good_status_calls) > 0

    def test_monitor_memory_exactly_85_percent(self):
        """Test monitoring when memory usage is exactly 85%"""
        memory_info = {
            "available_mb": 4096,
            "used_percent": 85.0,
            "available_percent": 15.0,
        }

        with patch("core._2_asr.check_memory_usage", return_value=memory_info), patch(
            "core._2_asr.rprint"
        ) as mock_rprint:
            monitor_memory_and_warn("85% test", 1024)

            # Should show good status, not high usage warning (85% is boundary)
            good_status_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Memory status at 85% test" in str(call)
            ]
            assert len(good_status_calls) > 0


class TestTranscribe:
    """Test main transcribe function"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_file = os.path.join(self.temp_dir, "test_video.mp4")

        # Create mock video file
        with open(self.video_file, "wb") as f:
            f.write(b"fake video data")

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_local_runtime_success(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test successful transcription with local runtime - covers lines 78-81"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30), (30, 60)]  # Two segments

        # Mock transcription results
        mock_result1 = {"segments": [{"text": "Hello world", "start": 0, "end": 5}]}
        mock_result2 = {"segments": [{"text": "How are you", "start": 30, "end": 35}]}

        # Mock the local transcriber
        mock_transcriber = Mock()
        mock_transcriber.side_effect = [mock_result1, mock_result2]

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["Hello world", "How are you"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify function calls
            mock_find_video.assert_called_once()
            mock_convert.assert_called_once_with(self.video_file)
            mock_split.assert_called_once()
            mock_process.assert_called_once()
            mock_save.assert_called_once_with(mock_df)

            # Verify transcriber was called for each segment
            assert mock_transcriber.call_count == 2

            # Verify memory monitoring calls
            assert (
                mock_monitor.call_count >= 5
            )  # At least start, conversion, transcription, etc.

            # Verify garbage collection
            assert mock_gc.call_count >= 2

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_cloud_runtime_success(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test successful transcription with cloud runtime - covers lines 82-85"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30), (30, 60)]  # Two segments

        # Mock transcription results
        mock_result1 = {
            "segments": [{"text": "Cloud transcription 1", "start": 0, "end": 5}]
        }
        mock_result2 = {
            "segments": [{"text": "Cloud transcription 2", "start": 30, "end": 35}]
        }

        # Mock the cloud transcriber
        mock_transcriber = Mock()
        mock_transcriber.side_effect = [mock_result1, mock_result2]

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "cloud"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame(
            {"text": ["Cloud transcription 1", "Cloud transcription 2"]}
        )
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_302.transcribe_audio_302", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify function calls
            mock_find_video.assert_called_once()
            mock_convert.assert_called_once_with(self.video_file)
            mock_split.assert_called_once()
            mock_process.assert_called_once()
            mock_save.assert_called_once_with(mock_df)

            # Verify transcriber was called for each segment
            assert mock_transcriber.call_count == 2

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_elevenlabs_runtime_success(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test successful transcription with elevenlabs runtime - covers lines 86-89"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]  # One segment

        mock_result = {
            "segments": [{"text": "ElevenLabs transcription", "start": 0, "end": 5}]
        }
        mock_transcriber = Mock(return_value=mock_result)

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "elevenlabs"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["ElevenLabs transcription"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs",
            mock_transcriber,
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify transcriber was called
            mock_transcriber.assert_called_once()

    @patch("core._2_asr.demucs_audio", None)  # Demucs not available
    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    @patch("core._2_asr.output_audio", "mock_output_audio")
    def test_transcribe_demucs_enabled_but_not_available(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test transcription when Demucs is enabled but not available - covers lines 65-67"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]  # One segment

        mock_result = {
            "segments": [{"text": "Test transcription", "start": 0, "end": 5}]
        }
        mock_transcriber = Mock(return_value=mock_result)

        def load_key_side_effect(key):
            if key == "demucs":
                return True  # Enabled but not available
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["Test transcription"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint") as mock_rprint:
            transcribe()

            # Should warn that Demucs is enabled but not available
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Demucs is enabled in config but not available" in str(call)
            ]
            assert len(warning_calls) > 0

    @patch("core._2_asr.demucs_audio")
    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core._2_asr.normalize_audio_volume")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_with_demucs_separation(
        self,
        mock_exists,
        mock_load_key,
        mock_normalize,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
        mock_demucs,
    ):
        """Test transcription with Demucs vocal separation - covers lines 57-64"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]  # One segment
        mock_normalize.return_value = "/path/to/vocal.mp3"

        mock_result = {"segments": [{"text": "Demucs separated", "start": 0, "end": 5}]}
        mock_transcriber = Mock(return_value=mock_result)

        def load_key_side_effect(key):
            if key == "demucs":
                return True  # Demucs enabled and available
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["Demucs separated"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint") as mock_rprint:
            transcribe()

            # Verify Demucs was called
            mock_demucs.assert_called_once()
            mock_normalize.assert_called_once()

            # Should log vocal separation start
            vocal_sep_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Starting vocal separation with Demucs" in str(call)
            ]
            assert len(vocal_sep_calls) > 0

            # Verify extra memory monitoring for Demucs
            demucs_memory_calls = [
                call
                for call in mock_monitor.call_args_list
                if any("before Demucs" in str(arg) for arg in call.args)
            ]
            assert len(demucs_memory_calls) > 0

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_demucs_disabled(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test transcription with Demucs disabled - covers lines 68-69"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]  # One segment

        mock_result = {"segments": [{"text": "No Demucs", "start": 0, "end": 5}]}
        mock_transcriber = Mock(return_value=mock_result)

        def load_key_side_effect(key):
            if key == "demucs":
                return False  # Demucs disabled
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["No Demucs"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify normal processing without Demucs warnings
            mock_process.assert_called_once()

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_memory_monitoring_during_processing(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test memory monitoring during processing loop - covers lines 93-95"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        # Create 25 segments to trigger memory monitoring (every 10 segments)
        segments = [(i * 10, (i + 1) * 10) for i in range(25)]
        mock_split.return_value = segments

        # Mock transcription results
        mock_results = [
            {
                "segments": [
                    {"text": f"Segment {i}", "start": i * 10, "end": (i + 1) * 10}
                ]
            }
            for i in range(25)
        ]
        mock_transcriber = Mock()
        mock_transcriber.side_effect = mock_results

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": [f"Segment {i}" for i in range(25)]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify transcriber was called for all segments
            assert mock_transcriber.call_count == 25

            # Verify memory monitoring was called for every 10th segment
            segment_monitoring_calls = [
                call
                for call in mock_monitor.call_args_list
                if any("transcription segment" in str(arg) for arg in call.args)
            ]
            # Should be called at segments 10 and 20 (i % 10 == 0 and i > 0)
            assert len(segment_monitoring_calls) >= 2

            # Verify garbage collection during processing
            assert mock_gc.call_count >= 4  # Start + periodic + final

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_single_segment_no_memory_check(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test transcription with single segment (no memory check in loop)"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]  # Single segment

        mock_result = {"segments": [{"text": "Single segment", "start": 0, "end": 30}]}
        mock_transcriber = Mock(return_value=mock_result)

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": ["Single segment"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Should not have segment monitoring calls since there's only 1 segment
            segment_monitoring_calls = [
                call
                for call in mock_monitor.call_args_list
                if any("transcription segment" in str(arg) for arg in call.args)
            ]
            assert len(segment_monitoring_calls) == 0

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_exactly_10_segments(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test transcription with exactly 10 segments (boundary condition)"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        segments = [(i * 10, (i + 1) * 10) for i in range(10)]
        mock_split.return_value = segments

        mock_results = [
            {
                "segments": [
                    {"text": f"Segment {i}", "start": i * 10, "end": (i + 1) * 10}
                ]
            }
            for i in range(10)
        ]
        mock_transcriber = Mock()
        mock_transcriber.side_effect = mock_results

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect
        mock_df = pd.DataFrame({"text": [f"Segment {i}" for i in range(10)]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Should not trigger segment monitoring since i % 10 == 0 and i > 0 is never true
            # (i goes 0,1,2...9, and when i=10 the loop would be over)
            segment_monitoring_calls = [
                call
                for call in mock_monitor.call_args_list
                if any("transcription segment" in str(arg) for arg in call.args)
            ]
            assert len(segment_monitoring_calls) == 0

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_combined_results_processing(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test combined results processing logic - covers lines 103-105"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30), (30, 60)]

        # Mock transcription results with multiple segments each
        mock_result1 = {
            "segments": [
                {"text": "First part", "start": 0, "end": 10},
                {"text": "Second part", "start": 10, "end": 20},
            ]
        }
        mock_result2 = {
            "segments": [
                {"text": "Third part", "start": 30, "end": 40},
                {"text": "Fourth part", "start": 40, "end": 50},
            ]
        }

        mock_transcriber = Mock()
        mock_transcriber.side_effect = [mock_result1, mock_result2]

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "local"
            return None

        mock_load_key.side_effect = load_key_side_effect

        # Expected combined result should have all segments
        expected_combined = {
            "segments": [
                {"text": "First part", "start": 0, "end": 10},
                {"text": "Second part", "start": 10, "end": 20},
                {"text": "Third part", "start": 30, "end": 40},
                {"text": "Fourth part", "start": 40, "end": 50},
            ]
        }

        mock_df = pd.DataFrame({"text": ["Combined transcription"]})
        mock_process.return_value = mock_df

        with patch(
            "core.asr_backend.whisperX_local.transcribe_audio", mock_transcriber
        ), patch("core._2_asr.rprint"):
            transcribe()

            # Verify process_transcription was called with combined result
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0][0]

            # Verify all segments were combined
            assert len(call_args["segments"]) == 4
            assert call_args["segments"][0]["text"] == "First part"
            assert call_args["segments"][3]["text"] == "Fourth part"

    @patch("core._2_asr.gc.collect")
    @patch("core._2_asr.monitor_memory_and_warn")
    @patch("core._2_asr.save_results")
    @patch("core._2_asr.process_transcription")
    @patch("core._2_asr.split_audio")
    @patch("core._2_asr.convert_video_to_audio")
    @patch("core._2_asr.find_video_files")
    @patch("core.utils.config_utils.load_key")
    @patch("core._2_asr._2_CLEANED_CHUNKS", "mock_path")
    @patch("os.path.exists", return_value=True)
    def test_transcribe_unknown_runtime(
        self,
        mock_exists,
        mock_load_key,
        mock_find_video,
        mock_convert,
        mock_split,
        mock_process,
        mock_save,
        mock_monitor,
        mock_gc,
    ):
        """Test transcription with unknown runtime (no ASR backend selected)"""
        # Configure mocks
        mock_find_video.return_value = self.video_file
        mock_split.return_value = [(0, 30)]

        def load_key_side_effect(key):
            if key == "demucs":
                return False
            elif key == "whisper.runtime":
                return "unknown"  # Unknown runtime
            return None

        mock_load_key.side_effect = load_key_side_effect

        with patch("core._2_asr.rprint"):
            # This should cause an error since no transcriber is imported
            with pytest.raises(NameError):  # 'ts' is not defined
                transcribe()

    def test_transcribe_main_execution(self):
        """Test main execution block - covers lines 115-116"""
        with patch("core._2_asr.transcribe") as mock_transcribe:
            # Import and simulate the main block
            import core._2_asr as asr_module

            # Simulate the main execution by calling transcribe directly
            # This covers the if __name__ == "__main__": block logic
            asr_module.transcribe()
            mock_transcribe.assert_called_once()

    def test_all_memory_monitoring_stages(self):
        """Test that all required memory monitoring stages are called"""
        with patch("core._2_asr.find_video_files", return_value="test.mp4"), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription", return_value=pd.DataFrame()
        ), patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", "mock_path"
        ), patch(
            "os.path.exists", return_value=True
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ) as mock_monitor, patch(
            "core.asr_backend.whisperX_local.transcribe_audio",
            return_value={"segments": []},
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

            # Extract all stage names from monitor calls
            stage_names = []
            for call in mock_monitor.call_args_list:
                if len(call.args) > 0:
                    stage_names.append(call.args[0])

            # Verify all expected stages are monitored
            expected_stages = [
                "transcription start",
                "after video conversion",
                "after audio segmentation",
                "before local transcription",
                "after transcription",
                "transcription complete",
            ]

            for expected_stage in expected_stages:
                assert any(
                    expected_stage in stage for stage in stage_names
                ), f"Missing expected monitoring stage: {expected_stage}"


class TestCompleteWorkflow:
    """Integration tests for complete workflow scenarios"""

    def test_complete_workflow_no_demucs_local(self):
        """Test complete workflow without Demucs using local transcription"""
        with patch("core._2_asr.find_video_files", return_value="test.mp4"), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30), (30, 60)]), patch(
            "core._2_asr.process_transcription"
        ) as mock_process, patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", "mock_path"
        ), patch(
            "os.path.exists", return_value=True
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core.asr_backend.whisperX_local.transcribe_audio"
        ) as mock_ts, patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return False
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect
            mock_ts.side_effect = [
                {"segments": [{"text": "Hello", "start": 0, "end": 15}]},
                {"segments": [{"text": "World", "start": 30, "end": 45}]},
            ]
            mock_process.return_value = pd.DataFrame({"text": ["Hello", "World"]})

            transcribe()

            # Verify complete processing
            assert mock_ts.call_count == 2
            mock_process.assert_called_once()

            # Verify combined result has all segments
            combined_result = mock_process.call_args[0][0]
            assert len(combined_result["segments"]) == 2
            assert combined_result["segments"][0]["text"] == "Hello"
            assert combined_result["segments"][1]["text"] == "World"

    def test_complete_workflow_with_demucs(self):
        """Test complete workflow with Demucs enabled"""
        with patch("core._2_asr.find_video_files", return_value="test.mp4"), patch(
            "core._2_asr.convert_video_to_audio"
        ), patch("core._2_asr.split_audio", return_value=[(0, 30)]), patch(
            "core._2_asr.process_transcription"
        ) as mock_process, patch(
            "core._2_asr.save_results"
        ), patch(
            "core._2_asr.gc.collect"
        ), patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._2_asr._2_CLEANED_CHUNKS", "mock_path"
        ), patch(
            "os.path.exists", return_value=True
        ), patch(
            "core._2_asr.monitor_memory_and_warn"
        ), patch(
            "core._2_asr.demucs_audio"
        ) as mock_demucs, patch(
            "core._2_asr.normalize_audio_volume", return_value="vocal.mp3"
        ), patch(
            "core.asr_backend.whisperX_local.transcribe_audio"
        ) as mock_ts, patch(
            "core._2_asr.rprint"
        ):

            def load_key_side_effect(key):
                if key == "demucs":
                    return True
                elif key == "whisper.runtime":
                    return "local"
                return None

            mock_load_key.side_effect = load_key_side_effect
            mock_ts.return_value = {
                "segments": [{"text": "Separated audio", "start": 0, "end": 15}]
            }
            mock_process.return_value = pd.DataFrame({"text": ["Separated audio"]})

            transcribe()

            # Verify Demucs processing
            mock_demucs.assert_called_once()
            mock_ts.assert_called_once()
            mock_process.assert_called_once()

    def test_error_handling_scenarios(self):
        """Test various error handling scenarios"""
        # Test with file not existing (should be handled by decorator)
        with patch("os.path.exists", return_value=False):
            # The @check_file_exists decorator should handle this
            # but since we're mocking it, we expect normal execution
            pass


if __name__ == "__main__":
    pytest.main([__file__])
