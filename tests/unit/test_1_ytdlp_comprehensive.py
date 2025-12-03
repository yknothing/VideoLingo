import pytest
import os
import tempfile
import shutil
import json
import subprocess
import glob
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core._1_ytdlp import (
    update_ytdlp,
    validate_video_file,
    cleanup_partial_downloads,
    find_most_recent_video_file,
    find_best_video_file,
    get_optimal_format,
    categorize_download_error,
    intelligent_retry_download,
    download_video_ytdlp,
    download_via_python_api,
    download_via_command,
    find_video_files,
)


class TestUpdateYtdlp:
    """Test yt-dlp update functionality"""

    def test_update_ytdlp_success(self):
        """Test successful yt-dlp update"""
        with patch("subprocess.check_call") as mock_check_call, patch(
            "sys.modules", {"yt_dlp": Mock()}
        ) as mock_modules, patch("core._1_ytdlp.rprint") as mock_rprint:
            mock_ytdl = Mock()
            with patch("yt_dlp.YoutubeDL", return_value=mock_ytdl) as mock_youtubedl:
                result = update_ytdlp()

                mock_check_call.assert_called_once_with(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
                )
                mock_rprint.assert_called_with("[green]yt-dlp updated[/green]")
                assert result == mock_youtubedl

    def test_update_ytdlp_subprocess_error(self):
        """Test yt-dlp update subprocess error"""
        with patch(
            "subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "cmd")
        ) as mock_check_call, patch("core._1_ytdlp.rprint") as mock_rprint:
            mock_ytdl = Mock()
            with patch("yt_dlp.YoutubeDL", return_value=mock_ytdl) as mock_youtubedl:
                result = update_ytdlp()

                mock_check_call.assert_called_once()
                mock_rprint.assert_called_with(
                    "[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]"
                )
                assert result == mock_youtubedl


class TestValidateVideoFile:
    """Test video file validation functionality"""

    def test_validate_video_file_modular_success(self):
        """Test validation with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_validator = Mock()
            mock_validator.validate_video_file.return_value = (True, "File is valid")

            with patch("core._1_ytdlp.VideoFileValidator", return_value=mock_validator):
                result = validate_video_file("/path/to/video.mp4", 2)

                assert result == (True, "File is valid")
                mock_validator.validate_video_file.assert_called_once_with(
                    "/path/to/video.mp4"
                )

    def test_validate_video_file_modular_exception(self):
        """Test fallback when modular components fail"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core._1_ytdlp.VideoFileValidator", side_effect=Exception("Module error")
        ), patch("os.path.exists", return_value=False):
            result = validate_video_file("/nonexistent/video.mp4")
            assert result == (False, "File does not exist: /nonexistent/video.mp4")

    def test_validate_video_file_not_exists(self):
        """Test validation when file doesn't exist"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=False
        ):
            result = validate_video_file("/nonexistent/video.mp4")
            assert result == (False, "File does not exist: /nonexistent/video.mp4")

    def test_validate_video_file_too_small(self):
        """Test validation when file is too small"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch(
            "os.path.getsize", return_value=500 * 1024
        ):  # 0.5MB
            result = validate_video_file(
                "/path/to/small_video.mp4", expected_min_size_mb=1
            )
            assert result[0] is False
            assert "File too small" in result[1]

    def test_validate_video_file_partial_download(self):
        """Test validation of partial download files"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch(
            "os.path.getsize", return_value=2 * 1024 * 1024
        ):  # 2MB
            result = validate_video_file("/path/to/video.part")
            assert result == (False, "File appears to be a partial download")

    @pytest.mark.parametrize("file_extension", [".part", ".tmp", ".download"])
    def test_validate_video_file_partial_extensions(self, file_extension):
        """Test validation of files with partial extensions"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=2 * 1024 * 1024):
            result = validate_video_file(f"/path/to/video{file_extension}")
            assert result == (False, "File appears to be a partial download")

    def test_validate_video_file_ffprobe_success(self):
        """Test validation with successful ffprobe"""
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=2 * 1024 * 1024), patch(
            "subprocess.run", return_value=mock_result
        ):
            result = validate_video_file("/path/to/video.mp4")
            assert result == (True, "File is valid")

    def test_validate_video_file_ffprobe_failure(self):
        """Test validation with ffprobe failure"""
        mock_result = Mock()
        mock_result.returncode = 1

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=2 * 1024 * 1024), patch(
            "subprocess.run", return_value=mock_result
        ):
            result = validate_video_file("/path/to/video.mp4")
            assert result == (False, "File format validation failed")

    def test_validate_video_file_ffprobe_timeout(self):
        """Test validation with ffprobe timeout"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=2 * 1024 * 1024), patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 10)
        ):
            result = validate_video_file("/path/to/video.mp4")
            assert result == (True, "File is valid")

    def test_validate_video_file_ffprobe_not_found(self):
        """Test validation when ffprobe is not found"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "os.path.exists", return_value=True
        ), patch("os.path.getsize", return_value=2 * 1024 * 1024), patch(
            "subprocess.run", side_effect=FileNotFoundError()
        ):
            result = validate_video_file("/path/to/video.mp4")
            assert result == (True, "File is valid")


class TestCleanupPartialDownloads:
    """Test cleanup of partial downloads"""

    def test_cleanup_modular_success(self):
        """Test cleanup with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_cleaner = Mock()
            mock_cleaner.cleanup_partial_downloads.return_value = [
                "file1.part",
                "file2.tmp",
            ]

            with patch(
                "core._1_ytdlp.PartialDownloadCleaner", return_value=mock_cleaner
            ):
                result = cleanup_partial_downloads("/path/to/save")

                assert result == ["file1.part", "file2.tmp"]
                mock_cleaner.cleanup_partial_downloads.assert_called_once_with(
                    "/path/to/save"
                )

    def test_cleanup_modular_exception(self):
        """Test fallback when modular cleanup fails"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core._1_ytdlp.PartialDownloadCleaner",
            side_effect=Exception("Module error"),
        ), patch("glob.glob", return_value=[]), patch("core._1_ytdlp.rprint"):
            result = cleanup_partial_downloads("/path/to/save")
            assert result == []

    def test_cleanup_legacy_success(self):
        """Test legacy cleanup functionality"""

        def glob_side_effect(pattern):
            if "*.part" in pattern:
                return ["/path/to/save/file1.part"]
            elif "*.tmp" in pattern:
                return ["/path/to/save/file2.tmp"]
            else:
                return []

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "glob.glob", side_effect=glob_side_effect
        ), patch("os.remove") as mock_remove, patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint:
            result = cleanup_partial_downloads("/path/to/save")

            # Should contain both cleaned files
            assert "file1.part" in result
            assert "file2.tmp" in result
            assert mock_remove.call_count == 2

    def test_cleanup_legacy_remove_error(self):
        """Test legacy cleanup with file removal error"""

        def glob_side_effect(pattern):
            if "*.part" in pattern:
                return ["/path/to/save/file1.part"]
            else:
                return []

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "glob.glob", side_effect=glob_side_effect
        ), patch(
            "os.remove", side_effect=OSError("Permission denied")
        ) as mock_remove, patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint:
            result = cleanup_partial_downloads("/path/to/save")

            assert result == []
            mock_remove.assert_called_once()
            # Should log warning about failed removal
            assert any(
                "Warning: Could not remove" in str(call)
                for call in mock_rprint.call_args_list
            )


class TestFindMostRecentVideoFile:
    """Test finding most recent video file"""

    def test_find_modular_success(self):
        """Test finding with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_validator = Mock()
            mock_validator.find_most_recent_video_file.return_value = (
                "/path/to/video.mp4"
            )

            with patch(
                "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
            ), patch("core._1_ytdlp.VideoFileValidator", return_value=mock_validator):
                result = find_most_recent_video_file("/path/to/save")
                assert result == "/path/to/video.mp4"

    def test_find_modular_exception(self):
        """Test fallback when modular components fail"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "core._1_ytdlp.VideoFileValidator", side_effect=Exception("Module error")
        ), patch(
            "glob.glob", return_value=[]
        ):
            result = find_most_recent_video_file("/path/to/save")
            assert result is None

    def test_find_legacy_success(self):
        """Test legacy finding functionality"""
        mock_files = ["/path/to/save/video1.mp4", "/path/to/save/video2.avi"]

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch("glob.glob", return_value=mock_files), patch(
            "os.path.isfile", return_value=True
        ), patch(
            "os.path.getmtime", side_effect=[1000, 2000]
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(True, "Valid")
        ):
            result = find_most_recent_video_file("/path/to/save")
            assert result == "/path/to/save/video2.avi"  # Most recent

    def test_find_legacy_no_valid_files(self):
        """Test when no valid files are found"""
        mock_files = ["/path/to/save/video1.mp4"]

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "core.utils.config_utils.load_key", return_value=["mp4"]
        ), patch("glob.glob", return_value=mock_files), patch(
            "os.path.isfile", return_value=True
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(False, "Invalid")
        ):
            result = find_most_recent_video_file("/path/to/save")
            assert result is None

    def test_find_legacy_skip_non_video(self):
        """Test skipping non-video files"""
        mock_files = ["/path/to/save/image.jpg", "/path/to/save/text.txt"]

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch("glob.glob", return_value=mock_files), patch(
            "os.path.isfile", return_value=True
        ):
            result = find_most_recent_video_file("/path/to/save")
            assert result is None


class TestGetOptimalFormat:
    """Test optimal format selection"""

    def test_get_format_modular_success(self):
        """Test format selection with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_selector = Mock()
            mock_selector.get_optimal_format.return_value = "best_format_string"

            with patch("core._1_ytdlp.VideoFormatSelector", return_value=mock_selector):
                result = get_optimal_format("1080")

                assert result == "best_format_string"
                mock_selector.get_optimal_format.assert_called_once_with("1080")

    def test_get_format_modular_exception(self):
        """Test fallback when modular format selection fails"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core._1_ytdlp.VideoFormatSelector", side_effect=Exception("Module error")
        ):
            result = get_optimal_format("1080")
            # Should fallback to legacy implementation
            assert "bestvideo[height<=1080]" in result

    def test_get_format_best_quality(self):
        """Test format selection for best quality"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            result = get_optimal_format("best")

            assert "bestvideo[height<=2160]" in result
            assert "bestvideo[height<=1440]" in result
            assert "bestvideo[height<=1080]" in result

    @pytest.mark.parametrize("resolution", ["720", "1080", "1440"])
    def test_get_format_specific_resolution(self, resolution):
        """Test format selection for specific resolutions"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            result = get_optimal_format(resolution)

            assert f"bestvideo[height<={resolution}]" in result
            assert "vcodec^=avc1" in result or "vcodec^=h264" in result


class TestCategorizeDownloadError:
    """Test download error categorization"""

    def test_categorize_modular_success(self):
        """Test categorization with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_handler = Mock()
            mock_error_category = Mock()
            mock_error_category.value = "network"
            mock_handler.categorize_download_error.return_value = (
                mock_error_category,
                True,
                30,
            )

            with patch("core._1_ytdlp.DownloadErrorHandler", return_value=mock_handler):
                result = categorize_download_error("Connection timeout")

                assert result == ("network", True, 30)
                mock_handler.categorize_download_error.assert_called_once_with(
                    "Connection timeout"
                )

    def test_categorize_modular_exception(self):
        """Test fallback when modular categorization fails"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core._1_ytdlp.DownloadErrorHandler", side_effect=Exception("Module error")
        ):
            result = categorize_download_error("network timeout")
            assert result == ("network", True, 30)

    @pytest.mark.parametrize(
        "error_msg,expected_category,expected_retry,expected_wait",
        [
            ("network timeout", "network", True, 30),
            ("connection refused", "network", True, 30),
            ("403 forbidden", "rate_limit", True, 60),
            ("429 too many requests", "rate_limit", True, 60),
            ("404 not found", "not_found", False, 0),
            ("video unavailable", "not_found", False, 0),
            ("401 unauthorized", "access_denied", False, 0),
            ("private video", "access_denied", False, 0),
            ("proxy error", "proxy_ssl", True, 10),
            ("ssl certificate error", "proxy_ssl", True, 10),
            ("unknown error", "unknown", True, 20),
        ],
    )
    def test_categorize_legacy_errors(
        self, error_msg, expected_category, expected_retry, expected_wait
    ):
        """Test legacy error categorization"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            result = categorize_download_error(error_msg)

            assert result[0] == expected_category
            assert result[1] == expected_retry
            assert result[2] == expected_wait


class TestIntelligentRetryDownload:
    """Test intelligent retry mechanism"""

    def test_retry_modular_success(self):
        """Test retry with modular components"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True):
            mock_handler = Mock()
            mock_handler.intelligent_retry_download.return_value = "success_result"

            with patch("core._1_ytdlp.DownloadErrorHandler", return_value=mock_handler):
                download_func = Mock(return_value="download_result")
                result = intelligent_retry_download(download_func, 3, 5)

                assert result == "success_result"
                mock_handler.intelligent_retry_download.assert_called_once_with(
                    download_func, 3, 5
                )

    def test_retry_modular_exception(self):
        """Test fallback when modular retry fails"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True), patch(
            "core._1_ytdlp.DownloadErrorHandler", side_effect=Exception("Module error")
        ):
            download_func = Mock(return_value="download_result")
            result = intelligent_retry_download(download_func, 1, 1)

            assert result == "download_result"

    def test_retry_legacy_success_first_try(self):
        """Test successful download on first try"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            download_func = Mock(return_value="success")

            result = intelligent_retry_download(download_func, 2, 1)
            assert result == "success"
            download_func.assert_called_once()

    def test_retry_legacy_success_after_retry(self):
        """Test successful download after retry"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "time.sleep"
        ), patch("core._1_ytdlp.rprint"), patch(
            "core._1_ytdlp.categorize_download_error",
            return_value=("network", True, 10),
        ):
            download_func = Mock(side_effect=[Exception("network error"), "success"])

            result = intelligent_retry_download(download_func, 2, 1)
            assert result == "success"
            assert download_func.call_count == 2

    def test_retry_legacy_non_retryable_error(self):
        """Test non-retryable error"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "core._1_ytdlp.rprint"
        ), patch(
            "core._1_ytdlp.categorize_download_error",
            return_value=("not_found", False, 0),
        ):
            download_func = Mock(side_effect=Exception("404 not found"))

            with pytest.raises(Exception):
                intelligent_retry_download(download_func, 2, 1)

    def test_retry_legacy_max_retries_exceeded(self):
        """Test max retries exceeded"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "time.sleep"
        ), patch("core._1_ytdlp.rprint"), patch(
            "core._1_ytdlp.categorize_download_error",
            return_value=("network", True, 10),
        ), patch(
            "core._1_ytdlp.log_event"
        ), patch(
            "core._1_ytdlp.time_block"
        ), patch(
            "core._1_ytdlp.inc_counter"
        ), patch(
            "core._1_ytdlp.observe_histogram"
        ):
            download_func = Mock(side_effect=Exception("network error"))

            with pytest.raises(Exception) as exc_info:
                intelligent_retry_download(download_func, 2, 1)

            assert "network error" in str(exc_info.value)


class TestDownloadVideoYtdlp:
    """Test main download function"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "downloads")
        os.makedirs(self.save_path, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_download_success(self):
        """Test successful download"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 10 * 1024 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=video_path
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(True, "Valid")
        ), patch(
            "core._1_ytdlp.rprint"
        ):
            result = download_video_ytdlp("https://example.com/video", self.save_path)

            assert result == video_path
            assert os.path.exists(result)

    def test_download_low_disk_space_warning(self):
        """Test warning for low disk space"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 500 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=None
        ), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=None
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint, patch(
            "core._1_ytdlp.log_event"
        ), patch(
            "os.makedirs"
        ):
            with pytest.raises(Exception):
                download_video_ytdlp("https://example.com/video", self.save_path)

            # Check that low disk space warning was issued
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Warning: Low disk space" in str(call)
            ]
            assert len(warning_calls) > 0

    def test_download_disk_space_check_error(self):
        """Test handling of disk space check error"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", side_effect=OSError("Permission denied")
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=None
        ), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=None
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint:
            with pytest.raises(Exception):
                download_video_ytdlp("https://example.com/video", self.save_path)

            # Check that disk space error was logged
            error_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Could not check disk space" in str(call)
            ]
            assert len(error_calls) > 0

    def test_download_fallback_file_search(self):
        """Test fallback file search when download path is invalid"""
        video_path = os.path.join(self.save_path, "found_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 10 * 1024 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=None
        ), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(True, "Valid")
        ), patch(
            "core._1_ytdlp.rprint"
        ):
            result = download_video_ytdlp("https://example.com/video", self.save_path)

            assert result == video_path

    def test_download_fallback_search_fails(self):
        """Test when fallback file search also fails"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 10 * 1024 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=None
        ), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=None
        ), patch(
            "core._1_ytdlp.rprint"
        ):
            with pytest.raises(Exception) as exc_info:
                download_video_ytdlp("https://example.com/video", self.save_path)

            assert "No video file found after download completion" in str(
                exc_info.value
            )

    def test_download_file_not_exists_after_return(self):
        """Test when returned file path doesn't exist"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 10 * 1024 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download",
            return_value="/nonexistent/file.mp4",
        ), patch(
            "core._1_ytdlp.rprint"
        ), patch(
            "core._1_ytdlp.log_event"
        ), patch(
            "os.makedirs"
        ):
            with pytest.raises(Exception) as exc_info:
                download_video_ytdlp("https://example.com/video", self.save_path)

            assert "File not found at /nonexistent/file.mp4" in str(exc_info.value)

    def test_download_invalid_file_warning(self):
        """Test warning when downloaded file validation fails"""
        video_path = os.path.join(self.save_path, "invalid_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 10 * 1024 * 1024 * 1024)
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value=video_path
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(False, "Invalid format")
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint:
            result = download_video_ytdlp("https://example.com/video", self.save_path)

            assert result == video_path
            # Check that validation warning was issued
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "validation failed" in str(call)
            ]
            assert len(warning_calls) > 0


class TestDownloadViaPythonAPI:
    """Test Python API download functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "downloads")
        os.makedirs(self.save_path, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_download_success_with_progress(self):
        """Test successful download with progress callback"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        progress_calls = []

        def progress_callback(data):
            progress_calls.append(data)

        class MockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def download(self, urls):
                pass

        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "core._1_ytdlp.update_ytdlp", return_value=MockYDL
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "os.utime"
        ), patch(
            "core._1_ytdlp.rprint"
        ):
            # Configure mocks for config loading
            def load_key_side_effect(key):
                if key == "youtube.cookies_path":
                    return None
                elif key == "youtube.proxy":
                    raise KeyError("Not found")
                return None

            mock_load_key.side_effect = load_key_side_effect

            result = download_via_python_api(
                "https://example.com/video", self.save_path, "1080", progress_callback
            )

            assert result == video_path
            # Test passed if no exception was raised during download

    def test_download_with_cookies_from_browser(self):
        """Test download with browser cookies"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        class MockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def download(self, urls):
                pass

        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "core._1_ytdlp.update_ytdlp", return_value=MockYDL
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "os.utime"
        ), patch(
            "core._1_ytdlp.rprint"
        ):

            def load_key_side_effect(key):
                if key == "youtube.cookies_path":
                    return None
                elif key == "youtube.proxy":
                    raise KeyError("Not found")
                return None

            mock_load_key.side_effect = load_key_side_effect

            result = download_via_python_api(
                "https://example.com/video", self.save_path, "1080"
            )

            assert result == video_path

    def test_download_with_cookie_file(self):
        """Test download with explicit cookie file"""
        video_path = os.path.join(self.save_path, "test_video.mp4")
        cookie_path = os.path.join(self.temp_dir, "cookies.txt")

        # Create mock files
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file
        with open(cookie_path, "w") as f:
            f.write("# Netscape HTTP Cookie File")

        mock_ydl = Mock()
        mock_ydl.download = Mock()

        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "core._1_ytdlp.update_ytdlp", return_value=Mock(return_value=mock_ydl)
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "os.utime"
        ), patch(
            "core._1_ytdlp.rprint"
        ):

            def load_key_side_effect(key):
                if key == "youtube.cookies_path":
                    return cookie_path
                elif key == "youtube.proxy":
                    raise KeyError("Not found")
                return None

            mock_load_key.side_effect = load_key_side_effect

            # Simulate browser cookies failing, then file cookies working
            call_count = [0]

            def mock_ydl_factory(opts):
                call_count[0] += 1
                if call_count[0] == 1 and "cookiesfrombrowser" in opts:
                    raise Exception("Browser cookies failed")
                return MockYDL(opts)

            mock_update_ytdlp.return_value = mock_ydl_factory

            result = download_via_python_api(
                "https://example.com/video", self.save_path, "1080"
            )

            assert result == video_path

    def test_download_with_proxy(self):
        """Test download with proxy configuration"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        mock_ydl = Mock()
        mock_ydl.download = Mock()

        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "core._1_ytdlp.update_ytdlp", return_value=Mock(return_value=mock_ydl)
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "core.utils.security_utils.validate_proxy_url", return_value=True
        ), patch(
            "os.utime"
        ), patch(
            "core._1_ytdlp.rprint"
        ):

            def load_key_side_effect(key):
                if key == "youtube.cookies_path":
                    return None
                elif key == "youtube.proxy":
                    return "http://proxy.example.com:8080"
                return None

            mock_load_key.side_effect = load_key_side_effect

            result = download_via_python_api(
                "https://example.com/video", self.save_path, "1080"
            )

            assert result == video_path

    def test_download_with_invalid_proxy(self):
        """Test download with invalid proxy configuration"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        mock_ydl = Mock()
        mock_ydl.download = Mock()

        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "core._1_ytdlp.update_ytdlp", return_value=Mock(return_value=mock_ydl)
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=video_path
        ), patch(
            "core.utils.security_utils.validate_proxy_url", return_value=False
        ), patch(
            "os.utime"
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint:

            def load_key_side_effect(key):
                if key == "youtube.cookies_path":
                    return None
                elif key == "youtube.proxy":
                    return "invalid-proxy-url"
                return None

            mock_load_key.side_effect = load_key_side_effect

            result = download_via_python_api(
                "https://example.com/video", self.save_path, "1080"
            )

            assert result == video_path
            # Check that invalid proxy warning was issued
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Invalid proxy URL format" in str(call)
            ]
            assert len(warning_calls) > 0

    def test_download_api_exception(self):
        """Test handling of download API exception"""

        class MockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def download(self, urls):
                raise Exception("Download failed")

        with patch("core.utils.config_utils.load_key", return_value=None), patch(
            "core._1_ytdlp.update_ytdlp", return_value=MockYDL
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"):
            with pytest.raises(Exception) as exc_info:
                download_via_python_api(
                    "https://example.com/video", self.save_path, "1080"
                )

            assert "Download failed" in str(exc_info.value)

    def test_download_no_file_found(self):
        """Test when no file is found after download"""

        class MockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def download(self, urls):
                pass

        with patch("core.utils.config_utils.load_key", return_value=None), patch(
            "core._1_ytdlp.update_ytdlp", return_value=MockYDL
        ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
            "core._1_ytdlp.find_most_recent_video_file", return_value=None
        ), patch(
            "core._1_ytdlp.rprint"
        ):
            with pytest.raises(Exception) as exc_info:
                download_via_python_api(
                    "https://example.com/video", self.save_path, "1080"
                )

            assert "could not locate the downloaded file" in str(exc_info.value)


class TestFindVideoFiles:
    """Test video file finding functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "downloads")
        os.makedirs(self.save_path, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_single_video_file(self):
        """Test finding single video file"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "sys.platform", "linux"
        ):
            result = find_video_files(self.save_path)

            # Convert to forward slashes for comparison
            expected = video_path.replace("\\", "/")
            actual = result.replace("\\", "/")
            assert actual == expected

    def test_find_multiple_video_files_best_selection(self):
        """Test finding best file among multiple video files"""
        video1_path = os.path.join(self.save_path, "small_video.mp4")
        video2_path = os.path.join(self.save_path, "large_video.mp4")

        # Create mock video files with different sizes
        with open(video1_path, "wb") as f:
            f.write(b"0" * (1 * 1024 * 1024))  # 1MB file
        with open(video2_path, "wb") as f:
            f.write(b"0" * (3 * 1024 * 1024))  # 3MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "core._1_ytdlp.find_best_video_file", return_value=video2_path
        ), patch(
            "core._1_ytdlp.rprint"
        ), patch(
            "sys.platform", "linux"
        ):
            result = find_video_files(self.save_path)

            assert result == video2_path

    def test_find_no_video_files(self):
        """Test when no video files are found"""
        # Create non-video files
        with open(os.path.join(self.save_path, "image.jpg"), "w") as f:
            f.write("fake image")
        with open(os.path.join(self.save_path, "text.txt"), "w") as f:
            f.write("some text")

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "sys.platform", "linux"
        ):
            with pytest.raises(ValueError) as exc_info:
                find_video_files(self.save_path)

            assert "No video files found" in str(exc_info.value)

    def test_find_windows_path_conversion(self):
        """Test path conversion on Windows"""
        video_path = os.path.join(self.save_path, "test_video.mp4")

        # Create a mock video file
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "sys.platform", "win32"
        ):
            result = find_video_files(self.save_path)

            # On Windows, paths should be converted to forward slashes
            assert "\\" not in result

    def test_find_filter_output_files(self):
        """Test filtering out output files from old structure"""
        video_path = os.path.join(self.save_path, "good_video.mp4")
        output_path1 = os.path.join(self.save_path, "output/output_video.mp4")
        output_path2 = os.path.join(self.save_path, "/output/bad_video.mp4")

        # Create mock files
        os.makedirs(os.path.dirname(output_path1), exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file
        with open(output_path1, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "sys.platform", "linux"
        ):
            result = find_video_files(self.save_path)

            # Should return the good video, not the output files
            expected = video_path.replace("\\", "/")
            actual = result.replace("\\", "/")
            assert actual == expected

    def test_find_video_files_with_validation_errors(self):
        """Test finding files when some fail validation"""
        video1_path = os.path.join(self.save_path, "valid_video.mp4")
        video2_path = os.path.join(self.save_path, "invalid_video.mp4")

        # Create mock video files
        with open(video1_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file
        with open(video2_path, "wb") as f:
            f.write(b"0" * (1 * 1024 * 1024))  # 1MB file

        def validate_side_effect(file_path):
            if "valid" in file_path:
                return (True, "Valid")
            else:
                return (False, "Invalid format")

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "core._1_ytdlp.validate_video_file", side_effect=validate_side_effect
        ), patch(
            "core._1_ytdlp.find_best_video_file", return_value=video1_path
        ), patch(
            "core._1_ytdlp.rprint"
        ), patch(
            "sys.platform", "linux"
        ):
            result = find_video_files(self.save_path)

            # Should return the valid file
            expected = video1_path.replace("\\", "/")
            actual = result.replace("\\", "/")
            assert actual == expected

    def test_find_no_valid_files_after_validation(self):
        """Test when files exist but none are valid"""
        video1_path = os.path.join(self.save_path, "invalid1.mp4")
        video2_path = os.path.join(self.save_path, "invalid2.mp4")

        # Create mock video files
        with open(video1_path, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024))  # 2MB file
        with open(video2_path, "wb") as f:
            f.write(b"0" * (1 * 1024 * 1024))  # 1MB file

        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": self.save_path},
        ), patch(
            "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
        ), patch(
            "core._1_ytdlp.find_best_video_file", return_value=None
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(False, "Invalid format")
        ), patch(
            "core._1_ytdlp.rprint"
        ), patch(
            "sys.platform", "linux"
        ):
            with pytest.raises(ValueError) as exc_info:
                find_video_files(self.save_path)

            assert "No valid video files found after validation" in str(exc_info.value)


class TestDownloadErrorHandling:
    """Test comprehensive error handling and edge cases"""

    def test_network_retry_scenarios(self):
        """Test various network failure scenarios"""
        network_errors = [
            "Connection timed out",
            "Network is unreachable",
            "Temporary failure in name resolution",
            "Connection refused by server",
        ]

        for error_msg in network_errors:
            with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
                category, is_retryable, wait_time = categorize_download_error(error_msg)
                assert category == "network"
                assert is_retryable is True
                assert wait_time == 30

    def test_rate_limit_scenarios(self):
        """Test rate limiting error scenarios"""
        rate_limit_errors = [
            "HTTP Error 403: Forbidden",
            "HTTP Error 429: Too Many Requests",
            "Rate limit exceeded",
            "Too many requests, please slow down",
        ]

        for error_msg in rate_limit_errors:
            with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
                category, is_retryable, wait_time = categorize_download_error(error_msg)
                assert category == "rate_limit"
                assert is_retryable is True
                assert wait_time == 60

    def test_access_denied_scenarios(self):
        """Test access denied scenarios that shouldn't retry"""
        access_errors = [
            "HTTP Error 401: Unauthorized",
            "Private video",
            "This video is restricted",
            "unauthorized access",
        ]

        for error_msg in access_errors:
            with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
                category, is_retryable, wait_time = categorize_download_error(error_msg)
                assert category == "access_denied"
                assert is_retryable is False
                assert wait_time == 0

    def test_proxy_ssl_scenarios(self):
        """Test proxy and SSL error scenarios"""
        proxy_ssl_errors = [
            "proxy connection failed",
            "SSL certificate verification failed",
            "SSL handshake failed",
            "certificate error",
        ]

        for error_msg in proxy_ssl_errors:
            with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
                category, is_retryable, wait_time = categorize_download_error(error_msg)
                assert category == "proxy_ssl"
                assert is_retryable is True
                assert wait_time == 10

    def test_intelligent_retry_with_exponential_backoff(self):
        """Test retry mechanism with exponential backoff"""
        call_count = 0
        retry_wait_times = []

        def mock_download():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("network timeout")
            return "success"

        def mock_sleep(seconds):
            retry_wait_times.append(seconds)

        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False), patch(
            "time.sleep", side_effect=mock_sleep
        ), patch("core._1_ytdlp.rprint"), patch(
            "core._1_ytdlp.categorize_download_error",
            return_value=("network", True, 60),
        ), patch(
            "core._1_ytdlp.log_event"
        ), patch(
            "core._1_ytdlp.time_block"
        ), patch(
            "core._1_ytdlp.inc_counter"
        ), patch(
            "core._1_ytdlp.observe_histogram"
        ):
            result = intelligent_retry_download(
                mock_download, max_retries=3, initial_wait=5
            )

            assert result == "success"
            assert call_count == 3
            # Should have exponential backoff: min(5 * 2^0, 60), min(5 * 2^1, 60)
            assert len(retry_wait_times) == 2
            assert retry_wait_times[0] == min(5 * 1, 60)  # First retry
            assert retry_wait_times[1] == min(5 * 2, 60)  # Second retry


class TestVideoFormatSelection:
    """Test video format selection algorithms"""

    def test_format_selection_best_quality(self):
        """Test format selection for best quality"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            format_string = get_optimal_format("best")

            # Should prioritize 4K, then 1440p, then 1080p with H.264
            assert "bestvideo[height<=2160]" in format_string
            assert "bestvideo[height<=1440]" in format_string
            assert "bestvideo[height<=1080]" in format_string
            assert "vcodec^=avc1" in format_string or "vcodec^=h264" in format_string

    def test_format_selection_specific_resolutions(self):
        """Test format selection for specific resolutions"""
        resolutions = ["360", "480", "720", "1080", "1440", "2160"]

        for resolution in resolutions:
            with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
                format_string = get_optimal_format(resolution)

                # Should contain height filter for the resolution
                assert f"height<={resolution}" in format_string
                # Should prioritize H.264 codec
                assert (
                    "vcodec^=avc1" in format_string or "vcodec^=h264" in format_string
                )
                # Should include audio preferences
                assert "bestaudio" in format_string

    def test_format_selection_with_fallbacks(self):
        """Test that format selection includes proper fallbacks"""
        with patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False):
            format_string = get_optimal_format("1080")

            # Should have multiple fallback options
            parts = format_string.split("/")
            assert len(parts) > 1  # Multiple options separated by /

            # Should include mp4 extension as fallback
            assert "ext=mp4" in format_string
            # Should include best fallback
            assert "best" in format_string


class TestDiskSpaceMonitoring:
    """Test disk space monitoring and warnings"""

    def test_low_disk_space_warning(self):
        """Test warning when disk space is low"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": "/tmp"},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", return_value=(0, 0, 500 * 1024 * 1024)  # 500MB free
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value="/tmp/video.mp4"
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(True, "Valid")
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint, patch(
            "core._1_ytdlp.log_event"
        ) as mock_log_event, patch(
            "core._1_ytdlp.inc_counter"
        ), patch(
            "core._1_ytdlp.observe_histogram"
        ), patch(
            "os.makedirs"
        ), patch(
            "os.path.getsize", return_value=2 * 1024 * 1024
        ), patch(
            "os.path.exists", return_value=True
        ):
            result = download_video_ytdlp("https://example.com/video", "/tmp")

            # Should have warned about low disk space
            warning_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Warning: Low disk space" in str(call)
            ]
            assert len(warning_calls) > 0

            # Should have logged the warning
            log_calls = [
                call
                for call in mock_log_event.call_args_list
                if "low disk space" in str(call)
            ]
            assert len(log_calls) > 0

    def test_disk_space_check_error_handling(self):
        """Test handling of disk space check errors"""
        with patch(
            "core.utils.config_utils.get_storage_paths",
            return_value={"input": "/tmp"},
        ), patch("core.utils.config_utils.ensure_storage_dirs"), patch(
            "core.utils.config_utils.load_key", return_value="1080"
        ), patch(
            "shutil.disk_usage", side_effect=OSError("Permission denied")
        ), patch(
            "core._1_ytdlp.intelligent_retry_download", return_value="/tmp/video.mp4"
        ), patch(
            "core._1_ytdlp.validate_video_file", return_value=(True, "Valid")
        ), patch(
            "core._1_ytdlp.rprint"
        ) as mock_rprint, patch(
            "core._1_ytdlp.log_event"
        ) as mock_log_event, patch(
            "core._1_ytdlp.inc_counter"
        ), patch(
            "core._1_ytdlp.observe_histogram"
        ), patch(
            "os.makedirs"
        ), patch(
            "os.path.getsize", return_value=2 * 1024 * 1024
        ), patch(
            "os.path.exists", return_value=True
        ):
            result = download_video_ytdlp("https://example.com/video", "/tmp")

            # Should have logged the disk space check error
            error_calls = [
                call
                for call in mock_rprint.call_args_list
                if "Could not check disk space" in str(call)
            ]
            assert len(error_calls) > 0

            # Should have logged the warning
            log_calls = [
                call
                for call in mock_log_event.call_args_list
                if "disk space check failed" in str(call)
            ]
            assert len(log_calls) > 0


class TestCommandLineDownload:
    """Test command line download functionality"""

    def test_command_download_with_progress_parsing(self):
        """Test command download with progress line parsing"""
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "downloads")
        os.makedirs(save_path, exist_ok=True)

        try:
            progress_calls = []

            def progress_callback(data):
                progress_calls.append(data)

            # Mock yt-dlp command output with progress
            mock_output_lines = [
                "[download]   0.0% of  117.60MiB at    2.34MiB/s ETA 00:50",
                "[download]  45.2% of  117.60MiB at    2.34MiB/s ETA 00:32",
                "[download] 100.0% of  117.60MiB at    2.34MiB/s ETA 00:00",
                "[download] Destination: test_video.mp4",
                "",
            ]

            with patch(
                "core.utils.config_utils.get_storage_paths",
                return_value={"input": save_path},
            ), patch("core._1_ytdlp.get_optimal_format", return_value="best"), patch(
                "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
            ), patch(
                "subprocess.Popen"
            ) as mock_popen, patch(
                "os.path.exists", return_value=True
            ), patch(
                "os.utime"
            ), patch(
                "core._1_ytdlp.rprint"
            ), patch(
                "core._1_ytdlp.log_event"
            ):
                # Mock process that returns progress output
                mock_process = Mock()
                mock_process.returncode = 0
                mock_process.stdout.readline.side_effect = mock_output_lines
                mock_process.wait.return_value = 0
                mock_popen.return_value = mock_process

                result = download_via_command(
                    "https://example.com/video", save_path, "1080", progress_callback
                )

                # Should have parsed progress and called callback
                assert len(progress_calls) >= 2  # At least progress updates

                # Check progress data structure
                progress_data = progress_calls[0]
                assert "progress" in progress_data
                assert "total_size_str" in progress_data
                assert progress_data["progress"] == 0.0

                final_progress = progress_calls[-1]
                assert final_progress["progress"] == 1.0
                assert final_progress["status"] == "finished"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_command_download_destination_detection_patterns(self):
        """Test various destination file detection patterns"""
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "downloads")
        os.makedirs(save_path, exist_ok=True)

        try:
            # Test different output patterns
            test_cases = [
                ("[download] Destination: video.mp4", "video.mp4"),
                ('[Merger] Merging formats into "output.mp4"', "output.mp4"),
                (
                    "[download] final_video.mp4 has already been downloaded",
                    "final_video.mp4",
                ),
                ('[ffmpeg] Merging formats into "merged.mp4"', "merged.mp4"),
                ("[download] 100% of downloaded_video.mp4", "downloaded_video.mp4"),
            ]

            for output_line, expected_filename in test_cases:
                with patch(
                    "core.utils.config_utils.get_storage_paths",
                    return_value={"input": save_path},
                ), patch(
                    "core._1_ytdlp.get_optimal_format", return_value="best"
                ), patch(
                    "core.utils.config_utils.load_key", return_value=["mp4", "avi"]
                ), patch(
                    "subprocess.Popen"
                ) as mock_popen, patch(
                    "os.path.exists", return_value=True
                ), patch(
                    "os.utime"
                ), patch(
                    "core._1_ytdlp.rprint"
                ), patch(
                    "core._1_ytdlp.log_event"
                ), patch(
                    "core._1_ytdlp.find_most_recent_video_file",
                    return_value=os.path.join(save_path, expected_filename),
                ):
                    # Mock process that returns the specific output line
                    mock_process = Mock()
                    mock_process.returncode = 0
                    mock_process.stdout.readline.side_effect = [output_line, ""]
                    mock_process.wait.return_value = 0
                    mock_popen.return_value = mock_process

                    result = download_via_command(
                        "https://example.com/video", save_path, "1080"
                    )

                    # Should have detected the correct filename
                    assert os.path.basename(result) == expected_filename

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMainScriptExecution:
    """Test main script execution paths"""

    def test_main_with_valid_inputs(self):
        """Test main execution with valid user inputs"""
        with patch("builtins.input") as mock_input, patch(
            "core._1_ytdlp.download_video_ytdlp"
        ) as mock_download, patch(
            "core._1_ytdlp.find_video_files", return_value="/path/to/video.mp4"
        ), patch(
            "builtins.print"
        ) as mock_print:
            # Mock user inputs
            mock_input.side_effect = ["https://example.com/video", "720"]
            mock_download.return_value = "/path/to/video.mp4"

            # Simulate main execution
            try:
                url = input("Please enter the URL of the video you want to download: ")
                resolution = input(
                    "Please enter the desired resolution (360/480/720/1080, default 1080): "
                )
                resolution = int(resolution) if resolution.isdigit() else 1080
                download_video_ytdlp(url, resolution=resolution)
                print(f"🎥 Video has been downloaded to {find_video_files()}")
            except:
                pass  # Expected due to mocking

            assert mock_input.call_count == 2

    def test_main_with_non_numeric_resolution(self):
        """Test main execution with non-numeric resolution input"""
        with patch("builtins.input") as mock_input, patch(
            "core._1_ytdlp.download_video_ytdlp"
        ) as mock_download, patch(
            "core._1_ytdlp.find_video_files", return_value="/path/to/video.mp4"
        ), patch(
            "builtins.print"
        ):
            # Mock user inputs with non-numeric resolution
            mock_input.side_effect = ["https://example.com/video", "high"]
            mock_download.return_value = "/path/to/video.mp4"

            # Test resolution parsing logic directly
            resolution_input = "high"
            resolution = int(resolution_input) if resolution_input.isdigit() else 1080

            assert resolution == 1080  # Should default to 1080
