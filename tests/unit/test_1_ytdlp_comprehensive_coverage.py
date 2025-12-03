"""
Comprehensive test suite for core/_1_ytdlp.py to achieve 85% branch coverage.
Focuses on all critical branches identified in the coverage analysis.
"""

import pytest
import os
import sys
import subprocess
import tempfile
import shutil
import json
import glob
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
import time

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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

    @patch("core._1_ytdlp.subprocess.check_call")
    @patch("core._1_ytdlp.sys.modules", {"yt_dlp": MagicMock()})
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    def test_update_ytdlp_success(self, mock_log_event, mock_rprint, mock_check_call):
        """Test successful yt-dlp update"""
        mock_ytdl = MagicMock()
        with patch("yt_dlp.YoutubeDL", return_value=mock_ytdl):
            result = update_ytdlp()

            mock_check_call.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
            )
            mock_rprint.assert_called_with("[green]yt-dlp updated[/green]")
            mock_log_event.assert_called_with(
                "info", "yt-dlp updated", stage="download", op="update_ytdlp"
            )
            assert result is not None

    @patch("core._1_ytdlp.subprocess.check_call")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    def test_update_ytdlp_failure(self, mock_log_event, mock_rprint, mock_check_call):
        """Test yt-dlp update failure"""
        error_msg = "Command failed"
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip", error_msg)

        mock_ytdl = MagicMock()
        with patch("yt_dlp.YoutubeDL", return_value=mock_ytdl):
            result = update_ytdlp()

            mock_rprint.assert_called_with(
                "[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]"
            )
            mock_log_event.assert_called_with(
                "warning",
                f"update yt-dlp failed: Command 'pip' returned non-zero exit status 1.",
                stage="download",
                op="update_ytdlp",
            )
            assert result is not None


class TestValidateVideoFile:
    """Test video file validation functionality"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_validate_video_file_not_exists(self):
        """Test validation of non-existent file"""
        non_existent_file = "/path/to/nonexistent.mp4"

        is_valid, error_msg = validate_video_file(non_existent_file)

        assert not is_valid
        assert "File does not exist" in error_msg

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=500 * 1024)  # 500KB
    def test_validate_video_file_too_small(self, mock_getsize, mock_exists):
        """Test validation of file that's too small"""
        test_file = "/path/to/small.mp4"

        is_valid, error_msg = validate_video_file(test_file, expected_min_size_mb=1)

        assert not is_valid
        assert "File too small" in error_msg

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=2 * 1024 * 1024)  # 2MB
    def test_validate_video_file_partial_download(self, mock_getsize, mock_exists):
        """Test validation of partial download file"""
        test_files = [
            "/path/to/video.part",
            "/path/to/video.tmp",
            "/path/to/video.download",
        ]

        for test_file in test_files:
            is_valid, error_msg = validate_video_file(test_file)
            assert not is_valid
            assert "partial download" in error_msg

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=5 * 1024 * 1024)  # 5MB
    @patch("core._1_ytdlp.subprocess.run")
    def test_validate_video_file_ffprobe_success(
        self, mock_subprocess_run, mock_getsize, mock_exists
    ):
        """Test validation with successful ffprobe"""
        mock_subprocess_run.return_value.returncode = 0
        test_file = "/path/to/valid.mp4"

        is_valid, error_msg = validate_video_file(test_file)

        assert is_valid
        assert error_msg == "File is valid"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=5 * 1024 * 1024)  # 5MB
    @patch("core._1_ytdlp.subprocess.run")
    def test_validate_video_file_ffprobe_failure(
        self, mock_subprocess_run, mock_getsize, mock_exists
    ):
        """Test validation with failed ffprobe"""
        mock_subprocess_run.return_value.returncode = 1
        test_file = "/path/to/invalid.mp4"

        is_valid, error_msg = validate_video_file(test_file)

        assert not is_valid
        assert "File format validation failed" in error_msg

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=5 * 1024 * 1024)  # 5MB
    @patch("core._1_ytdlp.subprocess.run")
    def test_validate_video_file_ffprobe_timeout(
        self, mock_subprocess_run, mock_getsize, mock_exists
    ):
        """Test validation with ffprobe timeout"""
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("ffprobe", 10)
        test_file = "/path/to/timeout.mp4"

        is_valid, error_msg = validate_video_file(test_file)

        # Should skip detailed validation and return valid
        assert is_valid
        assert error_msg == "File is valid"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=5 * 1024 * 1024)  # 5MB
    @patch("core._1_ytdlp.subprocess.run")
    def test_validate_video_file_ffprobe_not_found(
        self, mock_subprocess_run, mock_getsize, mock_exists
    ):
        """Test validation when ffprobe not found"""
        mock_subprocess_run.side_effect = FileNotFoundError("ffprobe not found")
        test_file = "/path/to/video.mp4"

        is_valid, error_msg = validate_video_file(test_file)

        # Should skip detailed validation and return valid
        assert is_valid
        assert error_msg == "File is valid"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_validate_video_file_modular_success(self):
        """Test validation using modular components"""
        test_file = "/path/to/video.mp4"

        with patch("core._1_ytdlp.VideoFileValidator") as mock_validator:
            mock_instance = mock_validator.return_value
            mock_instance.validate_video_file.return_value = (True, "Valid file")

            is_valid, error_msg = validate_video_file(test_file, expected_min_size_mb=2)

            mock_validator.assert_called_once_with(min_size_mb=2)
            mock_instance.validate_video_file.assert_called_once_with(test_file)
            assert is_valid
            assert error_msg == "Valid file"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.os.path.exists", return_value=True)
    @patch("core._1_ytdlp.os.path.getsize", return_value=5 * 1024 * 1024)
    def test_validate_video_file_modular_fallback(self, mock_getsize, mock_exists):
        """Test validation fallback when modular components fail"""
        test_file = "/path/to/video.mp4"

        with patch("core._1_ytdlp.VideoFileValidator") as mock_validator:
            mock_validator.side_effect = Exception("Modular component failed")

            with patch("core._1_ytdlp.subprocess.run") as mock_subprocess:
                mock_subprocess.return_value.returncode = 0

                is_valid, error_msg = validate_video_file(test_file)

                assert is_valid
                assert error_msg == "File is valid"


class TestCleanupPartialDownloads:
    """Test partial download cleanup functionality"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.remove")
    @patch("core._1_ytdlp.rprint")
    def test_cleanup_partial_downloads_success(
        self, mock_rprint, mock_remove, mock_glob
    ):
        """Test successful cleanup of partial downloads"""
        save_path = "/test/downloads"
        partial_files = [
            f"{save_path}/video.part",
            f"{save_path}/audio.tmp",
            f"{save_path}/file.download",
        ]

        mock_glob.side_effect = [
            [partial_files[0]],  # *.part
            [partial_files[1]],  # *.tmp
            [partial_files[2]],  # *.download
            [],  # *.f*
            [],  # *.ytdl
        ]

        cleaned_files = cleanup_partial_downloads(save_path)

        expected_calls = [
            call(os.path.join(save_path, "*.part")),
            call(os.path.join(save_path, "*.tmp")),
            call(os.path.join(save_path, "*.download")),
            call(os.path.join(save_path, "*.f*")),
            call(os.path.join(save_path, "*.ytdl")),
        ]
        mock_glob.assert_has_calls(expected_calls)

        assert len(cleaned_files) == 3
        assert all(
            name in ["video.part", "audio.tmp", "file.download"]
            for name in cleaned_files
        )
        mock_rprint.assert_called()

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.remove")
    @patch("core._1_ytdlp.rprint")
    def test_cleanup_partial_downloads_permission_error(
        self, mock_rprint, mock_remove, mock_glob
    ):
        """Test cleanup with permission errors"""
        save_path = "/test/downloads"
        partial_file = f"{save_path}/video.part"

        mock_glob.side_effect = [
            [partial_file],  # *.part
            [],
            [],
            [],
            [],  # other patterns
        ]
        mock_remove.side_effect = OSError("Permission denied")

        cleaned_files = cleanup_partial_downloads(save_path)

        assert len(cleaned_files) == 0
        mock_rprint.assert_any_call(
            f"[yellow]Warning: Could not remove {partial_file}: Permission denied[/yellow]"
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_cleanup_partial_downloads_modular(self):
        """Test cleanup using modular components"""
        save_path = "/test/downloads"

        with patch("core._1_ytdlp.PartialDownloadCleaner") as mock_cleaner:
            mock_instance = mock_cleaner.return_value
            mock_instance.cleanup_partial_downloads.return_value = [
                "file1.part",
                "file2.tmp",
            ]

            cleaned_files = cleanup_partial_downloads(save_path)

            mock_cleaner.assert_called_once()
            mock_instance.cleanup_partial_downloads.assert_called_once_with(save_path)
            assert cleaned_files == ["file1.part", "file2.tmp"]

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.glob.glob", return_value=[])
    def test_cleanup_partial_downloads_modular_fallback(self, mock_glob):
        """Test cleanup fallback when modular components fail"""
        save_path = "/test/downloads"

        with patch("core._1_ytdlp.PartialDownloadCleaner") as mock_cleaner:
            mock_cleaner.side_effect = Exception("Modular component failed")

            cleaned_files = cleanup_partial_downloads(save_path)

            assert cleaned_files == []


class TestFindMostRecentVideoFile:
    """Test finding most recent video file"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.os.path.getmtime")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_most_recent_video_file_success(
        self, mock_validate, mock_getmtime, mock_isfile, mock_glob, mock_load_key
    ):
        """Test successfully finding most recent video file"""
        save_path = "/test/downloads"
        mock_load_key.return_value = ["mp4", "mkv", "avi"]

        video_files = [
            f"{save_path}/video1.mp4",
            f"{save_path}/video2.mkv",
            f"{save_path}/video3.avi",
        ]

        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_validate.return_value = (True, "Valid")

        # Set different modification times
        mock_getmtime.side_effect = [1000, 3000, 2000]  # video2.mkv is most recent

        result = find_most_recent_video_file(save_path)

        assert result == f"{save_path}/video2.mkv"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    def test_find_most_recent_video_file_no_files(self, mock_glob, mock_load_key):
        """Test when no video files are found"""
        save_path = "/test/downloads"
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = []

        result = find_most_recent_video_file(save_path)

        assert result is None

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_most_recent_video_file_invalid_files(
        self, mock_validate, mock_isfile, mock_glob, mock_load_key
    ):
        """Test when all files are invalid"""
        save_path = "/test/downloads"
        mock_load_key.return_value = ["mp4", "mkv", "avi"]

        video_files = [f"{save_path}/video1.mp4", f"{save_path}/video2.mkv"]
        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_validate.return_value = (False, "Invalid file")

        result = find_most_recent_video_file(save_path)

        assert result is None

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    def test_find_most_recent_video_file_filter_extensions(
        self, mock_isfile, mock_glob, mock_load_key
    ):
        """Test filtering by allowed extensions"""
        save_path = "/test/downloads"
        mock_load_key.return_value = ["mp4", "mkv"]  # Only mp4 and mkv allowed

        all_files = [
            f"{save_path}/video1.mp4",  # Valid
            f"{save_path}/video2.avi",  # Invalid extension
            f"{save_path}/video3.mkv",  # Valid
            f"{save_path}/audio.mp3",  # Invalid extension
            f"{save_path}/image.jpg",  # Should be filtered out by endswith check
        ]

        mock_glob.return_value = all_files
        mock_isfile.return_value = True

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (True, "Valid")
            with patch("core._1_ytdlp.os.path.getmtime") as mock_getmtime:
                mock_getmtime.side_effect = [1000, 2000]  # video3.mkv is more recent

                result = find_most_recent_video_file(save_path)

                assert result == f"{save_path}/video3.mkv"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.load_key")
    def test_find_most_recent_video_file_modular(self, mock_load_key):
        """Test using modular components"""
        save_path = "/test/downloads"
        allowed_formats = ["mp4", "mkv", "avi"]
        mock_load_key.return_value = allowed_formats

        with patch("core._1_ytdlp.VideoFileValidator") as mock_validator:
            mock_instance = mock_validator.return_value
            mock_instance.find_most_recent_video_file.return_value = (
                "/test/downloads/video.mp4"
            )

            result = find_most_recent_video_file(save_path)

            mock_validator.assert_called_once()
            mock_instance.find_most_recent_video_file.assert_called_once_with(
                save_path, allowed_formats
            )
            assert result == "/test/downloads/video.mp4"


class TestFindBestVideoFile:
    """Test finding best video file"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.os.path.getsize")
    @patch("core._1_ytdlp.validate_video_file")
    @patch("core._1_ytdlp.rprint")
    def test_find_best_video_file_success(
        self, mock_rprint, mock_validate, mock_getsize, mock_isfile, mock_glob
    ):
        """Test successfully finding best video file"""
        save_path = "/test/downloads"
        allowed_formats = ["mp4", "mkv", "avi"]

        video_files = [
            f"{save_path}/video1.mp4",
            f"{save_path}/video2.mkv",
            f"{save_path}/video3.avi",
        ]

        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_validate.return_value = (True, "Valid")

        # Set different file sizes - video2.mkv is largest
        mock_getsize.side_effect = [
            100 * 1024 * 1024,
            200 * 1024 * 1024,
            50 * 1024 * 1024,
        ]

        result = find_best_video_file(save_path, allowed_formats)

        assert result == f"{save_path}/video2.mkv"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.glob.glob")
    def test_find_best_video_file_no_candidates(self, mock_glob):
        """Test when no valid candidates are found"""
        save_path = "/test/downloads"
        allowed_formats = ["mp4", "mkv", "avi"]
        mock_glob.return_value = []

        result = find_best_video_file(save_path, allowed_formats)

        assert result is None

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.validate_video_file")
    @patch("core._1_ytdlp.rprint")
    def test_find_best_video_file_invalid_files(
        self, mock_rprint, mock_validate, mock_isfile, mock_glob
    ):
        """Test when all files are invalid"""
        save_path = "/test/downloads"
        allowed_formats = ["mp4", "mkv", "avi"]

        video_files = [f"{save_path}/video1.mp4", f"{save_path}/video2.mkv"]
        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_validate.return_value = (False, "Invalid file")

        result = find_best_video_file(save_path, allowed_formats)

        assert result is None
        mock_rprint.assert_any_call(
            "[yellow]Skipping invalid file video1.mp4: Invalid file[/yellow]"
        )
        mock_rprint.assert_any_call(
            "[yellow]Skipping invalid file video2.mkv: Invalid file[/yellow]"
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_find_best_video_file_modular(self):
        """Test using modular components"""
        save_path = "/test/downloads"
        allowed_formats = ["mp4", "mkv", "avi"]

        with patch("core._1_ytdlp.VideoFileValidator") as mock_validator:
            mock_instance = mock_validator.return_value
            mock_instance.find_best_video_file.return_value = (
                "/test/downloads/best_video.mp4"
            )

            result = find_best_video_file(save_path, allowed_formats)

            mock_validator.assert_called_once()
            mock_instance.find_best_video_file.assert_called_once_with(
                save_path, allowed_formats
            )
            assert result == "/test/downloads/best_video.mp4"


class TestGetOptimalFormat:
    """Test optimal format selection"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_get_optimal_format_best(self):
        """Test getting optimal format for 'best' quality"""
        result = get_optimal_format("best")

        assert "bestvideo[height<=2160]" in result
        assert "bestvideo[height<=1440]" in result
        assert "bestvideo[height<=1080]" in result
        assert "bestvideo[height<=720]" in result
        assert "avc1" in result or "h264" in result

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_get_optimal_format_specific_resolution(self):
        """Test getting optimal format for specific resolution"""
        resolution = "1080"
        result = get_optimal_format(resolution)

        assert f"[height<={resolution}]" in result
        assert "avc1" in result or "h264" in result
        assert "bestaudio" in result

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_get_optimal_format_modular(self):
        """Test using modular format selector"""
        resolution = "720"

        with patch("core._1_ytdlp.VideoFormatSelector") as mock_selector:
            mock_instance = mock_selector.return_value
            mock_instance.get_optimal_format.return_value = "best[height<=720]"

            result = get_optimal_format(resolution)

            mock_selector.assert_called_once()
            mock_instance.get_optimal_format.assert_called_once_with(str(resolution))
            assert result == "best[height<=720]"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)  # Fallback scenario
    def test_get_optimal_format_modular_fallback(self):
        """Test fallback when modular components fail"""
        resolution = "480"

        with patch("core._1_ytdlp.VideoFormatSelector") as mock_selector:
            mock_selector.side_effect = Exception("Modular component failed")

            result = get_optimal_format(resolution)

            assert f"[height<={resolution}]" in result


class TestCategorizeDownloadError:
    """Test download error categorization"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_network(self):
        """Test network error categorization"""
        error_messages = [
            "Network timeout occurred",
            "Connection failed",
            "Temporary network issue",
            "Connection timed out",
        ]

        for error_msg in error_messages:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "network"
            assert is_retryable is True
            assert wait_time == 30

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_rate_limit(self):
        """Test rate limit error categorization"""
        error_messages = [
            "403 Forbidden",
            "429 Too Many Requests",
            "Rate limit exceeded",
            "Too many requests from this IP",
        ]

        for error_msg in error_messages:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "rate_limit"
            assert is_retryable is True
            assert wait_time == 60

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_not_found(self):
        """Test not found error categorization"""
        error_messages = [
            "404 Not Found",
            "Video not found",
            "Content unavailable",
            "This video is not available",
        ]

        for error_msg in error_messages:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "not_found"
            assert is_retryable is False
            assert wait_time == 0

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_access_denied(self):
        """Test access denied error categorization"""
        error_messages = [
            "401 Unauthorized",
            "Access denied",
            "Private video",
            "Restricted content",
        ]

        for error_msg in error_messages:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "access_denied"
            assert is_retryable is False
            assert wait_time == 0

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_proxy_ssl(self):
        """Test proxy/SSL error categorization"""
        error_messages = [
            "Proxy connection failed",
            "SSL certificate error",
            "Certificate verification failed",
        ]

        for error_msg in error_messages:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "proxy_ssl"
            assert is_retryable is True
            assert wait_time == 10

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    def test_categorize_download_error_unknown(self):
        """Test unknown error categorization"""
        error_msg = "Some unknown error occurred"
        category, is_retryable, wait_time = categorize_download_error(error_msg)

        assert category == "unknown"
        assert is_retryable is True
        assert wait_time == 20

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_categorize_download_error_modular(self):
        """Test using modular error handler"""
        error_msg = "Network timeout"

        with patch("core._1_ytdlp.DownloadErrorHandler") as mock_handler:
            mock_instance = mock_handler.return_value
            mock_category = MagicMock()
            mock_category.value = "network"
            mock_instance.categorize_download_error.return_value = (
                mock_category,
                True,
                30,
            )

            category, is_retryable, wait_time = categorize_download_error(error_msg)

            mock_handler.assert_called_once()
            mock_instance.categorize_download_error.assert_called_once_with(error_msg)
            assert category == "network"
            assert is_retryable is True
            assert wait_time == 30


class TestIntelligentRetryDownload:
    """Test intelligent retry mechanism"""

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    @patch("core._1_ytdlp.inc_counter")
    def test_intelligent_retry_download_success_first_attempt(
        self, mock_inc_counter, mock_time_block, mock_log_event
    ):
        """Test successful download on first attempt"""
        mock_download_func = Mock(return_value="success")
        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)

        result = intelligent_retry_download(mock_download_func, max_retries=3)

        assert result == "success"
        mock_download_func.assert_called_once()
        mock_log_event.assert_called_with(
            "info",
            "download retry start",
            stage="download",
            op="retry",
            max_retries=3,
            initial_wait=5,
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.categorize_download_error")
    @patch("core._1_ytdlp.time.sleep")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    @patch("core._1_ytdlp.inc_counter")
    @patch("core._1_ytdlp.observe_histogram")
    def test_intelligent_retry_download_success_after_retry(
        self,
        mock_observe_histogram,
        mock_inc_counter,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_sleep,
        mock_categorize,
    ):
        """Test successful download after retries"""
        mock_download_func = Mock()
        mock_download_func.side_effect = [
            Exception("Network timeout"),  # First attempt fails
            "success",  # Second attempt succeeds
        ]

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)
        mock_categorize.return_value = ("network", True, 30)

        result = intelligent_retry_download(
            mock_download_func, max_retries=3, initial_wait=5
        )

        assert result == "success"
        assert mock_download_func.call_count == 2
        mock_sleep.assert_called_once_with(30)  # Uses suggested wait time
        mock_inc_counter.assert_called_with(
            "download.retry_success", 1, stage="download", op="retry", attempt=1
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.categorize_download_error")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    @patch("core._1_ytdlp.inc_counter")
    def test_intelligent_retry_download_non_retryable_error(
        self,
        mock_inc_counter,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_categorize,
    ):
        """Test non-retryable error handling"""
        mock_download_func = Mock()
        error = Exception("404 Not Found")
        mock_download_func.side_effect = error

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)
        mock_categorize.return_value = ("not_found", False, 0)

        with pytest.raises(Exception) as exc_info:
            intelligent_retry_download(mock_download_func, max_retries=3)

        assert exc_info.value == error
        mock_download_func.assert_called_once()
        mock_rprint.assert_called_with(
            "[red]Non-retryable error (not_found): 404 Not Found[/red]"
        )
        mock_inc_counter.assert_called_with(
            "download.error", 1, stage="download", category="not_found"
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", False)
    @patch("core._1_ytdlp.categorize_download_error")
    @patch("core._1_ytdlp.time.sleep")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    @patch("core._1_ytdlp.inc_counter")
    @patch("core._1_ytdlp.observe_histogram")
    def test_intelligent_retry_download_max_retries_exceeded(
        self,
        mock_observe_histogram,
        mock_inc_counter,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_sleep,
        mock_categorize,
    ):
        """Test max retries exceeded"""
        mock_download_func = Mock()
        error = Exception("Network timeout")
        mock_download_func.side_effect = error

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)
        mock_categorize.return_value = ("network", True, 30)

        with pytest.raises(Exception) as exc_info:
            intelligent_retry_download(mock_download_func, max_retries=2)

        assert exc_info.value == error
        assert mock_download_func.call_count == 3  # Initial + 2 retries
        mock_rprint.assert_any_call(
            "[red]Download failed after 2 retries: Network timeout[/red]"
        )
        mock_inc_counter.assert_called_with(
            "download.error", 1, stage="download", category="network"
        )

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    def test_intelligent_retry_download_modular(self):
        """Test using modular error handler"""
        mock_download_func = Mock(return_value="success")

        with patch("core._1_ytdlp.DownloadErrorHandler") as mock_handler:
            mock_instance = mock_handler.return_value
            mock_instance.intelligent_retry_download.return_value = "success"

            result = intelligent_retry_download(
                mock_download_func, max_retries=3, initial_wait=5
            )

            mock_handler.assert_called_once()
            mock_instance.intelligent_retry_download.assert_called_once_with(
                mock_download_func, 3, 5
            )
            assert result == "success"


class TestDownloadVideoYtdlp:
    """Test main download function"""

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.ensure_storage_dirs")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.makedirs")
    @patch("core._1_ytdlp.shutil.disk_usage")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.validate_video_file")
    @patch("core._1_ytdlp.os.path.getsize")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    @patch("core._1_ytdlp.inc_counter")
    @patch("core._1_ytdlp.observe_histogram")
    def test_download_video_ytdlp_success(
        self,
        mock_observe_histogram,
        mock_inc_counter,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_getsize,
        mock_validate,
        mock_exists,
        mock_retry_download,
        mock_disk_usage,
        mock_makedirs,
        mock_load_key,
        mock_ensure_dirs,
        mock_get_paths,
    ):
        """Test successful video download"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        downloaded_file = f"{save_path}/test_video.mp4"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = MagicMock(
            free=5 * 1024 * 1024 * 1024
        )  # 5GB free
        mock_retry_download.return_value = downloaded_file
        mock_exists.return_value = True
        mock_validate.return_value = (True, "Valid file")
        mock_getsize.return_value = 100 * 1024 * 1024  # 100MB

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)

        result = download_video_ytdlp(url)

        assert result == downloaded_file
        mock_ensure_dirs.assert_called_once()
        mock_makedirs.assert_called_once_with(save_path, exist_ok=True)
        mock_inc_counter.assert_called_with("download.success", 1, stage="download")
        mock_observe_histogram.assert_called_with(
            "download.file_size_mb", 100.0, stage="download"
        )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.ensure_storage_dirs")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.makedirs")
    @patch("core._1_ytdlp.shutil.disk_usage")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    def test_download_video_ytdlp_low_disk_space_warning(
        self,
        mock_log_event,
        mock_rprint,
        mock_disk_usage,
        mock_makedirs,
        mock_load_key,
        mock_ensure_dirs,
        mock_get_paths,
    ):
        """Test low disk space warning"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = MagicMock(
            free=500 * 1024 * 1024
        )  # 500MB free (< 1GB)

        with patch("core._1_ytdlp.intelligent_retry_download") as mock_retry:
            mock_retry.side_effect = Exception("Test exception to stop execution")

            with pytest.raises(Exception):
                download_video_ytdlp(url)

            mock_rprint.assert_any_call(
                "[yellow]Warning: Low disk space (0.5GB available). Download may fail.[/yellow]"
            )
            mock_log_event.assert_any_call(
                "warning",
                "low disk space",
                stage="download",
                op="precheck",
                free_gb=0.5,
            )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.ensure_storage_dirs")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.makedirs")
    @patch("core._1_ytdlp.shutil.disk_usage")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    def test_download_video_ytdlp_fallback_search(
        self,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_exists,
        mock_find_recent,
        mock_retry_download,
        mock_disk_usage,
        mock_makedirs,
        mock_load_key,
        mock_ensure_dirs,
        mock_get_paths,
    ):
        """Test fallback file search when download function returns invalid path"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        fallback_file = f"{save_path}/fallback_video.mp4"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = MagicMock(free=5 * 1024 * 1024 * 1024)
        mock_retry_download.return_value = None  # Invalid path
        mock_exists.side_effect = [False, True]  # First call False, second True
        mock_find_recent.return_value = fallback_file

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (True, "Valid file")
            with patch("core._1_ytdlp.os.path.getsize") as mock_getsize:
                mock_getsize.return_value = 100 * 1024 * 1024

                result = download_video_ytdlp(url)

                assert result == fallback_file
                mock_rprint.assert_any_call(
                    "[yellow]Download function returned invalid path, searching for downloaded file...[/yellow]"
                )
                mock_rprint.assert_any_call(
                    f"[green]Found downloaded file: {os.path.basename(fallback_file)}[/green]"
                )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.ensure_storage_dirs")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.makedirs")
    @patch("core._1_ytdlp.shutil.disk_usage")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.rprint")
    @patch("core._1_ytdlp.log_event")
    @patch("core._1_ytdlp.time_block")
    def test_download_video_ytdlp_fallback_failure(
        self,
        mock_time_block,
        mock_log_event,
        mock_rprint,
        mock_exists,
        mock_find_recent,
        mock_retry_download,
        mock_disk_usage,
        mock_makedirs,
        mock_load_key,
        mock_ensure_dirs,
        mock_get_paths,
    ):
        """Test fallback failure handling"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = MagicMock(free=5 * 1024 * 1024 * 1024)
        mock_retry_download.return_value = None
        mock_exists.return_value = False
        mock_find_recent.return_value = None  # No file found

        mock_time_block.return_value.__enter__ = Mock(return_value=None)
        mock_time_block.return_value.__exit__ = Mock(return_value=False)

        with pytest.raises(Exception) as exc_info:
            download_video_ytdlp(url)

        assert "Download failed" in str(exc_info.value)
        assert "Unable to locate downloaded file" in str(exc_info.value)


class TestDownloadViaCommand:
    """Test command-line download functionality"""

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.subprocess.Popen")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.log_event")
    def test_download_via_command_success(
        self,
        mock_log_event,
        mock_exists,
        mock_popen,
        mock_get_format,
        mock_validate_proxy,
        mock_load_key,
        mock_get_paths,
    ):
        """Test successful command-line download"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        downloaded_file = f"{save_path}/test_video.mp4"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.side_effect = KeyError("No proxy configured")
        mock_get_format.return_value = "best[height<=1080]"

        # Mock subprocess
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout.readline = MagicMock(
            side_effect=[
                f"[download] Destination: {downloaded_file}\n",
                "",  # End of output
            ]
        )
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        mock_exists.return_value = True

        with patch("core._1_ytdlp.os.utime") as mock_utime:
            with patch("core._1_ytdlp.rprint") as mock_rprint:
                result = download_via_command(url, save_path, "1080")

                assert result == downloaded_file
                mock_rprint.assert_called_with(
                    f"[green]Download completed successfully: {os.path.basename(downloaded_file)}[/green]"
                )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.subprocess.Popen")
    def test_download_via_command_with_proxy(
        self,
        mock_popen,
        mock_get_format,
        mock_validate_proxy,
        mock_load_key,
        mock_get_paths,
    ):
        """Test command-line download with proxy configuration"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        proxy_url = "http://proxy.example.com:8080"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = proxy_url
        mock_validate_proxy.return_value = True
        mock_get_format.return_value = "best[height<=1080]"

        # Mock subprocess failure to check command construction
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout.readline = MagicMock(return_value="")
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        with pytest.raises(Exception):
            download_via_command(url, save_path, "1080")

        # Verify proxy was added to command
        call_args = mock_popen.call_args[0][0]  # Get the command arguments
        assert "--proxy" in call_args
        assert proxy_url in call_args

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.environ.get")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.subprocess.Popen")
    def test_download_via_command_environment_proxy(
        self,
        mock_popen,
        mock_get_format,
        mock_validate_proxy,
        mock_env_get,
        mock_load_key,
        mock_get_paths,
    ):
        """Test command-line download with environment proxy"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        proxy_url = "http://env-proxy.example.com:8080"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.side_effect = Exception("No proxy in config")
        mock_env_get.side_effect = [proxy_url, None, None, None]  # HTTPS_PROXY set
        mock_validate_proxy.return_value = True
        mock_get_format.return_value = "best[height<=1080]"

        # Mock subprocess failure to check command construction
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout.readline = MagicMock(return_value="")
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        with pytest.raises(Exception):
            download_via_command(url, save_path, "1080")

        # Verify proxy was added to command
        call_args = mock_popen.call_args[0][0]
        assert "--proxy" in call_args
        assert proxy_url in call_args

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.subprocess.Popen")
    def test_download_via_command_process_failure(
        self, mock_popen, mock_get_format, mock_load_key, mock_get_paths
    ):
        """Test command-line download process failure"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.side_effect = KeyError("No proxy configured")
        mock_get_format.return_value = "best[height<=1080]"

        # Mock subprocess failure
        mock_process = MagicMock()
        mock_process.returncode = 1
        output_lines = ["ERROR: Video not available", "Failed to download"]
        mock_process.stdout.readline = MagicMock(side_effect=output_lines + [""])
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        with pytest.raises(Exception) as exc_info:
            download_via_command(url, save_path, "1080")

        assert "yt-dlp command failed" in str(exc_info.value)

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.subprocess.Popen")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    def test_download_via_command_fallback_search(
        self,
        mock_find_recent,
        mock_exists,
        mock_popen,
        mock_get_format,
        mock_load_key,
        mock_get_paths,
    ):
        """Test fallback file search when path detection fails"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        fallback_file = f"{save_path}/fallback_video.mp4"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.side_effect = KeyError("No proxy configured")
        mock_get_format.return_value = "best[height<=1080]"

        # Mock subprocess success but no path detected
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout.readline = MagicMock(
            side_effect=[
                "[download] 100% of some_video.mp4",  # No clear path pattern
                "",
            ]
        )
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        mock_exists.side_effect = [False, True]  # Path not detected, fallback found
        mock_find_recent.return_value = fallback_file

        with patch("core._1_ytdlp.os.utime") as mock_utime:
            with patch("core._1_ytdlp.rprint") as mock_rprint:
                result = download_via_command(url, save_path, "1080")

                assert result == fallback_file
                mock_rprint.assert_any_call(
                    "[yellow]Could not determine file path from output, searching directory...[/yellow]"
                )


class TestDownloadViaPythonApi:
    """Test Python API download functionality"""

    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.os.environ.get")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core._1_ytdlp.os.path.exists")
    def test_download_via_python_api_success(
        self,
        mock_exists,
        mock_update_ytdlp,
        mock_get_format,
        mock_env_get,
        mock_validate_proxy,
        mock_load_key,
    ):
        """Test successful Python API download"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        downloaded_file = f"{save_path}/test_video.mp4"

        # Setup mocks
        mock_load_key.side_effect = [
            None,  # cookies_path
            None,  # proxy
        ]
        mock_env_get.return_value = None
        mock_get_format.return_value = "best[height<=1080]"

        # Mock YoutubeDL
        mock_ytdl_class = MagicMock()
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__ = MagicMock(
            return_value=mock_ytdl_instance
        )
        mock_ytdl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_update_ytdlp.return_value = mock_ytdl_class

        mock_exists.return_value = True

        # Mock progress callback behavior
        def simulate_progress(progress_callback):
            if progress_callback:
                # Simulate downloading progress
                progress_callback(
                    {
                        "status": "downloading",
                        "downloaded_bytes": 50 * 1024 * 1024,
                        "total_bytes": 100 * 1024 * 1024,
                        "speed": 1024 * 1024,
                        "eta": 50,
                    }
                )
                # Simulate finished
                progress_callback({"status": "finished", "filename": downloaded_file})

        with patch("core._1_ytdlp.os.utime") as mock_utime:
            with patch("core._1_ytdlp.rprint") as mock_rprint:
                # Mock the download method to simulate progress hooks
                def mock_download(urls):
                    # Simulate calling progress hook
                    ydl_opts = mock_ytdl_class.call_args[0][0]
                    if "progress_hooks" in ydl_opts and ydl_opts["progress_hooks"]:
                        progress_hook = ydl_opts["progress_hooks"][0]
                        progress_hook(
                            {
                                "status": "downloading",
                                "downloaded_bytes": 50 * 1024 * 1024,
                                "total_bytes": 100 * 1024 * 1024,
                                "speed": 1024 * 1024,
                                "eta": 50,
                            }
                        )
                        progress_hook(
                            {"status": "finished", "filename": downloaded_file}
                        )

                mock_ytdl_instance.download = mock_download

                result = download_via_python_api(
                    url, save_path, "1080", progress_callback=simulate_progress
                )

                assert result == downloaded_file
                mock_rprint.assert_called_with(
                    f"[green]Download completed successfully: {os.path.basename(downloaded_file)}[/green]"
                )

    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.update_ytdlp")
    def test_download_via_python_api_with_proxy(
        self, mock_update_ytdlp, mock_get_format, mock_validate_proxy, mock_load_key
    ):
        """Test Python API download with proxy"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        proxy_url = "http://proxy.example.com:8080"

        # Setup mocks
        mock_load_key.side_effect = [
            None,  # cookies_path
            proxy_url,  # proxy
        ]
        mock_validate_proxy.return_value = True
        mock_get_format.return_value = "best[height<=1080]"

        # Mock YoutubeDL
        mock_ytdl_class = MagicMock()
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__ = MagicMock(
            return_value=mock_ytdl_instance
        )
        mock_ytdl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ytdl_instance.download.side_effect = Exception(
            "Download failed for proxy test"
        )
        mock_update_ytdlp.return_value = mock_ytdl_class

        with pytest.raises(Exception):
            download_via_python_api(url, save_path, "1080")

        # Verify proxy was set in options
        ydl_opts = mock_ytdl_class.call_args[0][0]
        assert ydl_opts["proxy"] == proxy_url

    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.os.environ.get")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.update_ytdlp")
    def test_download_via_python_api_environment_proxy(
        self,
        mock_update_ytdlp,
        mock_get_format,
        mock_env_get,
        mock_validate_proxy,
        mock_load_key,
    ):
        """Test Python API download with environment proxy"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        proxy_url = "http://env-proxy.example.com:8080"

        # Setup mocks
        mock_load_key.side_effect = [
            None,  # cookies_path
            KeyError("No proxy configured"),  # proxy
        ]
        mock_env_get.side_effect = [proxy_url, None, None, None]  # HTTPS_PROXY set
        mock_validate_proxy.return_value = True
        mock_get_format.return_value = "best[height<=1080]"

        # Mock YoutubeDL
        mock_ytdl_class = MagicMock()
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__ = MagicMock(
            return_value=mock_ytdl_instance
        )
        mock_ytdl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ytdl_instance.download.side_effect = Exception(
            "Download failed for env proxy test"
        )
        mock_update_ytdlp.return_value = mock_ytdl_class

        with pytest.raises(Exception):
            download_via_python_api(url, save_path, "1080")

        # Verify proxy was set in options
        ydl_opts = mock_ytdl_class.call_args[0][0]
        assert ydl_opts["proxy"] == proxy_url

    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.update_ytdlp")
    def test_download_via_python_api_invalid_proxy_warning(
        self, mock_update_ytdlp, mock_get_format, mock_validate_proxy, mock_load_key
    ):
        """Test invalid proxy URL warning"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        invalid_proxy = "invalid-proxy-url"

        # Setup mocks
        mock_load_key.side_effect = [
            None,  # cookies_path
            invalid_proxy,  # proxy
        ]
        mock_validate_proxy.return_value = False
        mock_get_format.return_value = "best[height<=1080]"

        # Mock YoutubeDL
        mock_ytdl_class = MagicMock()
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__ = MagicMock(
            return_value=mock_ytdl_instance
        )
        mock_ytdl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ytdl_instance.download.side_effect = Exception(
            "Download failed for invalid proxy test"
        )
        mock_update_ytdlp.return_value = mock_ytdl_class

        with patch("core._1_ytdlp.rprint") as mock_rprint:
            with pytest.raises(Exception):
                download_via_python_api(url, save_path, "1080")

            mock_rprint.assert_any_call(
                f"[yellow]Warning: Invalid proxy URL format: {invalid_proxy}[/yellow]"
            )

    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.os.path.exists")
    @patch("core._1_ytdlp.get_optimal_format")
    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    def test_download_via_python_api_fallback_search(
        self,
        mock_find_recent,
        mock_update_ytdlp,
        mock_get_format,
        mock_exists,
        mock_load_key,
    ):
        """Test fallback file search when path holder is empty"""
        url = "https://www.youtube.com/watch?v=test"
        save_path = "/test/downloads"
        fallback_file = f"{save_path}/fallback_video.mp4"

        # Setup mocks
        mock_load_key.side_effect = [
            None,  # cookies_path
            KeyError("No proxy configured"),  # proxy
        ]
        mock_get_format.return_value = "best[height<=1080]"

        # Mock YoutubeDL
        mock_ytdl_class = MagicMock()
        mock_ytdl_instance = MagicMock()
        mock_ytdl_class.return_value.__enter__ = MagicMock(
            return_value=mock_ytdl_instance
        )
        mock_ytdl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_update_ytdlp.return_value = mock_ytdl_class

        # Mock download without setting filename in holder
        mock_ytdl_instance.download = MagicMock()

        mock_exists.return_value = False  # Path holder is empty/invalid
        mock_find_recent.return_value = fallback_file

        with patch("core._1_ytdlp.rprint") as mock_rprint:
            with patch("core._1_ytdlp.os.utime"):
                result = download_via_python_api(url, save_path, "1080")

                assert result == fallback_file
                mock_rprint.assert_any_call(
                    "[yellow]Could not determine file path from Python API, searching directory...[/yellow]"
                )


class TestFindVideoFiles:
    """Test finding video files functionality"""

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.sys.platform", "win32")
    def test_find_video_files_single_file(
        self, mock_isfile, mock_glob, mock_load_key, mock_get_paths
    ):
        """Test finding single video file"""
        save_path = "/test/downloads"
        video_file = f"{save_path}/test_video.mp4"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = [video_file.replace("/", "\\")]  # Windows path
        mock_isfile.return_value = True

        result = find_video_files()

        assert result == video_file  # Should be converted back to forward slashes

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.find_best_video_file")
    @patch("core._1_ytdlp.os.path.getsize")
    @patch("core._1_ytdlp.rprint")
    def test_find_video_files_multiple_files(
        self,
        mock_rprint,
        mock_getsize,
        mock_find_best,
        mock_isfile,
        mock_glob,
        mock_load_key,
        mock_get_paths,
    ):
        """Test finding multiple video files"""
        save_path = "/test/downloads"
        video_files = [
            f"{save_path}/video1.mp4",
            f"{save_path}/video2.mkv",
            f"{save_path}/video3.avi",
        ]
        best_file = video_files[1]  # video2.mkv

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_find_best.return_value = best_file
        mock_getsize.return_value = 100 * 1024 * 1024  # 100MB

        result = find_video_files()

        assert result == best_file
        mock_rprint.assert_any_call(
            f"[yellow]Multiple video files found (3). Selecting the best one...[/yellow]"
        )
        mock_rprint.assert_any_call(
            f"[green]Selected video file: {os.path.basename(best_file)} (100.0MB)[/green]"
        )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    def test_find_video_files_no_files_found(
        self, mock_glob, mock_load_key, mock_get_paths
    ):
        """Test when no video files are found"""
        save_path = "/test/downloads"

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = []

        with pytest.raises(ValueError) as exc_info:
            find_video_files()

        assert "No video files found" in str(exc_info.value)

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    @patch("core._1_ytdlp.find_best_video_file")
    @patch("core._1_ytdlp.validate_video_file")
    @patch("core._1_ytdlp.os.path.getsize")
    @patch("core._1_ytdlp.rprint")
    def test_find_video_files_no_valid_files(
        self,
        mock_rprint,
        mock_getsize,
        mock_validate,
        mock_find_best,
        mock_isfile,
        mock_glob,
        mock_load_key,
        mock_get_paths,
    ):
        """Test when no valid video files are found"""
        save_path = "/test/downloads"
        video_files = [f"{save_path}/video1.mp4", f"{save_path}/video2.mkv"]

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = video_files
        mock_isfile.return_value = True
        mock_find_best.return_value = None  # No valid files
        mock_validate.return_value = (False, "Invalid file")
        mock_getsize.return_value = 5 * 1024 * 1024  # 5MB

        with pytest.raises(ValueError) as exc_info:
            find_video_files()

        assert "No valid video files found after validation" in str(exc_info.value)
        mock_rprint.assert_any_call(
            "[red]No valid video files found. File details:[/red]"
        )

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    def test_find_video_files_filter_extensions(
        self, mock_isfile, mock_glob, mock_load_key, mock_get_paths
    ):
        """Test filtering by allowed extensions"""
        save_path = "/test/downloads"
        all_files = [
            f"{save_path}/video1.mp4",  # Valid
            f"{save_path}/video2.avi",  # Invalid extension (not in allowed)
            f"{save_path}/audio.mp3",  # Invalid extension
            f"{save_path}/image.jpg",  # Should be filtered out
            f"{save_path}/subtitle.txt",  # Should be filtered out
            f"{save_path}/valid.mkv",  # Valid
        ]

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv"]  # Only mp4 and mkv allowed
        mock_glob.return_value = all_files
        mock_isfile.return_value = True

        with patch("core._1_ytdlp.find_best_video_file") as mock_find_best:
            mock_find_best.return_value = f"{save_path}/valid.mkv"
            with patch("core._1_ytdlp.os.path.getsize") as mock_getsize:
                mock_getsize.return_value = 100 * 1024 * 1024

                result = find_video_files()

                # Should find the best among valid files
                assert result == f"{save_path}/valid.mkv"

    @patch("core._1_ytdlp.get_storage_paths")
    @patch("core._1_ytdlp.load_key")
    @patch("core._1_ytdlp.glob.glob")
    @patch("core._1_ytdlp.os.path.isfile")
    def test_find_video_files_filter_output_prefixes(
        self, mock_isfile, mock_glob, mock_load_key, mock_get_paths
    ):
        """Test filtering out output files from old structure"""
        save_path = "/test/downloads"
        all_files = [
            f"{save_path}/video1.mp4",  # Valid
            f"output/output_final.mp4",  # Should be filtered out
            f"/output/processed.mp4",  # Should be filtered out
            f"{save_path}/normal_video.mkv",  # Valid
        ]

        # Setup mocks
        mock_get_paths.return_value = {"input": save_path}
        mock_load_key.return_value = ["mp4", "mkv", "avi"]
        mock_glob.return_value = all_files
        mock_isfile.return_value = True

        with patch("core._1_ytdlp.find_best_video_file") as mock_find_best:
            mock_find_best.return_value = f"{save_path}/normal_video.mkv"
            with patch("core._1_ytdlp.os.path.getsize") as mock_getsize:
                mock_getsize.return_value = 100 * 1024 * 1024

                result = find_video_files()

                # Should find the best among non-output files
                assert result == f"{save_path}/normal_video.mkv"
