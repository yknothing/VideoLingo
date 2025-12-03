"""
Comprehensive test suite for core/_1_ytdlp.py module
Target: 85%+ branch coverage for video download functionality
"""

import pytest
import os
import tempfile
import shutil
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import json
import time
import sys
from contextlib import contextmanager

# Import the module under test
try:
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
except ImportError as e:
    pytest.skip(f"Could not import ytdlp module: {e}", allow_module_level=True)


class TestUpdateYtdlp:
    """Test yt-dlp update functionality"""

    @patch("subprocess.check_call")
    @patch("sys.modules", {"yt_dlp": MagicMock()})
    def test_update_ytdlp_success(self, mock_modules, mock_check_call):
        """Test successful yt-dlp update"""
        mock_check_call.return_value = None

        with patch("core._1_ytdlp.YoutubeDL") as mock_ytdl:
            mock_ytdl_instance = MagicMock()
            mock_ytdl.return_value = mock_ytdl_instance

            result = update_ytdlp()

            # Verify subprocess call
            mock_check_call.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
            )
            assert result == mock_ytdl

    @patch("subprocess.check_call")
    def test_update_ytdlp_subprocess_failure(self, mock_check_call):
        """Test yt-dlp update with subprocess failure"""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip")

        with patch("core._1_ytdlp.YoutubeDL") as mock_ytdl:
            result = update_ytdlp()
            assert result == mock_ytdl

    @patch("subprocess.check_call")
    def test_update_ytdlp_import_cleanup(self, mock_check_call):
        """Test module cleanup during update"""
        mock_check_call.return_value = None

        # Mock sys.modules with yt_dlp present
        original_modules = sys.modules.copy()
        sys.modules["yt_dlp"] = MagicMock()

        try:
            with patch("core._1_ytdlp.YoutubeDL") as mock_ytdl:
                update_ytdlp()
                # yt_dlp should be removed from modules
                assert "yt_dlp" not in sys.modules
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


class TestValidateVideoFile:
    """Test video file validation functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_video.mp4")

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_video_file_not_exists(self):
        """Test validation of non-existent file"""
        is_valid, message = validate_video_file("nonexistent.mp4")
        assert not is_valid
        assert "does not exist" in message

    def test_validate_video_file_too_small(self):
        """Test validation of file too small"""
        # Create a very small file
        with open(self.test_file, "wb") as f:
            f.write(b"small")

        is_valid, message = validate_video_file(self.test_file, expected_min_size_mb=1)
        assert not is_valid
        assert "too small" in message

    def test_validate_video_file_partial_download(self):
        """Test validation of partial download files"""
        partial_files = [
            os.path.join(self.temp_dir, "video.part"),
            os.path.join(self.temp_dir, "video.tmp"),
            os.path.join(self.temp_dir, "video.download"),
        ]

        for partial_file in partial_files:
            # Create file large enough
            with open(partial_file, "wb") as f:
                f.write(b"x" * (2 * 1024 * 1024))  # 2MB

            is_valid, message = validate_video_file(partial_file)
            assert not is_valid
            assert "partial download" in message

    @patch("subprocess.run")
    def test_validate_video_file_ffprobe_success(self, mock_run):
        """Test successful ffprobe validation"""
        # Create file large enough
        with open(self.test_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        # Mock successful ffprobe
        mock_run.return_value = Mock(returncode=0)

        is_valid, message = validate_video_file(self.test_file)
        assert is_valid
        assert message == "File is valid"

    @patch("subprocess.run")
    def test_validate_video_file_ffprobe_failure(self, mock_run):
        """Test ffprobe validation failure"""
        # Create file large enough
        with open(self.test_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        # Mock failed ffprobe
        mock_run.return_value = Mock(returncode=1)

        is_valid, message = validate_video_file(self.test_file)
        assert not is_valid
        assert "format validation failed" in message

    @patch("subprocess.run")
    def test_validate_video_file_ffprobe_timeout(self, mock_run):
        """Test ffprobe timeout handling"""
        # Create file large enough
        with open(self.test_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 10)

        is_valid, message = validate_video_file(self.test_file)
        assert is_valid  # Should skip detailed validation and pass
        assert message == "File is valid"

    @patch("subprocess.run")
    def test_validate_video_file_ffprobe_not_found(self, mock_run):
        """Test missing ffprobe handling"""
        # Create file large enough
        with open(self.test_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        # Mock ffprobe not found
        mock_run.side_effect = FileNotFoundError()

        is_valid, message = validate_video_file(self.test_file)
        assert is_valid  # Should skip detailed validation and pass
        assert message == "File is valid"

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.VideoFileValidator")
    def test_validate_video_file_modular_success(self, mock_validator_class):
        """Test modular validator success"""
        mock_validator = Mock()
        mock_validator.validate_video_file.return_value = (
            True,
            "Modular validation passed",
        )
        mock_validator_class.return_value = mock_validator

        is_valid, message = validate_video_file(self.test_file, expected_min_size_mb=2)

        assert is_valid
        assert message == "Modular validation passed"
        mock_validator_class.assert_called_once_with(min_size_mb=2)

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.VideoFileValidator")
    def test_validate_video_file_modular_fallback(self, mock_validator_class):
        """Test modular validator fallback to legacy"""
        # Mock modular validator to raise exception
        mock_validator_class.side_effect = Exception("Modular validator failed")

        # Create file large enough for legacy validation
        with open(self.test_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            is_valid, message = validate_video_file(self.test_file)
            assert is_valid
            assert message == "File is valid"


class TestCleanupPartialDownloads:
    """Test partial download cleanup functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cleanup_partial_downloads_no_files(self):
        """Test cleanup when no partial files exist"""
        cleaned = cleanup_partial_downloads(self.temp_dir)
        assert cleaned == []

    def test_cleanup_partial_downloads_success(self):
        """Test successful cleanup of partial files"""
        # Create test partial files
        partial_files = [
            os.path.join(self.temp_dir, "video.part"),
            os.path.join(self.temp_dir, "video.tmp"),
            os.path.join(self.temp_dir, "video.download"),
            os.path.join(self.temp_dir, "video.f123"),
            os.path.join(self.temp_dir, "video.ytdl"),
        ]

        for file_path in partial_files:
            with open(file_path, "w") as f:
                f.write("partial content")

        cleaned = cleanup_partial_downloads(self.temp_dir)

        # All files should be cleaned
        assert len(cleaned) == len(partial_files)

        # Files should no longer exist
        for file_path in partial_files:
            assert not os.path.exists(file_path)

    def test_cleanup_partial_downloads_permission_error(self):
        """Test cleanup with permission errors"""
        partial_file = os.path.join(self.temp_dir, "video.part")

        # Create file
        with open(partial_file, "w") as f:
            f.write("content")

        # Mock os.remove to raise OSError
        with patch("os.remove") as mock_remove:
            mock_remove.side_effect = OSError("Permission denied")

            cleaned = cleanup_partial_downloads(self.temp_dir)
            assert cleaned == []  # No files should be reported as cleaned

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.PartialDownloadCleaner")
    def test_cleanup_partial_downloads_modular_success(self, mock_cleaner_class):
        """Test modular cleaner success"""
        mock_cleaner = Mock()
        mock_cleaner.cleanup_partial_downloads.return_value = [
            "file1.part",
            "file2.tmp",
        ]
        mock_cleaner_class.return_value = mock_cleaner

        cleaned = cleanup_partial_downloads(self.temp_dir)

        assert cleaned == ["file1.part", "file2.tmp"]
        mock_cleaner.cleanup_partial_downloads.assert_called_once_with(self.temp_dir)

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.PartialDownloadCleaner")
    def test_cleanup_partial_downloads_modular_fallback(self, mock_cleaner_class):
        """Test modular cleaner fallback to legacy"""
        # Mock modular cleaner to raise exception
        mock_cleaner_class.side_effect = Exception("Modular cleaner failed")

        # Create test file for legacy cleanup
        partial_file = os.path.join(self.temp_dir, "video.part")
        with open(partial_file, "w") as f:
            f.write("content")

        cleaned = cleanup_partial_downloads(self.temp_dir)
        assert "video.part" in cleaned
        assert not os.path.exists(partial_file)


class TestFindMostRecentVideoFile:
    """Test finding most recent video file"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core.utils.config_utils.load_key")
    def test_find_most_recent_video_file_no_files(self, mock_load_key):
        """Test when no video files exist"""
        mock_load_key.return_value = ["mp4", "avi", "mkv"]

        result = find_most_recent_video_file(self.temp_dir)
        assert result is None

    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_most_recent_video_file_single_valid(
        self, mock_validate, mock_load_key
    ):
        """Test with single valid video file"""
        mock_load_key.return_value = ["mp4", "avi", "mkv"]
        mock_validate.return_value = (True, "Valid")

        # Create test video file
        video_file = os.path.join(self.temp_dir, "test.mp4")
        with open(video_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB

        result = find_most_recent_video_file(self.temp_dir)
        assert result == video_file

    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_most_recent_video_file_multiple_files(
        self, mock_validate, mock_load_key
    ):
        """Test with multiple video files, should return most recent"""
        mock_load_key.return_value = ["mp4", "avi", "mkv"]
        mock_validate.return_value = (True, "Valid")

        # Create test video files with different timestamps
        old_file = os.path.join(self.temp_dir, "old.mp4")
        new_file = os.path.join(self.temp_dir, "new.mp4")

        # Create old file
        with open(old_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))

        # Wait a bit and create new file
        time.sleep(0.1)
        with open(new_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))

        result = find_most_recent_video_file(self.temp_dir)
        assert result == new_file

    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_most_recent_video_file_skip_invalid(
        self, mock_validate, mock_load_key
    ):
        """Test skipping invalid video files"""
        mock_load_key.return_value = ["mp4", "avi", "mkv"]

        # Create test files
        valid_file = os.path.join(self.temp_dir, "valid.mp4")
        invalid_file = os.path.join(self.temp_dir, "invalid.mp4")

        with open(valid_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))
        with open(invalid_file, "wb") as f:
            f.write(b"x" * (2 * 1024 * 1024))

        # Make invalid_file newer but invalid
        time.sleep(0.1)
        os.utime(invalid_file)

        # Mock validation: invalid_file is invalid, valid_file is valid
        def validate_side_effect(file_path):
            if "invalid" in file_path:
                return (False, "Invalid file")
            return (True, "Valid")

        mock_validate.side_effect = validate_side_effect

        result = find_most_recent_video_file(self.temp_dir)
        assert result == valid_file

    @patch("core.utils.config_utils.load_key")
    def test_find_most_recent_video_file_skip_non_video_extensions(self, mock_load_key):
        """Test skipping non-video file extensions"""
        mock_load_key.return_value = ["mp4", "avi"]

        # Create files with different extensions
        video_file = os.path.join(self.temp_dir, "video.mp4")
        text_file = os.path.join(self.temp_dir, "readme.txt")
        image_file = os.path.join(self.temp_dir, "thumb.jpg")

        for file_path in [video_file, text_file, image_file]:
            with open(file_path, "wb") as f:
                f.write(b"x" * (2 * 1024 * 1024))

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (True, "Valid")

            result = find_most_recent_video_file(self.temp_dir)
            assert result == video_file

    @patch("core.utils.config_utils.load_key")
    def test_find_most_recent_video_file_skip_excluded_extensions(self, mock_load_key):
        """Test skipping explicitly excluded file extensions"""
        mock_load_key.return_value = [
            "mp4",
            "jpg",
        ]  # jpg is allowed but should be excluded

        # Create files
        video_file = os.path.join(self.temp_dir, "video.mp4")
        jpg_file = os.path.join(self.temp_dir, "image.jpg")

        for file_path in [video_file, jpg_file]:
            with open(file_path, "wb") as f:
                f.write(b"x" * (2 * 1024 * 1024))

        # Make jpg file newer
        time.sleep(0.1)
        os.utime(jpg_file)

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (True, "Valid")

            result = find_most_recent_video_file(self.temp_dir)
            assert result == video_file  # Should skip jpg despite being newer

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.VideoFileValidator")
    def test_find_most_recent_video_file_modular_success(
        self, mock_validator_class, mock_load_key
    ):
        """Test modular validator success"""
        mock_load_key.return_value = ["mp4", "avi"]
        mock_validator = Mock()
        mock_validator.find_most_recent_video_file.return_value = "modular_result.mp4"
        mock_validator_class.return_value = mock_validator

        result = find_most_recent_video_file(self.temp_dir)

        assert result == "modular_result.mp4"
        mock_validator.find_most_recent_video_file.assert_called_once_with(
            self.temp_dir, ["mp4", "avi"]
        )


class TestGetOptimalFormat:
    """Test optimal format selection"""

    def test_get_optimal_format_best(self):
        """Test optimal format for 'best' resolution"""
        result = get_optimal_format("best")

        # Should contain H.264 preferences and fallbacks
        assert "avc1" in result
        assert "h264" in result
        assert "bestvideo" in result
        assert "bestaudio" in result

    def test_get_optimal_format_specific_resolution(self):
        """Test optimal format for specific resolution"""
        result = get_optimal_format("1080")

        # Should contain resolution filter and codec preferences
        assert "height<=1080" in result
        assert "avc1" in result
        assert "h264" in result
        assert "bestvideo" in result
        assert "bestaudio" in result

    def test_get_optimal_format_various_resolutions(self):
        """Test optimal format for various resolutions"""
        resolutions = ["720", "1080", "1440", "2160"]

        for res in resolutions:
            result = get_optimal_format(res)
            assert f"height<={res}" in result
            assert "avc1" in result or "h264" in result

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.VideoFormatSelector")
    def test_get_optimal_format_modular_success(self, mock_selector_class):
        """Test modular format selector success"""
        mock_selector = Mock()
        mock_selector.get_optimal_format.return_value = "modular_format_string"
        mock_selector_class.return_value = mock_selector

        result = get_optimal_format("1080")

        assert result == "modular_format_string"
        mock_selector.get_optimal_format.assert_called_once_with("1080")

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.VideoFormatSelector")
    def test_get_optimal_format_modular_fallback(self, mock_selector_class):
        """Test modular format selector fallback to legacy"""
        # Mock modular selector to raise exception
        mock_selector_class.side_effect = Exception("Modular selector failed")

        result = get_optimal_format("1080")

        # Should fallback to legacy implementation
        assert "height<=1080" in result
        assert "avc1" in result


class TestCategorizeDownloadError:
    """Test download error categorization"""

    def test_categorize_download_error_network(self):
        """Test network error categorization"""
        network_errors = [
            "Network error occurred",
            "Connection timeout",
            "Temporary failure",
            "network unreachable",
        ]

        for error_msg in network_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "network"
            assert is_retryable is True
            assert wait_time == 30

    def test_categorize_download_error_rate_limit(self):
        """Test rate limit error categorization"""
        rate_limit_errors = [
            "HTTP Error 403: Forbidden",
            "HTTP Error 429: Too Many Requests",
            "Rate limit exceeded",
            "too many requests",
        ]

        for error_msg in rate_limit_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "rate_limit"
            assert is_retryable is True
            assert wait_time == 60

    def test_categorize_download_error_not_found(self):
        """Test not found error categorization"""
        not_found_errors = [
            "HTTP Error 404: Not Found",
            "Video not found",
            "Content unavailable",
        ]

        for error_msg in not_found_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "not_found"
            assert is_retryable is False
            assert wait_time == 0

    def test_categorize_download_error_access_denied(self):
        """Test access denied error categorization"""
        access_errors = [
            "HTTP Error 401: Unauthorized",
            "Private video",
            "Restricted content",
        ]

        for error_msg in access_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "access_denied"
            assert is_retryable is False
            assert wait_time == 0

    def test_categorize_download_error_proxy_ssl(self):
        """Test proxy/SSL error categorization"""
        proxy_ssl_errors = [
            "Proxy error",
            "SSL certificate verify failed",
            "Certificate verification error",
        ]

        for error_msg in proxy_ssl_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "proxy_ssl"
            assert is_retryable is True
            assert wait_time == 10

    def test_categorize_download_error_unknown(self):
        """Test unknown error categorization"""
        unknown_errors = [
            "Something went wrong",
            "Unexpected error occurred",
            "Random failure message",
        ]

        for error_msg in unknown_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == "unknown"
            assert is_retryable is True
            assert wait_time == 20

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.DownloadErrorHandler")
    def test_categorize_download_error_modular_success(self, mock_handler_class):
        """Test modular error handler success"""
        from enum import Enum

        class ErrorCategory(Enum):
            NETWORK = "network"

        mock_handler = Mock()
        mock_handler.categorize_download_error.return_value = (
            ErrorCategory.NETWORK,
            True,
            45,
        )
        mock_handler_class.return_value = mock_handler

        category, is_retryable, wait_time = categorize_download_error("test error")

        assert category == "network"
        assert is_retryable is True
        assert wait_time == 45


class TestIntelligentRetryDownload:
    """Test intelligent retry mechanism"""

    def test_intelligent_retry_download_success_first_attempt(self):
        """Test successful download on first attempt"""
        mock_download_func = Mock(return_value="downloaded_file.mp4")

        result = intelligent_retry_download(mock_download_func, max_retries=3)

        assert result == "downloaded_file.mp4"
        mock_download_func.assert_called_once()

    @patch("time.sleep")
    def test_intelligent_retry_download_success_after_retry(self, mock_sleep):
        """Test successful download after retry"""
        mock_download_func = Mock()
        mock_download_func.side_effect = [
            Exception("Network error"),
            "downloaded_file.mp4",
        ]

        with patch("core._1_ytdlp.categorize_download_error") as mock_categorize:
            mock_categorize.return_value = ("network", True, 30)

            result = intelligent_retry_download(
                mock_download_func, max_retries=3, initial_wait=5
            )

            assert result == "downloaded_file.mp4"
            assert mock_download_func.call_count == 2
            mock_sleep.assert_called_once_with(
                5
            )  # initial_wait * 2^0, capped by suggested_wait

    @patch("time.sleep")
    def test_intelligent_retry_download_max_retries_exceeded(self, mock_sleep):
        """Test max retries exceeded"""
        mock_download_func = Mock()
        mock_download_func.side_effect = Exception("Network error")

        with patch("core._1_ytdlp.categorize_download_error") as mock_categorize:
            mock_categorize.return_value = ("network", True, 30)

            with pytest.raises(Exception, match="Network error"):
                intelligent_retry_download(mock_download_func, max_retries=2)

            assert mock_download_func.call_count == 3  # initial + 2 retries

    def test_intelligent_retry_download_non_retryable_error(self):
        """Test non-retryable error handling"""
        mock_download_func = Mock()
        mock_download_func.side_effect = Exception("Video not found")

        with patch("core._1_ytdlp.categorize_download_error") as mock_categorize:
            mock_categorize.return_value = ("not_found", False, 0)

            with pytest.raises(Exception, match="Video not found"):
                intelligent_retry_download(mock_download_func, max_retries=3)

            mock_download_func.assert_called_once()  # No retries for non-retryable

    @patch("time.sleep")
    def test_intelligent_retry_download_exponential_backoff(self, mock_sleep):
        """Test exponential backoff timing"""
        mock_download_func = Mock()
        mock_download_func.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            Exception("Network error"),
            Exception("Network error"),  # Will exceed max_retries
        ]

        with patch("core._1_ytdlp.categorize_download_error") as mock_categorize:
            mock_categorize.return_value = ("network", True, 100)  # High suggested wait

            with pytest.raises(Exception):
                intelligent_retry_download(
                    mock_download_func, max_retries=3, initial_wait=2
                )

            # Check exponential backoff: min(initial_wait * 2^attempt, suggested_wait)
            expected_waits = [2, 4, 8]  # All less than suggested_wait=100
            mock_sleep.assert_has_calls([call(wait) for wait in expected_waits])

    @patch("time.sleep")
    def test_intelligent_retry_download_wait_time_capping(self, mock_sleep):
        """Test wait time is capped by suggested wait time"""
        mock_download_func = Mock()
        mock_download_func.side_effect = [
            Exception("Rate limit error"),
            Exception("Rate limit error"),
        ]

        with patch("core._1_ytdlp.categorize_download_error") as mock_categorize:
            mock_categorize.return_value = (
                "rate_limit",
                True,
                15,
            )  # Low suggested wait

            with pytest.raises(Exception):
                intelligent_retry_download(
                    mock_download_func, max_retries=1, initial_wait=20
                )

            # Should use suggested_wait (15) instead of initial_wait * 2^0 (20)
            mock_sleep.assert_called_once_with(15)

    @patch("core._1_ytdlp.MODULAR_COMPONENTS_AVAILABLE", True)
    @patch("core._1_ytdlp.DownloadErrorHandler")
    def test_intelligent_retry_download_modular_success(self, mock_handler_class):
        """Test modular error handler in retry logic"""
        mock_handler = Mock()
        mock_handler.intelligent_retry_download.return_value = "modular_result.mp4"
        mock_handler_class.return_value = mock_handler

        mock_download_func = Mock()

        result = intelligent_retry_download(
            mock_download_func, max_retries=3, initial_wait=5
        )

        assert result == "modular_result.mp4"
        mock_handler.intelligent_retry_download.assert_called_once_with(
            mock_download_func, 3, 5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
