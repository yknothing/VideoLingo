# Comprehensive Test Coverage for core/_1_ytdlp.py
# This module is the foundation of VideoLingo's video processing pipeline
# Target: 85%+ branch coverage for this critical 854-line module

import pytest
import os
import glob
import subprocess
import sys
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call, mock_open

try:
    # Import the module under test
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
        MODULAR_COMPONENTS_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestUpdateYtdlp:
    """Test yt-dlp update functionality"""

    @patch('subprocess.check_call')
    @patch('sys.modules', {'yt_dlp': Mock()})
    @patch('core._1_ytdlp.rprint')
    def test_update_ytdlp_success(self, mock_rprint, mock_subprocess):
        """Test successful yt-dlp update"""
        mock_subprocess.return_value = None
        mock_ytdl_class = Mock()
        
        with patch('yt_dlp.YoutubeDL', mock_ytdl_class):
            result = update_ytdlp()
        
        mock_subprocess.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
        )
        mock_rprint.assert_called_with("[green]yt-dlp updated[/green]")
        assert result == mock_ytdl_class

    @patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip'))
    @patch('core._1_ytdlp.rprint')
    def test_update_ytdlp_failure(self, mock_rprint, mock_subprocess):
        """Test yt-dlp update failure handling"""
        mock_ytdl_class = Mock()
        
        with patch('yt_dlp.YoutubeDL', mock_ytdl_class):
            result = update_ytdlp()
        
        mock_rprint.assert_called_with("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
        assert result == mock_ytdl_class

    @patch('subprocess.check_call')
    @patch('sys.modules', {})
    def test_update_ytdlp_no_existing_module(self, mock_subprocess):
        """Test update when yt_dlp not in sys.modules"""
        mock_ytdl_class = Mock()
        
        with patch('yt_dlp.YoutubeDL', mock_ytdl_class):
            result = update_ytdlp()
        
        assert result == mock_ytdl_class


class TestVideoFileValidation:
    """Test video file validation functionality"""

    def test_validate_video_file_modular_success(self, temp_config_dir):
        """Test validation with modular components available"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        test_file = temp_config_dir / "test_video.mp4"
        test_file.write_bytes(b"fake video content" * 100000)  # 1.6MB file
        
        with patch('core.download.VideoFileValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.validate_video_file.return_value = (True, "Valid file")
            mock_validator_class.return_value = mock_validator
            
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            assert is_valid == True
            assert error_msg == "Valid file"
            mock_validator_class.assert_called_once_with(min_size_mb=1)

    def test_validate_video_file_legacy_success(self, temp_config_dir):
        """Test legacy validation for valid file"""
        test_file = temp_config_dir / "test_video.mp4"
        test_file.write_bytes(b"fake video content" * 100000)  # 1.6MB file
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            assert is_valid == True
            assert error_msg == "File is valid"

    def test_validate_video_file_not_exists(self):
        """Test validation when file doesn't exist"""
        is_valid, error_msg = validate_video_file("/nonexistent/file.mp4", 1)
        
        assert is_valid == False
        assert "File does not exist" in error_msg

    def test_validate_video_file_too_small(self, temp_config_dir):
        """Test validation when file is too small"""
        test_file = temp_config_dir / "small_video.mp4"
        test_file.write_bytes(b"tiny")
        
        is_valid, error_msg = validate_video_file(str(test_file), 1)
        
        assert is_valid == False
        assert "File too small" in error_msg

    def test_validate_video_file_partial_download(self, temp_config_dir):
        """Test validation rejects partial downloads"""
        test_cases = [".part", ".tmp", ".download"]
        
        for extension in test_cases:
            test_file = temp_config_dir / f"video{extension}"
            test_file.write_bytes(b"fake content" * 100000)
            
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            assert is_valid == False
            assert "partial download" in error_msg

    def test_validate_video_file_ffprobe_failure(self, temp_config_dir):
        """Test validation when ffprobe fails"""
        test_file = temp_config_dir / "corrupted_video.mp4"
        test_file.write_bytes(b"fake content" * 100000)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            assert is_valid == False
            assert "File format validation failed" in error_msg

    def test_validate_video_file_ffprobe_timeout(self, temp_config_dir):
        """Test validation when ffprobe times out"""
        test_file = temp_config_dir / "video.mp4"
        test_file.write_bytes(b"fake content" * 100000)
        
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('ffprobe', 10)):
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            # Should skip detailed validation and pass
            assert is_valid == True
            assert error_msg == "File is valid"

    def test_validate_video_file_ffprobe_not_found(self, temp_config_dir):
        """Test validation when ffprobe not available"""
        test_file = temp_config_dir / "video.mp4"
        test_file.write_bytes(b"fake content" * 100000)
        
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            is_valid, error_msg = validate_video_file(str(test_file), 1)
            
            # Should skip detailed validation and pass
            assert is_valid == True
            assert error_msg == "File is valid"


class TestPartialDownloadCleanup:
    """Test cleanup of partial downloads"""

    def test_cleanup_partial_downloads_modular(self, temp_config_dir):
        """Test cleanup with modular components"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        with patch('core.download.PartialDownloadCleaner') as mock_cleaner_class:
            mock_cleaner = Mock()
            mock_cleaner.cleanup_partial_downloads.return_value = ["file1.part", "file2.tmp"]
            mock_cleaner_class.return_value = mock_cleaner
            
            result = cleanup_partial_downloads(str(temp_config_dir))
            
            assert result == ["file1.part", "file2.tmp"]
            mock_cleaner_class.assert_called_once()

    @patch('core._1_ytdlp.rprint')
    def test_cleanup_partial_downloads_legacy(self, mock_rprint, temp_config_dir):
        """Test legacy cleanup functionality"""
        # Create test partial files
        partial_files = [
            temp_config_dir / "video.part",
            temp_config_dir / "audio.tmp", 
            temp_config_dir / "merge.download",
            temp_config_dir / "fragment.f142",
            temp_config_dir / "temp.ytdl"
        ]
        
        for file_path in partial_files:
            file_path.write_bytes(b"partial content")
        
        result = cleanup_partial_downloads(str(temp_config_dir))
        
        # All partial files should be cleaned up
        expected_cleaned = [f.name for f in partial_files]
        assert set(result) == set(expected_cleaned)
        
        # Verify files are actually removed
        for file_path in partial_files:
            assert not file_path.exists()

    @patch('os.remove', side_effect=OSError("Permission denied"))
    @patch('core._1_ytdlp.rprint')
    def test_cleanup_partial_downloads_error_handling(self, mock_rprint, mock_remove, temp_config_dir):
        """Test error handling during cleanup"""
        test_file = temp_config_dir / "stuck.part"
        test_file.write_bytes(b"content")
        
        result = cleanup_partial_downloads(str(temp_config_dir))
        
        # Should handle errors gracefully
        assert result == []  # No files successfully cleaned
        mock_rprint.assert_called()
        assert "Warning: Could not remove" in str(mock_rprint.call_args)

    def test_cleanup_partial_downloads_no_files(self, temp_config_dir):
        """Test cleanup when no partial files exist"""
        result = cleanup_partial_downloads(str(temp_config_dir))
        
        assert result == []


class TestVideoFileFinding:
    """Test video file discovery functionality"""

    @patch('core.utils.config_utils.load_key', return_value=['mp4', 'avi', 'mkv'])
    def test_find_most_recent_video_file_success(self, mock_load_key, temp_config_dir):
        """Test finding most recent video file"""
        # Create test video files with different timestamps
        old_video = temp_config_dir / "old_video.mp4"
        old_video.write_bytes(b"video content" * 100000)
        
        time.sleep(0.1)  # Ensure different timestamps
        
        recent_video = temp_config_dir / "recent_video.mp4"  
        recent_video.write_bytes(b"video content" * 100000)
        
        # Mock validation to pass for both files
        with patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid")):
            result = find_most_recent_video_file(str(temp_config_dir))
            
            assert result == str(recent_video)

    @patch('core.utils.config_utils.load_key', return_value=['mp4', 'avi'])
    def test_find_most_recent_video_file_modular(self, mock_load_key, temp_config_dir):
        """Test modular file finding if available"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        expected_file = str(temp_config_dir / "video.mp4")
        
        with patch('core.download.VideoFileValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.find_most_recent_video_file.return_value = expected_file
            mock_validator_class.return_value = mock_validator
            
            result = find_most_recent_video_file(str(temp_config_dir))
            
            assert result == expected_file

    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_most_recent_video_file_none_found(self, mock_load_key, temp_config_dir):
        """Test when no video files found"""
        result = find_most_recent_video_file(str(temp_config_dir))
        
        assert result is None

    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_most_recent_video_file_invalid_files(self, mock_load_key, temp_config_dir):
        """Test when all files fail validation"""
        test_video = temp_config_dir / "invalid.mp4"
        test_video.write_bytes(b"content")
        
        with patch('core._1_ytdlp.validate_video_file', return_value=(False, "Invalid")):
            result = find_most_recent_video_file(str(temp_config_dir))
            
            assert result is None

    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_most_recent_video_file_skip_non_video(self, mock_load_key, temp_config_dir):
        """Test skipping non-video files"""
        # Create non-video files that should be skipped
        (temp_config_dir / "image.jpg").write_bytes(b"image")
        (temp_config_dir / "subtitle.txt").write_bytes(b"subs")
        (temp_config_dir / "metadata.json").write_bytes(b"{}")
        (temp_config_dir / "thumbnail.png").write_bytes(b"thumb")
        (temp_config_dir / "partial.part").write_bytes(b"partial")
        
        result = find_most_recent_video_file(str(temp_config_dir))
        
        assert result is None


class TestBestVideoFileFinding:
    """Test best video file selection"""

    def test_find_best_video_file_modular(self, temp_config_dir):
        """Test modular best file finding"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        allowed_formats = ['mp4', 'avi']
        expected_file = str(temp_config_dir / "best_video.mp4")
        
        with patch('core.download.VideoFileValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.find_best_video_file.return_value = expected_file
            mock_validator_class.return_value = mock_validator
            
            result = find_best_video_file(str(temp_config_dir), allowed_formats)
            
            assert result == expected_file

    @patch('core._1_ytdlp.rprint')
    def test_find_best_video_file_legacy_success(self, mock_rprint, temp_config_dir):
        """Test legacy best file selection - chooses largest valid file"""
        allowed_formats = ['mp4', 'avi']
        
        # Create files of different sizes
        small_video = temp_config_dir / "small.mp4"
        small_video.write_bytes(b"content" * 1000)
        
        large_video = temp_config_dir / "large.mp4"  
        large_video.write_bytes(b"content" * 10000)
        
        with patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid")):
            result = find_best_video_file(str(temp_config_dir), allowed_formats)
            
            assert result == str(large_video)

    def test_find_best_video_file_invalid_files_logged(self, temp_config_dir):
        """Test logging of invalid files during selection"""
        allowed_formats = ['mp4']
        
        invalid_video = temp_config_dir / "invalid.mp4"
        invalid_video.write_bytes(b"content")
        
        with patch('core._1_ytdlp.validate_video_file', return_value=(False, "Corrupted file")), \
             patch('core._1_ytdlp.rprint') as mock_rprint:
            
            result = find_best_video_file(str(temp_config_dir), allowed_formats)
            
            assert result is None
            mock_rprint.assert_called_with(
                "[yellow]Skipping invalid file invalid.mp4: Corrupted file[/yellow]"
            )

    def test_find_best_video_file_wrong_format(self, temp_config_dir):
        """Test skipping files with wrong format"""
        allowed_formats = ['mp4']
        
        wrong_format = temp_config_dir / "video.avi"
        wrong_format.write_bytes(b"content")
        
        result = find_best_video_file(str(temp_config_dir), allowed_formats)
        
        assert result is None


class TestOptimalFormatSelection:
    """Test video format optimization"""

    def test_get_optimal_format_modular(self):
        """Test modular format selection"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        with patch('core.download.VideoFormatSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector.get_optimal_format.return_value = "optimized_format_string"
            mock_selector_class.return_value = mock_selector
            
            result = get_optimal_format("1080")
            
            assert result == "optimized_format_string"
            mock_selector.get_optimal_format.assert_called_once_with("1080")

    def test_get_optimal_format_best_quality(self):
        """Test optimal format for best quality"""
        result = get_optimal_format("best")
        
        # Should contain prioritized H.264 formats
        assert "bestvideo[height<=2160][vcodec^=avc1]" in result
        assert "bestvideo[height<=1080][vcodec^=h264]" in result
        assert "bestaudio[acodec^=mp4a]" in result

    def test_get_optimal_format_specific_resolution(self):
        """Test optimal format for specific resolution"""
        result = get_optimal_format("720")
        
        # Should contain resolution-specific filters
        assert "[height<=720]" in result
        assert "bestvideo" in result
        assert "bestaudio" in result
        assert "[vcodec^=avc1]" in result

    def test_get_optimal_format_various_resolutions(self):
        """Test format selection for various resolutions"""
        test_resolutions = ["360", "480", "720", "1080", "1440", "2160"]
        
        for resolution in test_resolutions:
            result = get_optimal_format(resolution)
            
            # Should contain the resolution filter
            assert f"[height<={resolution}]" in result
            # Should prioritize H.264 codec
            assert "vcodec^=avc1" in result or "vcodec^=h264" in result


class TestErrorCategorization:
    """Test download error categorization and retry logic"""

    def test_categorize_download_error_modular(self):
        """Test modular error categorization"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        error_msg = "Connection timeout"
        
        with patch('core.download.DownloadErrorHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_category = Mock()
            mock_category.value = "network"
            mock_handler.categorize_download_error.return_value = (mock_category, True, 30)
            mock_handler_class.return_value = mock_handler
            
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "network"
            assert is_retryable == True
            assert wait_time == 30

    def test_categorize_download_error_network_issues(self):
        """Test network error categorization"""
        network_errors = [
            "network timeout occurred",
            "connection failed",
            "temporary network issue",
            "timeout expired"
        ]
        
        for error_msg in network_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "network"
            assert is_retryable == True
            assert wait_time == 30

    def test_categorize_download_error_rate_limiting(self):
        """Test rate limit error categorization"""
        rate_limit_errors = [
            "403 Forbidden",
            "429 Too Many Requests", 
            "rate limit exceeded",
            "too many requests per minute"
        ]
        
        for error_msg in rate_limit_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "rate_limit"
            assert is_retryable == True
            assert wait_time == 60

    def test_categorize_download_error_not_found(self):
        """Test not found error categorization"""
        not_found_errors = [
            "404 Not Found",
            "video not found",
            "content unavailable"
        ]
        
        for error_msg in not_found_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "not_found" 
            assert is_retryable == False
            assert wait_time == 0

    def test_categorize_download_error_access_denied(self):
        """Test access denied error categorization"""
        access_errors = [
            "401 Unauthorized",
            "private video",
            "restricted content"
        ]
        
        for error_msg in access_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "access_denied"
            assert is_retryable == False
            assert wait_time == 0

    def test_categorize_download_error_proxy_ssl(self):
        """Test proxy/SSL error categorization"""
        proxy_ssl_errors = [
            "proxy connection failed",
            "SSL handshake error",
            "certificate verification failed"
        ]
        
        for error_msg in proxy_ssl_errors:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            
            assert category == "proxy_ssl"
            assert is_retryable == True
            assert wait_time == 10

    def test_categorize_download_error_unknown(self):
        """Test unknown error categorization"""
        category, is_retryable, wait_time = categorize_download_error("unknown error type")
        
        assert category == "unknown"
        assert is_retryable == True
        assert wait_time == 20


class TestIntelligentRetry:
    """Test intelligent retry mechanism"""

    def test_intelligent_retry_modular(self):
        """Test modular retry mechanism"""
        if not MODULAR_COMPONENTS_AVAILABLE:
            pytest.skip("Modular components not available")
        
        download_func = Mock(return_value="success")
        
        with patch('core.download.DownloadErrorHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler.intelligent_retry_download.return_value = "success"
            mock_handler_class.return_value = mock_handler
            
            result = intelligent_retry_download(download_func, max_retries=3)
            
            assert result == "success"

    @patch('time.sleep')
    @patch('core._1_ytdlp.rprint')
    def test_intelligent_retry_success_first_attempt(self, mock_rprint, mock_sleep):
        """Test successful download on first attempt"""
        download_func = Mock(return_value="downloaded_file.mp4")
        
        result = intelligent_retry_download(download_func, max_retries=3)
        
        assert result == "downloaded_file.mp4"
        download_func.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    @patch('core._1_ytdlp.rprint')
    @patch('core._1_ytdlp.categorize_download_error', return_value=("network", True, 30))
    def test_intelligent_retry_success_after_retries(self, mock_categorize, mock_rprint, mock_sleep):
        """Test successful download after retries"""
        download_func = Mock(side_effect=[
            Exception("network timeout"),
            Exception("connection failed"), 
            "downloaded_file.mp4"
        ])
        
        result = intelligent_retry_download(download_func, max_retries=3)
        
        assert result == "downloaded_file.mp4"
        assert download_func.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('time.sleep')
    @patch('core._1_ytdlp.rprint')
    @patch('core._1_ytdlp.categorize_download_error', return_value=("network", True, 30))
    def test_intelligent_retry_max_retries_exceeded(self, mock_categorize, mock_rprint, mock_sleep):
        """Test max retries exceeded"""
        error = Exception("persistent network error")
        download_func = Mock(side_effect=error)
        
        with pytest.raises(Exception, match="persistent network error"):
            intelligent_retry_download(download_func, max_retries=2)
        
        assert download_func.call_count == 3  # Initial + 2 retries
        mock_rprint.assert_called_with(
            "[red]Download failed after 2 retries: persistent network error[/red]"
        )

    @patch('core._1_ytdlp.rprint')
    @patch('core._1_ytdlp.categorize_download_error', return_value=("not_found", False, 0))
    def test_intelligent_retry_non_retryable_error(self, mock_categorize, mock_rprint):
        """Test non-retryable error handling"""
        error = Exception("404 video not found")
        download_func = Mock(side_effect=error)
        
        with pytest.raises(Exception, match="404 video not found"):
            intelligent_retry_download(download_func, max_retries=3)
        
        download_func.assert_called_once()  # No retries for non-retryable errors
        mock_rprint.assert_called_with(
            "[red]Non-retryable error (not_found): 404 video not found[/red]"
        )

    @patch('time.sleep')
    @patch('core._1_ytdlp.rprint') 
    @patch('core._1_ytdlp.categorize_download_error')
    def test_intelligent_retry_wait_time_calculation(self, mock_categorize, mock_rprint, mock_sleep):
        """Test wait time calculation with exponential backoff"""
        mock_categorize.return_value = ("network", True, 100)  # Suggested wait: 100s
        error = Exception("network error")
        download_func = Mock(side_effect=[error, error, error, error])
        
        with pytest.raises(Exception):
            intelligent_retry_download(download_func, max_retries=3, initial_wait=5)
        
        # Verify sleep calls with exponential backoff capped by suggested wait
        expected_waits = [5, 10, 20]  # min(initial_wait * 2^attempt, suggested_wait)
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_waits == expected_waits


class TestDownloadViaCommand:
    """Test command-line yt-dlp download functionality"""

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key')
    @patch('subprocess.Popen')
    @patch('core._1_ytdlp.get_optimal_format', return_value="best[height<=1080]")
    @patch('core._1_ytdlp.validate_proxy_url', return_value=True)
    @patch('core._1_ytdlp.rprint')
    def test_download_via_command_success(self, mock_rprint, mock_validate_proxy, mock_format, mock_popen, mock_load_key, mock_paths):
        """Test successful command-line download"""
        mock_load_key.side_effect = lambda key: {
            'youtube.proxy': 'http://proxy:8080'
        }.get(key, None)
        
        # Mock process output
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "[download] Destination: /test/input/video.mp4\n",
            "[download] 100% of 50.0MiB at 2.5MiB/s ETA 00:00\n",
            ""  # End of output
        ]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # Mock file existence and timestamp update
        with patch('os.path.exists', return_value=True), \
             patch('time.time', return_value=1234567890), \
             patch('os.utime') as mock_utime:
            
            result = download_via_command(
                "https://www.youtube.com/watch?v=test", 
                "/test/input", 
                "1080"
            )
            
            assert result == "/test/input/video.mp4"
            mock_utime.assert_called_once()

    @patch('subprocess.Popen')
    def test_download_via_command_with_progress_callback(self, mock_popen):
        """Test command-line download with progress callback"""
        progress_data = []
        
        def progress_callback(data):
            progress_data.append(data)
        
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "[download]  45.2% of  117.60MiB at    2.34MiB/s ETA 00:32\n",
            "[download] 100% of 117.60MiB at 2.34MiB/s ETA 00:00\n", 
            ""
        ]
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            download_via_command(
                "https://www.youtube.com/watch?v=test",
                progress_callback=progress_callback
            )
        
        # Should have received progress updates
        assert len(progress_data) >= 1
        assert progress_data[0]["status"] == "downloading"
        assert 0 <= progress_data[0]["progress"] <= 1

    @patch('subprocess.Popen')
    def test_download_via_command_process_failure(self, mock_popen):
        """Test command process failure handling"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "ERROR: Video unavailable\n",
            ""
        ]
        mock_process.wait.return_value = None
        mock_process.returncode = 1  # Error return code
        mock_popen.return_value = mock_process
        
        with pytest.raises(Exception, match="yt-dlp command failed"):
            download_via_command("https://invalid.url")

    @patch('core.utils.config_utils.load_key', side_effect=KeyError())
    @patch('os.environ.get', return_value='http://env-proxy:8080')
    @patch('core._1_ytdlp.validate_proxy_url', return_value=True)
    @patch('subprocess.Popen')
    def test_download_via_command_environment_proxy(self, mock_popen, mock_validate, mock_env, mock_load_key):
        """Test using environment proxy when config proxy not available"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [""]
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'):
            
            download_via_command("https://www.youtube.com/watch?v=test")
        
        # Verify proxy was added to command
        call_args = mock_popen.call_args[0][0]
        assert "--proxy" in call_args
        assert "http://env-proxy:8080" in call_args

    @patch('subprocess.Popen')
    def test_download_via_command_fallback_file_detection(self, mock_popen):
        """Test fallback file detection when output parsing fails"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "[download] Starting download...\n",  # No destination pattern
            ""
        ]
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/fallback/video.mp4") as mock_find, \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            result = download_via_command("https://www.youtube.com/watch?v=test")
            
            assert result == "/fallback/video.mp4"
            mock_find.assert_called_once()

    @patch('subprocess.Popen')
    def test_download_via_command_no_file_found(self, mock_popen):
        """Test error when no file found after download"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [""]
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value=None), \
             patch('core._1_ytdlp.rprint'):
            
            with pytest.raises(Exception, match="could not locate the downloaded file"):
                download_via_command("https://www.youtube.com/watch?v=test")


class TestDownloadViaPythonAPI:
    """Test Python API yt-dlp download functionality"""

    @patch('core.utils.config_utils.load_key')
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('yt_dlp.YoutubeDL')
    @patch('core._1_ytdlp.validate_proxy_url', return_value=True)
    def test_download_via_python_api_success(self, mock_validate_proxy, mock_ytdl_class, mock_update, mock_load_key):
        """Test successful Python API download"""
        mock_load_key.side_effect = lambda key: {
            'youtube.cookies_path': '/test/cookies.txt',
            'youtube.proxy': 'http://proxy:8080'
        }.get(key, None)
        
        # Mock YoutubeDL instance
        mock_ytdl = Mock()
        mock_ytdl_class.return_value.__enter__ = Mock(return_value=mock_ytdl)
        mock_ytdl_class.return_value.__exit__ = Mock(return_value=False)
        
        with patch('os.path.exists', return_value=True), \
             patch('time.time', return_value=1234567890), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            # Test without progress callback first
            result = download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input", 
                "1080"
            )
            
            mock_ytdl.download.assert_called_once_with(["https://www.youtube.com/watch?v=test"])

    def test_download_via_python_api_progress_callback(self):
        """Test Python API download with progress callback"""
        progress_data = []
        
        def progress_callback(data):
            progress_data.append(data)
        
        with patch('core.utils.config_utils.load_key', return_value=None), \
             patch('core._1_ytdlp.update_ytdlp'), \
             patch('yt_dlp.YoutubeDL') as mock_ytdl_class, \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            # Mock YoutubeDL to trigger progress hook
            mock_ytdl = Mock()
            mock_ytdl_class.return_value.__enter__ = Mock(return_value=mock_ytdl)
            mock_ytdl_class.return_value.__exit__ = Mock(return_value=False)
            
            # Capture the progress hook function
            def side_effect(opts):
                if 'progress_hooks' in opts and opts['progress_hooks']:
                    hook = opts['progress_hooks'][0]
                    # Simulate download progress
                    hook({
                        'status': 'downloading',
                        'downloaded_bytes': 50 * 1024 * 1024,
                        'total_bytes': 100 * 1024 * 1024,
                        'speed': 2 * 1024 * 1024,
                        'eta': 25
                    })
                    # Simulate completion
                    hook({
                        'status': 'finished',
                        'filename': '/test/downloaded.mp4'
                    })
                return Mock()
            
            mock_ytdl_class.side_effect = side_effect
            
            download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input",
                "1080", 
                progress_callback
            )
            
            # Should have received progress updates
            assert len(progress_data) >= 1

    @patch('core.utils.config_utils.load_key', return_value=None)
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('yt_dlp.YoutubeDL')
    def test_download_via_python_api_cookie_handling(self, mock_ytdl_class, mock_update, mock_load_key):
        """Test cookie handling in Python API"""
        mock_ytdl = Mock()
        mock_ytdl_class.return_value.__enter__ = Mock(return_value=mock_ytdl)
        mock_ytdl_class.return_value.__exit__ = Mock(return_value=False)
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input",
                "1080"
            )
            
            # Verify YoutubeDL was called with browser cookies by default
            ytdl_opts = mock_ytdl_class.call_args[0][0]
            assert 'cookiesfrombrowser' in ytdl_opts
            assert ytdl_opts['cookiesfrombrowser'] == ('chrome',)

    @patch('core.utils.config_utils.load_key')
    @patch('core._1_ytdlp.update_ytdlp') 
    @patch('yt_dlp.YoutubeDL')
    def test_download_via_python_api_cookie_file_fallback(self, mock_ytdl_class, mock_update, mock_load_key):
        """Test fallback to cookie file when browser cookies fail"""
        mock_load_key.side_effect = lambda key: {
            'youtube.cookies_path': '/test/cookies.txt'
        }.get(key, None)
        
        # Mock YoutubeDL to raise exception for browser cookies
        def ytdl_side_effect(opts):
            if 'cookiesfrombrowser' in opts:
                raise Exception("Browser cookies failed")
            return Mock()
        
        mock_ytdl_class.side_effect = ytdl_side_effect
        
        with patch('os.path.exists', return_value=True), \
             patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            # Should not raise exception, should fallback gracefully
            result = download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input", 
                "1080"
            )
            
            assert result == "/test/video.mp4"

    @patch('core.utils.config_utils.load_key', return_value=None)
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('yt_dlp.YoutubeDL')
    def test_download_via_python_api_exception_handling(self, mock_ytdl_class, mock_update, mock_load_key):
        """Test exception handling in Python API"""
        mock_ytdl = Mock()
        mock_ytdl.download.side_effect = Exception("Download failed")
        mock_ytdl_class.return_value.__enter__ = Mock(return_value=mock_ytdl)
        mock_ytdl_class.return_value.__exit__ = Mock(return_value=False)
        
        with pytest.raises(Exception, match="Download failed"):
            download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input",
                "1080"
            )

    @patch('core.utils.config_utils.load_key', return_value=None)
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('yt_dlp.YoutubeDL')
    def test_download_via_python_api_fallback_file_search(self, mock_ytdl_class, mock_update, mock_load_key):
        """Test fallback file search when filename not captured"""
        mock_ytdl = Mock()
        mock_ytdl_class.return_value.__enter__ = Mock(return_value=mock_ytdl)
        mock_ytdl_class.return_value.__exit__ = Mock(return_value=False)
        
        with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/fallback/video.mp4") as mock_find, \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            result = download_via_python_api(
                "https://www.youtube.com/watch?v=test",
                "/test/input",
                "1080"
            )
            
            assert result == "/fallback/video.mp4"
            mock_find.assert_called_once()


class TestMainDownloadFunction:
    """Test main download_video_ytdlp function"""

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs')
    @patch('core.utils.config_utils.load_key', return_value="1080")
    @patch('os.makedirs')
    @patch('shutil.disk_usage')
    @patch('core._1_ytdlp.intelligent_retry_download')
    @patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid"))
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=50*1024*1024)  # 50MB
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_success(self, mock_rprint, mock_getsize, mock_exists, 
                                         mock_validate, mock_retry, mock_disk_usage,
                                         mock_makedirs, mock_load_key, mock_ensure_dirs, mock_paths):
        """Test successful main download function"""
        # Mock disk usage - plenty of space
        mock_disk_usage.return_value.free = 10 * 1024 * 1024 * 1024  # 10GB
        
        # Mock successful download
        mock_retry.return_value = "/test/input/video.mp4"
        
        result = download_video_ytdlp("https://www.youtube.com/watch?v=test")
        
        assert result == "/test/input/video.mp4"
        mock_ensure_dirs.assert_called_once()
        mock_makedirs.assert_called_once_with("/test/input", exist_ok=True)
        mock_retry.assert_called_once()

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs')
    @patch('core.utils.config_utils.load_key', return_value="720")
    @patch('os.makedirs')
    @patch('shutil.disk_usage')
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_low_disk_space_warning(self, mock_rprint, mock_disk_usage, 
                                                        mock_makedirs, mock_load_key, 
                                                        mock_ensure_dirs, mock_paths):
        """Test low disk space warning"""
        # Mock low disk space
        mock_disk_usage.return_value.free = 512 * 1024 * 1024  # 512MB (< 1GB)
        
        with patch('core._1_ytdlp.intelligent_retry_download', return_value="/test/video.mp4"), \
             patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid")), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=50*1024*1024):
            
            download_video_ytdlp("https://www.youtube.com/watch?v=test")
            
            # Should have warned about low disk space
            warning_calls = [call for call in mock_rprint.call_args_list 
                           if "Low disk space" in str(call)]
            assert len(warning_calls) > 0

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs')
    @patch('core.utils.config_utils.load_key', return_value="1080")
    @patch('os.makedirs')
    @patch('shutil.disk_usage', side_effect=Exception("Cannot check disk space"))
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_disk_check_error(self, mock_rprint, mock_disk_usage,
                                                   mock_makedirs, mock_load_key,
                                                   mock_ensure_dirs, mock_paths):
        """Test handling disk space check errors"""
        with patch('core._1_ytdlp.intelligent_retry_download', return_value="/test/video.mp4"), \
             patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid")), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=50*1024*1024):
            
            download_video_ytdlp("https://www.youtube.com/watch?v=test")
            
            # Should have logged disk check error
            error_calls = [call for call in mock_rprint.call_args_list 
                          if "Could not check disk space" in str(call)]
            assert len(error_calls) > 0

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs')
    @patch('core.utils.config_utils.load_key', return_value="1080")
    @patch('os.makedirs')
    @patch('shutil.disk_usage')
    @patch('core._1_ytdlp.intelligent_retry_download')
    @patch('core._1_ytdlp.find_most_recent_video_file', return_value="/fallback/video.mp4")
    @patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid"))
    @patch('os.path.exists')
    @patch('os.path.getsize', return_value=50*1024*1024)
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_fallback_file_search(self, mock_rprint, mock_getsize, mock_exists,
                                                      mock_validate, mock_find_recent, mock_retry,
                                                      mock_disk_usage, mock_makedirs, mock_load_key,
                                                      mock_ensure_dirs, mock_paths):
        """Test fallback file search when download returns invalid path"""
        mock_disk_usage.return_value.free = 10 * 1024 * 1024 * 1024
        
        # Mock download returns None/invalid path
        mock_retry.return_value = None
        
        # Mock file existence check - first call False (for None), second True (for fallback)
        mock_exists.side_effect = [False, True, True]
        
        result = download_video_ytdlp("https://www.youtube.com/watch?v=test")
        
        assert result == "/fallback/video.mp4"
        mock_find_recent.assert_called_once()

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs')
    @patch('core.utils.config_utils.load_key', return_value="1080")
    @patch('os.makedirs')
    @patch('shutil.disk_usage')
    @patch('core._1_ytdlp.intelligent_retry_download', return_value=None)
    @patch('core._1_ytdlp.find_most_recent_video_file', return_value=None)
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_no_file_found(self, mock_rprint, mock_find_recent, mock_retry,
                                               mock_disk_usage, mock_makedirs, mock_load_key,
                                               mock_ensure_dirs, mock_paths):
        """Test error when no file found after download"""
        mock_disk_usage.return_value.free = 10 * 1024 * 1024 * 1024
        
        with pytest.raises(Exception, match="No video file found after download completion"):
            download_video_ytdlp("https://www.youtube.com/watch?v=test")

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.ensure_storage_dirs') 
    @patch('core.utils.config_utils.load_key', return_value="1080")
    @patch('os.makedirs')
    @patch('shutil.disk_usage')
    @patch('core._1_ytdlp.intelligent_retry_download', return_value="/test/video.mp4")
    @patch('core._1_ytdlp.validate_video_file', return_value=(False, "Corrupted file"))
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=50*1024*1024)
    @patch('core._1_ytdlp.rprint')
    def test_download_video_ytdlp_validation_warning(self, mock_rprint, mock_getsize, mock_exists,
                                                    mock_validate, mock_retry, mock_disk_usage,
                                                    mock_makedirs, mock_load_key, mock_ensure_dirs, mock_paths):
        """Test validation warning for corrupted file"""
        mock_disk_usage.return_value.free = 10 * 1024 * 1024 * 1024
        
        result = download_video_ytdlp("https://www.youtube.com/watch?v=test")
        
        # Should still return the file but log warning
        assert result == "/test/video.mp4"
        warning_calls = [call for call in mock_rprint.call_args_list 
                        if "validation failed" in str(call)]
        assert len(warning_calls) > 0


class TestFindVideoFiles:
    """Test video file discovery functionality"""

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4', 'avi', 'mkv'])
    @patch('sys.platform', 'linux')
    def test_find_video_files_single_file(self, mock_load_key, mock_paths, temp_config_dir):
        """Test finding single video file"""
        video_file = temp_config_dir / "video.mp4"
        video_file.write_bytes(b"video content")
        
        with patch('glob.glob', return_value=[str(video_file)]), \
             patch('os.path.isfile', return_value=True):
            
            result = find_video_files(str(temp_config_dir))
            
            assert result == str(video_file).replace("\\", "/")

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4', 'avi'])
    @patch('sys.platform', 'win32')
    def test_find_video_files_windows_path_normalization(self, mock_load_key, mock_paths, temp_config_dir):
        """Test Windows path normalization"""
        video_file = temp_config_dir / "video.mp4"
        video_file.write_bytes(b"video content")
        
        with patch('glob.glob', return_value=[str(video_file).replace("/", "\\")]), \
             patch('os.path.isfile', return_value=True):
            
            result = find_video_files(str(temp_config_dir))
            
            # Should normalize backslashes to forward slashes
            assert "\\" not in result

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_video_files_multiple_files_selects_best(self, mock_load_key, mock_paths, temp_config_dir):
        """Test selecting best file when multiple found"""
        video1 = temp_config_dir / "video1.mp4"
        video2 = temp_config_dir / "video2.mp4" 
        video1.write_bytes(b"content")
        video2.write_bytes(b"content")
        
        with patch('glob.glob', return_value=[str(video1), str(video2)]), \
             patch('os.path.isfile', return_value=True), \
             patch('core._1_ytdlp.find_best_video_file', return_value=str(video2)), \
             patch('os.path.getsize', return_value=1024), \
             patch('core._1_ytdlp.rprint'):
            
            result = find_video_files(str(temp_config_dir))
            
            assert result == str(video2).replace("\\", "/")

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_video_files_no_files_found(self, mock_load_key, mock_paths):
        """Test error when no video files found"""
        with patch('glob.glob', return_value=[]):
            
            with pytest.raises(ValueError, match="No video files found"):
                find_video_files("/empty/directory")

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    @patch('core._1_ytdlp.rprint')
    def test_find_video_files_no_valid_files_detailed_error(self, mock_rprint, mock_load_key, mock_paths, temp_config_dir):
        """Test detailed error when no valid files found"""
        invalid_video = temp_config_dir / "invalid.mp4"
        invalid_video.write_bytes(b"content")
        
        with patch('glob.glob', return_value=[str(invalid_video)]), \
             patch('os.path.isfile', return_value=True), \
             patch('core._1_ytdlp.find_best_video_file', return_value=None), \
             patch('core._1_ytdlp.validate_video_file', return_value=(False, "Corrupted")), \
             patch('os.path.getsize', return_value=1024):
            
            with pytest.raises(ValueError, match="No valid video files found after validation"):
                find_video_files(str(temp_config_dir))
            
            # Should have logged file details
            detail_calls = [call for call in mock_rprint.call_args_list 
                           if "File details:" in str(call)]
            assert len(detail_calls) > 0

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_video_files_filter_output_files(self, mock_load_key, mock_paths, temp_config_dir):
        """Test filtering out output files from old structure"""
        good_video = temp_config_dir / "input_video.mp4"
        output_video1 = temp_config_dir / "output/output_video.mp4"  
        output_video2 = temp_config_dir / "/output_something.mp4"
        
        good_video.write_bytes(b"content")
        
        with patch('glob.glob', return_value=[str(good_video), str(output_video1), str(output_video2)]), \
             patch('os.path.isfile', return_value=True):
            
            result = find_video_files(str(temp_config_dir))
            
            assert result == str(good_video).replace("\\", "/")

    @patch('core.utils.config_utils.get_storage_paths', return_value={'input': '/test/input'})
    @patch('core.utils.config_utils.load_key', return_value=['mp4'])
    def test_find_video_files_skip_non_video_extensions(self, mock_load_key, mock_paths, temp_config_dir):
        """Test skipping files with non-video extensions"""
        video_file = temp_config_dir / "video.mp4"
        image_file = temp_config_dir / "image.jpg"
        text_file = temp_config_dir / "readme.txt"
        
        video_file.write_bytes(b"video")
        image_file.write_bytes(b"image") 
        text_file.write_bytes(b"text")
        
        with patch('glob.glob', return_value=[str(video_file), str(image_file), str(text_file)]), \
             patch('os.path.isfile', return_value=True):
            
            result = find_video_files(str(temp_config_dir))
            
            assert result == str(video_file).replace("\\", "/")


class TestProgressCallbackIntegration:
    """Test progress callback functionality across different download methods"""

    def test_command_download_progress_parsing(self):
        """Test progress parsing in command download"""
        from core._1_ytdlp import download_via_command
        
        progress_data = []
        def progress_callback(data):
            progress_data.append(data)
        
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.stdout.readline.side_effect = [
                "[download]  25.5% of  200.0MiB at    5.2MiB/s ETA 01:30\n",
                "[download]  50.0% of  200.0MiB at    5.0MiB/s ETA 01:00\n", 
                "[download] 100.0% of  200.0MiB at    5.5MiB/s ETA 00:00\n",
                ""
            ]
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
                 patch('os.path.exists', return_value=True), \
                 patch('os.utime'):
                
                download_via_command(
                    "https://test.url",
                    progress_callback=progress_callback
                )
        
        # Should have captured multiple progress updates
        assert len(progress_data) >= 2
        
        # Verify progress values are reasonable
        for data in progress_data[:-1]:  # Exclude final completion
            assert 0 <= data["progress"] <= 1
            assert data["status"] == "downloading"

    def test_python_api_progress_callback_data_structure(self):
        """Test progress callback data structure in Python API"""
        progress_data = []
        def progress_callback(data):
            progress_data.append(data)
        
        with patch('core.utils.config_utils.load_key', return_value=None), \
             patch('core._1_ytdlp.update_ytdlp'), \
             patch('yt_dlp.YoutubeDL') as mock_ytdl_class, \
             patch('os.path.exists', return_value=True), \
             patch('os.utime'), \
             patch('core._1_ytdlp.rprint'):
            
            # Setup YoutubeDL mock to call progress hook
            def ytdl_side_effect(opts):
                mock_ytdl = Mock()
                if 'progress_hooks' in opts and opts['progress_hooks']:
                    hook = opts['progress_hooks'][0]
                    # Test with slow download speed
                    hook({
                        'status': 'downloading',
                        'downloaded_bytes': 1 * 1024 * 1024,  # 1MB
                        'total_bytes': 100 * 1024 * 1024,     # 100MB
                        'speed': 50 * 1024,                   # 50KB/s (slow)
                        'eta': 2000
                    })
                mock_instance = Mock()
                mock_instance.__enter__ = Mock(return_value=mock_ytdl)
                mock_instance.__exit__ = Mock(return_value=False) 
                return mock_instance
            
            mock_ytdl_class.side_effect = ytdl_side_effect
            
            download_via_python_api(
                "https://test.url",
                "/test/input",
                "1080",
                progress_callback
            )
        
        # Should have received progress with warning about slow speed
        assert len(progress_data) >= 1
        data = progress_data[0]
        assert "warning" in data
        assert "Slow download speed" in data["warning"]
        assert data["speed_mbps"] < 0.1


class TestSecurityAndSanitization:
    """Test security features and path sanitization"""

    def test_command_download_path_sanitization(self):
        """Test path sanitization in command download"""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.stdout.readline.side_effect = [
                '[download] Destination: /test/../../../etc/passwd\n',
                ""
            ]
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            with patch('core._1_ytdlp.sanitize_path') as mock_sanitize, \
                 patch('os.path.exists', return_value=True), \
                 patch('os.utime'), \
                 patch('core._1_ytdlp.rprint'):
                
                mock_sanitize.return_value = "/safe/path/video.mp4"
                
                result = download_via_command("https://test.url", "/test/input")
                
                # Should have called sanitize_path
                mock_sanitize.assert_called()
                assert result == "/safe/path/video.mp4"

    def test_command_download_filename_sanitization_fallback(self):
        """Test filename sanitization fallback when path sanitization fails"""
        with patch('subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.stdout.readline.side_effect = [
                '[download] Destination: malicious<>filename.mp4\n',
                ""
            ]
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            with patch('core._1_ytdlp.sanitize_path', side_effect=Exception("Sanitization failed")), \
                 patch('core._1_ytdlp.sanitize_filename', return_value="safe_filename.mp4"), \
                 patch('os.path.exists', return_value=True), \
                 patch('os.utime'), \
                 patch('core._1_ytdlp.rprint'):
                
                result = download_via_command("https://test.url", "/test/input")
                
                # Should fallback to filename sanitization
                assert "safe_filename.mp4" in result

    def test_proxy_url_validation(self):
        """Test proxy URL validation in downloads"""
        with patch('core.utils.config_utils.load_key', return_value='invalid-proxy-url'), \
             patch('core._1_ytdlp.validate_proxy_url', return_value=False), \
             patch('core._1_ytdlp.rprint') as mock_rprint, \
             patch('subprocess.Popen') as mock_popen:
            
            mock_process = Mock()
            mock_process.returncode = 0
            mock_process.stdout.readline.side_effect = [""]
            mock_popen.return_value = mock_process
            
            with patch('core._1_ytdlp.find_most_recent_video_file', return_value="/test/video.mp4"), \
                 patch('os.path.exists', return_value=True), \
                 patch('os.utime'):
                
                download_via_command("https://test.url")
                
                # Should have warned about invalid proxy
                warning_calls = [call for call in mock_rprint.call_args_list 
                               if "Invalid proxy URL format" in str(call)]
                assert len(warning_calls) > 0


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/unit/test_ytdlp_comprehensive.py -v
    pytest.main([__file__, "-v", "--tb=short"])
