"""
Comprehensive test suite for VideoLingo download functionality
Testing with 90%+ branch coverage and TDD approach
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the specific module directly to avoid torch conflicts
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ytdlp", 
    os.path.join(os.path.dirname(__file__), '..', '..', 'core', '_1_ytdlp.py')
)
ytdlp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ytdlp_module)

# Extract functions we need
download_video_ytdlp = ytdlp_module.download_video_ytdlp
download_via_python_api = ytdlp_module.download_via_python_api
download_via_command = ytdlp_module.download_via_command
validate_video_file = ytdlp_module.validate_video_file
cleanup_partial_downloads = ytdlp_module.cleanup_partial_downloads
find_most_recent_video_file = ytdlp_module.find_most_recent_video_file
find_best_video_file = ytdlp_module.find_best_video_file
get_optimal_format = ytdlp_module.get_optimal_format
categorize_download_error = ytdlp_module.categorize_download_error
intelligent_retry_download = ytdlp_module.intelligent_retry_download
sanitize_filename = ytdlp_module.sanitize_filename


class TestSanitizeFilename:
    """Test filename sanitization functionality"""
    
    def test_sanitize_filename_removes_illegal_characters(self):
        """Test that illegal characters are removed from filenames"""
        filename = 'video<>:"/\\|?*.mp4'
        result = sanitize_filename(filename)
        assert result == 'video.mp4'
        
    def test_sanitize_filename_strips_dots_spaces(self):
        """Test that leading/trailing dots and spaces are stripped"""
        filename = ' .video.mp4. '
        result = sanitize_filename(filename)
        assert result == 'video.mp4'
        
    def test_sanitize_filename_handles_empty_string(self):
        """Test that empty strings return default name"""
        filename = ''
        result = sanitize_filename(filename)
        assert result == 'video'
        
    def test_sanitize_filename_handles_only_illegal_chars(self):
        """Test that filenames with only illegal characters return default"""
        filename = '<>:"/\\|?*'
        result = sanitize_filename(filename)
        assert result == 'video'


class TestVideoFileValidation:
    """Test video file validation functionality"""
    
    def test_validate_video_file_nonexistent_file(self):
        """Test validation of non-existent file"""
        is_valid, error_msg = validate_video_file('/nonexistent/file.mp4')
        assert not is_valid
        assert 'does not exist' in error_msg
        
    def test_validate_video_file_too_small(self):
        """Test validation of file that's too small"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(b'small')
            tmp_path = tmp.name
            
        try:
            is_valid, error_msg = validate_video_file(tmp_path, expected_min_size_mb=1)
            assert not is_valid
            assert 'too small' in error_msg
        finally:
            os.unlink(tmp_path)
            
    def test_validate_video_file_partial_download(self):
        """Test validation of partial download files"""
        with tempfile.NamedTemporaryFile(suffix='.part', delete=False) as tmp:
            tmp.write(b'x' * (2 * 1024 * 1024))  # 2MB
            tmp_path = tmp.name
            
        try:
            is_valid, error_msg = validate_video_file(tmp_path)
            assert not is_valid
            assert 'partial download' in error_msg
        finally:
            os.unlink(tmp_path)
            
    def test_validate_video_file_valid_file(self):
        """Test validation of valid video file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(b'x' * (2 * 1024 * 1024))  # 2MB
            tmp_path = tmp.name
            
        try:
            is_valid, error_msg = validate_video_file(tmp_path)
            assert is_valid
            assert error_msg == 'File is valid'
        finally:
            os.unlink(tmp_path)


class TestPartialDownloadCleanup:
    """Test partial download cleanup functionality"""
    
    def test_cleanup_partial_downloads_removes_partial_files(self):
        """Test that partial files are removed during cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create partial files
            partial_files = ['video.part', 'video.tmp', 'video.download']
            for filename in partial_files:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write('partial content')
                    
            # Create normal file that should be preserved
            normal_file = os.path.join(temp_dir, 'video.mp4')
            with open(normal_file, 'w') as f:
                f.write('normal content')
                
            cleaned_files = cleanup_partial_downloads(temp_dir)
            
            # Check that partial files were removed
            assert len(cleaned_files) == 3
            for filename in partial_files:
                assert not os.path.exists(os.path.join(temp_dir, filename))
                
            # Check that normal file was preserved
            assert os.path.exists(normal_file)
            
    def test_cleanup_partial_downloads_empty_directory(self):
        """Test cleanup in empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_files = cleanup_partial_downloads(temp_dir)
            assert len(cleaned_files) == 0


class TestVideoFileFinding:
    """Test video file finding functionality"""
    
    @patch('core._1_ytdlp.load_key')
    def test_find_most_recent_video_file_single_file(self, mock_load_key):
        """Test finding most recent video file with single file"""
        mock_load_key.return_value = ['mp4', 'avi', 'mov']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid video file
            video_file = os.path.join(temp_dir, 'test.mp4')
            with open(video_file, 'wb') as f:
                f.write(b'x' * (2 * 1024 * 1024))  # 2MB
                
            result = find_most_recent_video_file(temp_dir)
            assert result == video_file
            
    @patch('core._1_ytdlp.load_key')
    def test_find_most_recent_video_file_multiple_files(self, mock_load_key):
        """Test finding most recent video file with multiple files"""
        mock_load_key.return_value = ['mp4', 'avi', 'mov']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple video files
            video1 = os.path.join(temp_dir, 'test1.mp4')
            video2 = os.path.join(temp_dir, 'test2.mp4')
            
            with open(video1, 'wb') as f:
                f.write(b'x' * (2 * 1024 * 1024))
            with open(video2, 'wb') as f:
                f.write(b'x' * (2 * 1024 * 1024))
                
            # Make video2 newer
            import time
            time.sleep(0.1)
            os.utime(video2)
            
            result = find_most_recent_video_file(temp_dir)
            assert result == video2
            
    @patch('core._1_ytdlp.load_key')
    def test_find_most_recent_video_file_no_files(self, mock_load_key):
        """Test finding video file when none exist"""
        mock_load_key.return_value = ['mp4', 'avi', 'mov']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = find_most_recent_video_file(temp_dir)
            assert result is None
            
    def test_find_best_video_file_largest_file(self):
        """Test finding best video file based on size"""
        allowed_formats = ['mp4', 'avi', 'mov']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files of different sizes
            small_file = os.path.join(temp_dir, 'small.mp4')
            large_file = os.path.join(temp_dir, 'large.mp4')
            
            with open(small_file, 'wb') as f:
                f.write(b'x' * (1 * 1024 * 1024))  # 1MB
            with open(large_file, 'wb') as f:
                f.write(b'x' * (5 * 1024 * 1024))  # 5MB
                
            result = find_best_video_file(temp_dir, allowed_formats)
            assert result == large_file


class TestFormatSelection:
    """Test video format selection functionality"""
    
    def test_get_optimal_format_best_quality(self):
        """Test optimal format selection for best quality"""
        result = get_optimal_format('best')
        assert 'bestvideo[height<=2160]' in result
        assert 'h264' in result
        assert 'avc1' in result
        
    def test_get_optimal_format_specific_resolution(self):
        """Test optimal format selection for specific resolution"""
        result = get_optimal_format('1080')
        assert '[height<=1080]' in result
        assert 'h264' in result
        assert 'bestaudio' in result
        
    def test_get_optimal_format_4k_resolution(self):
        """Test optimal format selection for 4K resolution"""
        result = get_optimal_format('2160')
        assert '[height<=2160]' in result
        assert 'h264' in result


class TestErrorCategorization:
    """Test download error categorization"""
    
    def test_categorize_download_error_network(self):
        """Test categorization of network errors"""
        error_msg = "Network timeout occurred"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "network"
        assert is_retryable is True
        assert wait_time == 30
        
    def test_categorize_download_error_rate_limit(self):
        """Test categorization of rate limit errors"""
        error_msg = "Too many requests, rate limit exceeded"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "rate_limit"
        assert is_retryable is True
        assert wait_time == 60
        
    def test_categorize_download_error_not_found(self):
        """Test categorization of not found errors"""
        error_msg = "Video not found or unavailable"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "not_found"
        assert is_retryable is False
        assert wait_time == 0
        
    def test_categorize_download_error_access_denied(self):
        """Test categorization of access denied errors"""
        error_msg = "Unauthorized access, video is private"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "access_denied"
        assert is_retryable is False
        assert wait_time == 0
        
    def test_categorize_download_error_unknown(self):
        """Test categorization of unknown errors"""
        error_msg = "Something went wrong"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "unknown"
        assert is_retryable is True
        assert wait_time == 20


class TestIntelligentRetry:
    """Test intelligent retry mechanism"""
    
    def test_intelligent_retry_download_success_first_try(self):
        """Test retry mechanism with successful first attempt"""
        mock_func = Mock(return_value="success")
        
        result = intelligent_retry_download(mock_func, max_retries=3)
        assert result == "success"
        assert mock_func.call_count == 1
        
    def test_intelligent_retry_download_success_after_retries(self):
        """Test retry mechanism with success after retries"""
        mock_func = Mock(side_effect=[
            Exception("Network timeout"),
            Exception("Network timeout"),
            "success"
        ])
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = intelligent_retry_download(mock_func, max_retries=3)
            assert result == "success"
            assert mock_func.call_count == 3
            
    def test_intelligent_retry_download_max_retries_exceeded(self):
        """Test retry mechanism when max retries exceeded"""
        mock_func = Mock(side_effect=Exception("Network timeout"))
        
        with patch('time.sleep'):
            with pytest.raises(Exception):
                intelligent_retry_download(mock_func, max_retries=2)
            assert mock_func.call_count == 3  # Initial + 2 retries
            
    def test_intelligent_retry_download_non_retryable_error(self):
        """Test retry mechanism with non-retryable error"""
        mock_func = Mock(side_effect=Exception("Video not found"))
        
        with pytest.raises(Exception):
            intelligent_retry_download(mock_func, max_retries=3)
        assert mock_func.call_count == 1  # Should not retry


class TestDownloadViaPythonAPI:
    """Test download via Python API functionality"""
    
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('core._1_ytdlp.load_key')
    @patch('core._1_ytdlp.find_most_recent_video_file')
    @patch('os.path.exists')
    @patch('os.utime')
    def test_download_via_python_api_success(self, mock_utime, mock_exists, 
                                           mock_find_recent, mock_load_key, 
                                           mock_update_ytdlp):
        """Test successful download via Python API"""
        # Setup mocks
        mock_load_key.return_value = None
        mock_youtubedl_class = Mock()
        mock_youtubedl_instance = Mock()
        mock_youtubedl_class.return_value.__enter__.return_value = mock_youtubedl_instance
        mock_update_ytdlp.return_value = mock_youtubedl_class
        mock_exists.return_value = True
        mock_find_recent.return_value = "/test/video.mp4"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_via_python_api(
                "https://example.com/video", 
                temp_dir, 
                "1080"
            )
            
            assert result == "/test/video.mp4"
            mock_youtubedl_instance.download.assert_called_once()
            mock_utime.assert_called_once()
            
    @patch('core._1_ytdlp.update_ytdlp')
    @patch('core._1_ytdlp.load_key')
    def test_download_via_python_api_download_failure(self, mock_load_key, mock_update_ytdlp):
        """Test download failure via Python API"""
        mock_load_key.return_value = None
        mock_youtubedl_class = Mock()
        mock_youtubedl_instance = Mock()
        mock_youtubedl_instance.download.side_effect = Exception("Download failed")
        mock_youtubedl_class.return_value.__enter__.return_value = mock_youtubedl_instance
        mock_update_ytdlp.return_value = mock_youtubedl_class
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception):
                download_via_python_api(
                    "https://example.com/video", 
                    temp_dir, 
                    "1080"
                )


class TestDownloadViaCommand:
    """Test download via external command functionality"""
    
    @patch('subprocess.Popen')
    @patch('core._1_ytdlp.find_most_recent_video_file')
    @patch('core._1_ytdlp.load_key')
    @patch('os.path.exists')
    @patch('os.utime')
    def test_download_via_command_success(self, mock_utime, mock_exists, 
                                        mock_load_key, mock_find_recent, 
                                        mock_popen):
        """Test successful download via external command"""
        # Setup mocks
        mock_load_key.return_value = None
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [
            "[download] Destination: /test/video.mp4\n",
            "[download] 100% of 10MB\n",
            ""
        ]
        mock_popen.return_value = mock_process
        mock_exists.return_value = True
        mock_find_recent.return_value = "/test/video.mp4"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_via_command(
                "https://example.com/video", 
                temp_dir, 
                "1080"
            )
            
            assert result == "/test/video.mp4"
            mock_popen.assert_called_once()
            mock_utime.assert_called_once()
            
    @patch('subprocess.Popen')
    @patch('core._1_ytdlp.load_key')
    def test_download_via_command_failure(self, mock_load_key, mock_popen):
        """Test download failure via external command"""
        mock_load_key.return_value = None
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout.readline.side_effect = [
            "ERROR: Video not found\n",
            ""
        ]
        mock_popen.return_value = mock_process
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception):
                download_via_command(
                    "https://example.com/video", 
                    temp_dir, 
                    "1080"
                )


class TestMainDownloadFunction:
    """Test main download function integration"""
    
    @patch('core._1_ytdlp.intelligent_retry_download')
    @patch('core._1_ytdlp.get_storage_paths')
    @patch('core._1_ytdlp.load_key')
    @patch('core._1_ytdlp.ensure_storage_dirs')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('core._1_ytdlp.validate_video_file')
    def test_download_video_ytdlp_success(self, mock_validate, mock_getsize, 
                                        mock_exists, mock_makedirs, 
                                        mock_ensure_dirs, mock_load_key, 
                                        mock_get_paths, mock_retry):
        """Test successful video download"""
        # Setup mocks
        mock_get_paths.return_value = {'input': '/test/input'}
        mock_load_key.return_value = '1080'
        mock_retry.return_value = '/test/video.mp4'
        mock_exists.return_value = True
        mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
        mock_validate.return_value = (True, "File is valid")
        
        result = download_video_ytdlp("https://example.com/video")
        
        assert result == '/test/video.mp4'
        mock_retry.assert_called_once()
        mock_validate.assert_called_once()
        
    @patch('core._1_ytdlp.intelligent_retry_download')
    @patch('core._1_ytdlp.get_storage_paths')
    @patch('core._1_ytdlp.load_key')
    @patch('core._1_ytdlp.ensure_storage_dirs')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('core._1_ytdlp.find_most_recent_video_file')
    def test_download_video_ytdlp_fallback_to_file_search(self, mock_find_recent, 
                                                        mock_exists, mock_makedirs,
                                                        mock_ensure_dirs, mock_load_key,
                                                        mock_get_paths, mock_retry):
        """Test download fallback to file search"""
        # Setup mocks
        mock_get_paths.return_value = {'input': '/test/input'}
        mock_load_key.return_value = '1080'
        mock_retry.return_value = None  # Simulate path detection failure
        mock_exists.side_effect = [False, True]  # First false, then true for fallback
        mock_find_recent.return_value = '/test/fallback_video.mp4'
        
        with patch('os.path.getsize', return_value=10*1024*1024):
            with patch('core._1_ytdlp.validate_video_file', return_value=(True, "Valid")):
                result = download_video_ytdlp("https://example.com/video")
                
                assert result == '/test/fallback_video.mp4'
                mock_find_recent.assert_called_once()
                
    @patch('core._1_ytdlp.intelligent_retry_download')
    @patch('core._1_ytdlp.get_storage_paths')
    @patch('core._1_ytdlp.load_key')
    @patch('core._1_ytdlp.ensure_storage_dirs')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('core._1_ytdlp.find_most_recent_video_file')
    def test_download_video_ytdlp_complete_failure(self, mock_find_recent, 
                                                 mock_exists, mock_makedirs,
                                                 mock_ensure_dirs, mock_load_key,
                                                 mock_get_paths, mock_retry):
        """Test complete download failure"""
        # Setup mocks
        mock_get_paths.return_value = {'input': '/test/input'}
        mock_load_key.return_value = '1080'
        mock_retry.return_value = None
        mock_exists.return_value = False
        mock_find_recent.return_value = None
        
        with pytest.raises(Exception):
            download_video_ytdlp("https://example.com/video")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=core._1_ytdlp', '--cov-report=term-missing'])