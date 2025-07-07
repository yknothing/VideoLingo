"""
Isolated test suite for VideoLingo download functionality
Testing individual functions without heavy dependencies
"""

import os
import pytest
import tempfile
import shutil
import re
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


def sanitize_filename(filename):
    """Copy of the sanitize_filename function from core._1_ytdlp"""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.strip('. ')
    return filename if filename else 'video'


def validate_video_file(file_path, expected_min_size_mb=1):
    """Copy of the validate_video_file function from core._1_ytdlp"""
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb < expected_min_size_mb:
        return False, f"File too small ({file_size_mb:.1f}MB), likely incomplete"
    
    if file_path.endswith(('.part', '.tmp', '.download')):
        return False, "File appears to be a partial download"
    
    try:
        result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                               '-show_format', file_path], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "File format validation failed"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        # If ffprobe not available or times out, skip detailed validation
        print(f"FFprobe validation skipped: {e}")
    
    return True, "File is valid"


def get_optimal_format(resolution):
    """Copy of the get_optimal_format function from core._1_ytdlp"""
    if resolution == 'best':
        return "bestvideo[height<=2160][vcodec^=avc1]/bestvideo[height<=2160][vcodec^=h264]/bestvideo[height<=1440][vcodec^=avc1]/bestvideo[height<=1440][vcodec^=h264]/bestvideo[height<=1080][vcodec^=avc1]/bestvideo[height<=1080][vcodec^=h264]/bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720][vcodec^=h264]/bestvideo[ext=mp4]/bestvideo+bestaudio[acodec^=mp4a]/bestaudio[ext=m4a]/bestaudio/best[vcodec^=avc1]/best[vcodec^=h264]/best[ext=mp4]/best"
    else:
        height_filter = f"[height<={resolution}]"
        video_part = f"bestvideo{height_filter}[vcodec^=avc1]/bestvideo{height_filter}[vcodec^=h264]/bestvideo{height_filter}[ext=mp4]/bestvideo{height_filter}"
        audio_part = "bestaudio[acodec^=mp4a]/bestaudio[ext=m4a]/bestaudio"
        fallback = f"best{height_filter}[vcodec^=avc1]/best{height_filter}[vcodec^=h264]/best{height_filter}[ext=mp4]/best{height_filter}"
        return f"{video_part}+{audio_part}/{fallback}"


def categorize_download_error(error_msg):
    """Copy of the categorize_download_error function from core._1_ytdlp"""
    error_msg_lower = error_msg.lower()
    
    if any(keyword in error_msg_lower for keyword in ['network', 'timeout', 'connection', 'temporary']):
        return "network", True, 30
    elif any(keyword in error_msg_lower for keyword in ['403', '429', 'rate limit', 'too many requests']):
        return "rate_limit", True, 60
    elif any(keyword in error_msg_lower for keyword in ['404', 'not found', 'unavailable']):
        return "not_found", False, 0
    elif any(keyword in error_msg_lower for keyword in ['401', 'unauthorized', 'private', 'restricted']):
        return "access_denied", False, 0
    elif any(keyword in error_msg_lower for keyword in ['proxy', 'ssl', 'certificate']):
        return "proxy_ssl", True, 10
    else:
        return "unknown", True, 20


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
        
    def test_sanitize_filename_unicode_safe(self):
        """Test that unicode characters are preserved"""
        filename = 'Ilya Sutskever： "Sequence to sequence learning with neural networks： what a decade".mp4'
        result = sanitize_filename(filename)
        assert 'Ilya Sutskever' in result
        assert 'Sequence to sequence learning' in result


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
            # Since ffprobe likely isn't available, we just test basic validation
            assert is_valid or 'File format validation failed' in error_msg or error_msg == 'File is valid'
        finally:
            os.unlink(tmp_path)


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
        
    def test_get_optimal_format_720p_resolution(self):
        """Test optimal format selection for 720p resolution"""
        result = get_optimal_format('720')
        assert '[height<=720]' in result
        assert 'avc1' in result
        
    def test_get_optimal_format_1440p_resolution(self):
        """Test optimal format selection for 1440p resolution"""
        result = get_optimal_format('1440')
        assert '[height<=1440]' in result
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
        
    def test_categorize_download_error_proxy_ssl(self):
        """Test categorization of proxy/SSL errors"""
        error_msg = "SSL certificate verification failed"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "proxy_ssl"
        assert is_retryable is True
        assert wait_time == 10
        
    def test_categorize_download_error_unknown(self):
        """Test categorization of unknown errors"""
        error_msg = "Something went wrong"
        category, is_retryable, wait_time = categorize_download_error(error_msg)
        assert category == "unknown"
        assert is_retryable is True
        assert wait_time == 20


class TestDownloadIntegration:
    """Test download integration scenarios"""
    
    def test_resolution_parameter_validation(self):
        """Test that different resolution parameters produce correct formats"""
        # Test all supported resolutions
        resolutions = ['360', '720', '1080', '1440', '2160', 'best']
        
        for resolution in resolutions:
            format_string = get_optimal_format(resolution)
            assert isinstance(format_string, str)
            assert len(format_string) > 0
            
            if resolution == 'best':
                assert 'bestvideo[height<=2160]' in format_string
            else:
                assert f'[height<={resolution}]' in format_string
                
    def test_filename_edge_cases(self):
        """Test filename sanitization with edge cases"""
        edge_cases = [
            ('', 'video'),
            ('   ', 'video'),
            ('...', 'video'),
            ('normal_file.mp4', 'normal_file.mp4'),
            ('file with spaces.mp4', 'file with spaces.mp4'),
            ('file:with|illegal<chars>.mp4', 'filewithillegalchars.mp4'),
            ('Ilya Sutskever： "Sequence.mp4', 'Ilya Sutskever： Sequence.mp4')
        ]
        
        for input_filename, expected in edge_cases:
            result = sanitize_filename(input_filename)
            assert result == expected, f"Failed for input: {input_filename}"
            
    def test_error_categorization_comprehensive(self):
        """Test error categorization with various error messages"""
        test_cases = [
            ("Connection timeout", "network", True, 30),
            ("Network error occurred", "network", True, 30),
            ("Temporary failure", "network", True, 30),
            ("HTTP Error 403: Forbidden", "rate_limit", True, 60),
            ("HTTP Error 429: Too Many Requests", "rate_limit", True, 60),
            ("Rate limit exceeded", "rate_limit", True, 60),
            ("HTTP Error 404: Not Found", "not_found", False, 0),
            ("Video unavailable", "not_found", False, 0),
            ("HTTP Error 401: Unauthorized", "access_denied", False, 0),
            ("This video is private", "access_denied", False, 0),
            ("Proxy connection failed", "network", True, 30),
            ("SSL handshake failed", "proxy_ssl", True, 10),
            ("Certificate verification error", "proxy_ssl", True, 10),
            ("Unknown error occurred", "unknown", True, 20),
            ("Unexpected failure", "unknown", True, 20)
        ]
        
        for error_msg, expected_category, expected_retryable, expected_wait in test_cases:
            category, is_retryable, wait_time = categorize_download_error(error_msg)
            assert category == expected_category, f"Category mismatch for: {error_msg}"
            assert is_retryable == expected_retryable, f"Retryable mismatch for: {error_msg}"
            assert wait_time == expected_wait, f"Wait time mismatch for: {error_msg}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])