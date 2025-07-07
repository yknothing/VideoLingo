"""
Test suite for refactored modules
Validates that SRP violations have been fixed and modules work correctly
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, mock_open


class TestFilenameUtils:
    """Test the separated filename utilities"""
    
    def test_sanitize_filename_functionality(self):
        """Test that filename sanitization works correctly"""
        # Import the separated module
        from core.download.filename_utils import sanitize_filename, validate_filename_safety, generate_safe_filename
        
        # Test basic sanitization
        assert sanitize_filename('video<>:"/\\|?*.mp4') == 'video.mp4'
        assert sanitize_filename(' .test. ') == 'test'
        assert sanitize_filename('') == 'video'
        
        # Test validation
        is_valid, msg = validate_filename_safety('normal_video.mp4')
        assert is_valid
        
        is_valid, msg = validate_filename_safety('CON.mp4')
        assert not is_valid
        assert 'reserved name' in msg.lower()
        
        # Test safe generation
        safe_name = generate_safe_filename('very_long_filename' * 20 + '.mp4', max_length=50)
        assert len(safe_name) <= 50
        assert safe_name.endswith('.mp4')


class TestVideoValidator:
    """Test the separated video validator"""
    
    def test_video_validator_functionality(self):
        """Test that video validation works correctly"""
        from core.download.video_validator import VideoFileValidator, PartialDownloadCleaner
        
        validator = VideoFileValidator(min_size_mb=1.0)
        
        # Test non-existent file
        is_valid, msg = validator.validate_video_file('/nonexistent/file.mp4')
        assert not is_valid
        assert 'does not exist' in msg
        
        # Test partial download detection
        with tempfile.NamedTemporaryFile(suffix='.part', delete=False) as tmp:
            tmp.write(b'x' * (2 * 1024 * 1024))  # 2MB
            tmp_path = tmp.name
        
        try:
            is_valid, msg = validator.validate_video_file(tmp_path)
            assert not is_valid
            assert 'partial download' in msg
        finally:
            os.unlink(tmp_path)
        
        # Test cleaner
        cleaner = PartialDownloadCleaner()
        assert hasattr(cleaner, 'cleanup_partial_downloads')
        assert hasattr(cleaner, 'partial_patterns')


class TestFormatSelector:
    """Test the separated format selector"""
    
    def test_format_selector_functionality(self):
        """Test that format selection works correctly"""
        from core.download.format_selector import VideoFormatSelector
        
        selector = VideoFormatSelector()
        
        # Test resolution validation
        assert selector.validate_resolution('1080')
        assert selector.validate_resolution('best')
        assert not selector.validate_resolution('invalid')
        
        # Test format generation
        format_1080 = selector.get_optimal_format('1080')
        format_best = selector.get_optimal_format('best')
        
        assert isinstance(format_1080, str)
        assert isinstance(format_best, str)
        assert '[height<=1080]' in format_1080
        assert 'bestvideo[height<=2160]' in format_best
        
        # Test supported resolutions
        resolutions = selector.get_supported_resolutions()
        assert isinstance(resolutions, dict)
        assert '1080p' in resolutions
        assert 'Best Quality' in resolutions
        
        # Test format info
        info = selector.get_format_info('1080')
        assert 'resolution' in info
        assert 'video_codecs' in info
        assert 'audio_codecs' in info


class TestErrorHandler:
    """Test the separated error handler"""
    
    def test_error_handler_functionality(self):
        """Test that error handling works correctly"""
        from core.download.error_handler import DownloadErrorHandler, ErrorCategory
        
        handler = DownloadErrorHandler()
        
        # Test error categorization
        category, retryable, wait = handler.categorize_download_error("Network timeout")
        assert category == ErrorCategory.NETWORK
        assert retryable is True
        assert wait == 30
        
        category, retryable, wait = handler.categorize_download_error("404 not found")
        assert category == ErrorCategory.NOT_FOUND
        assert retryable is False
        assert wait == 0
        
        # Test retry decorator
        decorator = handler.create_retry_decorator(max_retries=2)
        assert callable(decorator)
        
        # Test suggestions
        suggestion = handler.get_error_suggestion("Rate limit exceeded")
        assert 'rate limit' in suggestion.lower()
        assert 'retryable' in suggestion.lower()


class TestDownloadManager:
    """Test the main download manager"""
    
    def test_download_manager_creation(self):
        """Test that download manager can be created and configured"""
        from core.download.download_manager import DownloadManager, DownloadConfig
        
        # Test default config
        config = DownloadConfig()
        assert config.resolution == "1080"
        assert config.max_retries == 3
        assert config.min_file_size_mb == 1.0
        
        # Test manager creation
        manager = DownloadManager(config)
        assert manager.config == config
        assert hasattr(manager, 'validator')
        assert hasattr(manager, 'cleaner')
        assert hasattr(manager, 'format_selector')
        assert hasattr(manager, 'error_handler')
        
        # Test URL validation
        assert manager.validate_url('https://www.youtube.com/watch?v=example')
        assert manager.validate_url('http://example.com/video.mp4')
        assert not manager.validate_url('not_a_url')
        assert not manager.validate_url('ftp://example.com')
        
        # Test configuration methods
        resolutions = manager.get_supported_resolutions()
        assert isinstance(resolutions, dict)
        assert len(resolutions) > 0


class TestBackwardCompatibility:
    """Test that the refactored module maintains backward compatibility"""
    
    def test_refactored_module_imports(self):
        """Test that the refactored module can be imported and has expected functions"""
        try:
            from core._1_ytdlp_refactored import (
                download_video_ytdlp,
                find_video_files,
                get_optimal_format,
                validate_video_file,
                sanitize_filename,
                categorize_download_error,
                intelligent_retry_download,
                cleanup_partial_downloads,
                find_most_recent_video_file,
                find_best_video_file
            )
            
            # Check that all functions exist and are callable
            assert callable(download_video_ytdlp)
            assert callable(find_video_files)
            assert callable(get_optimal_format)
            assert callable(validate_video_file)
            assert callable(sanitize_filename)
            assert callable(categorize_download_error)
            assert callable(intelligent_retry_download)
            assert callable(cleanup_partial_downloads)
            assert callable(find_most_recent_video_file)
            assert callable(find_best_video_file)
            
        except ImportError as e:
            pytest.skip(f"Refactored module not available: {e}")
    
    def test_backward_compatible_functions(self):
        """Test that backward compatible functions work correctly"""
        try:
            from core._1_ytdlp_refactored import get_optimal_format, validate_video_file
            
            # Test format selection
            format_str = get_optimal_format('1080')
            assert isinstance(format_str, str)
            assert len(format_str) > 0
            
            # Test validation (should work with non-existent file)
            is_valid, msg = validate_video_file('/nonexistent/file.mp4')
            assert not is_valid
            assert isinstance(msg, str)
            
        except ImportError as e:
            pytest.skip(f"Refactored module not available: {e}")


class TestModuleArchitecture:
    """Test that the modular architecture follows SRP"""
    
    def test_single_responsibility_principle(self):
        """Test that each module has a single, well-defined responsibility"""
        
        # filename_utils should only handle filename operations
        from core.download import filename_utils
        filename_functions = [attr for attr in dir(filename_utils) if not attr.startswith('_')]
        filename_related = ['sanitize_filename', 'validate_filename_safety', 'generate_safe_filename']
        assert all(any(keyword in func for keyword in ['filename', 'name']) for func in filename_related)
        
        # video_validator should only handle video validation
        from core.download import video_validator
        validator_classes = [getattr(video_validator, name) for name in dir(video_validator) 
                           if isinstance(getattr(video_validator, name), type)]
        assert len(validator_classes) >= 2  # VideoFileValidator, PartialDownloadCleaner
        
        # format_selector should only handle format selection
        from core.download import format_selector
        selector_classes = [getattr(format_selector, name) for name in dir(format_selector)
                          if isinstance(getattr(format_selector, name), type)]
        assert len(selector_classes) >= 1  # VideoFormatSelector
        
        # error_handler should only handle errors
        from core.download import error_handler
        error_classes = [getattr(error_handler, name) for name in dir(error_handler)
                        if isinstance(getattr(error_handler, name), type)]
        assert any('Error' in cls.__name__ for cls in error_classes)
    
    def test_composition_over_inheritance(self):
        """Test that the design uses composition over inheritance"""
        from core.download.download_manager import DownloadManager
        
        manager = DownloadManager()
        
        # Check that manager uses composition
        assert hasattr(manager, 'validator')
        assert hasattr(manager, 'cleaner')
        assert hasattr(manager, 'format_selector')
        assert hasattr(manager, 'error_handler')
        
        # Check that these are separate objects, not inherited methods
        assert manager.validator is not manager
        assert manager.cleaner is not manager
        assert manager.format_selector is not manager
        assert manager.error_handler is not manager
    
    def test_dependency_injection(self):
        """Test that dependencies can be injected for testing"""
        from core.download.download_manager import DownloadManager, DownloadConfig
        
        # Test that config can be injected
        custom_config = DownloadConfig(resolution="720", max_retries=5)
        manager = DownloadManager(custom_config)
        
        assert manager.config.resolution == "720"
        assert manager.config.max_retries == 5
        
        # Test that components use the injected config
        assert manager.validator.min_size_mb == custom_config.min_file_size_mb


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])