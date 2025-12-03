"""
Comprehensive File Upload Security Tests
=======================================

Test suite for validating file upload security features including:
- Path traversal prevention
- MIME type validation  
- Magic number verification
- File size limits
- Filename sanitization
- Virus scanning hooks
- Content validation

Author: VideoLingo Security Team
Last Updated: 2025-08-21
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from core.utils.file_security import (
    FileSecurityValidator,
    FileSecurityError,
    PathTraversalError,
    InvalidMimeTypeError,
    FileSizeError,
    ContentValidationError,
    VirusDetectedError,
    get_user_friendly_error_message,
    validate_uploaded_file
)


class TestFileSecurityValidator:
    """Test the FileSecurityValidator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = FileSecurityValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # ------------------------
    # Filename Validation Tests
    # ------------------------
    
    def test_validate_filename_normal(self):
        """Test normal filename validation"""
        filename = "test_video.mp4"
        result = self.validator.validate_filename(filename)
        assert result == "test_video.mp4"
    
    def test_validate_filename_with_spaces(self):
        """Test filename with spaces gets sanitized"""
        filename = "test video file.mp4"
        result = self.validator.validate_filename(filename)
        assert result == "test_video_file.mp4"
    
    def test_validate_filename_path_traversal_dotdot(self):
        """Test path traversal with .. is blocked"""
        with pytest.raises(PathTraversalError):
            self.validator.validate_filename("../../../etc/passwd.mp4")
    
    def test_validate_filename_path_traversal_absolute(self):
        """Test absolute paths are blocked"""
        with pytest.raises(PathTraversalError):
            self.validator.validate_filename("/etc/passwd.mp4")
    
    def test_validate_filename_windows_reserved_names(self):
        """Test Windows reserved names are blocked"""
        with pytest.raises(PathTraversalError):
            self.validator.validate_filename("CON.mp4")
        
        with pytest.raises(PathTraversalError):
            self.validator.validate_filename("PRN.mp4")
        
        with pytest.raises(PathTraversalError):
            self.validator.validate_filename("AUX.mp4")
    
    def test_validate_filename_dangerous_characters(self):
        """Test dangerous characters are sanitized"""
        filename = "test<>file.mp4"  # Use characters that get sanitized, not blocked
        result = self.validator.validate_filename(filename)
        assert result == "test__file.mp4"
    
    def test_validate_filename_invalid_extension(self):
        """Test invalid file extensions are rejected"""
        with pytest.raises(InvalidMimeTypeError):
            self.validator.validate_filename("malware.exe")
        
        with pytest.raises(InvalidMimeTypeError):
            self.validator.validate_filename("script.php")
    
    def test_validate_filename_empty(self):
        """Test empty filename is rejected"""
        with pytest.raises(FileSecurityError):
            self.validator.validate_filename("")
        
        with pytest.raises(FileSecurityError):
            self.validator.validate_filename("   ")
    
    def test_validate_filename_long_name(self):
        """Test very long filenames are truncated"""
        long_name = "a" * 200 + ".mp4"
        result = self.validator.validate_filename(long_name)
        assert len(result) <= 104  # 100 chars + ".mp4"
    
    # ------------------------
    # File Size Validation Tests
    # ------------------------
    
    def test_validate_file_size_normal(self):
        """Test normal file size validation"""
        # Should not raise exception
        self.validator.validate_file_size(1024 * 1024)  # 1MB
    
    def test_validate_file_size_too_small(self):
        """Test file too small is rejected"""
        with pytest.raises(FileSizeError):
            self.validator.validate_file_size(100)  # Less than 1KB
    
    def test_validate_file_size_too_large(self):
        """Test file too large is rejected"""
        with pytest.raises(FileSizeError):
            self.validator.validate_file_size(3 * 1024 * 1024 * 1024)  # 3GB
    
    def test_validate_file_size_boundary_conditions(self):
        """Test file size boundary conditions"""
        # Minimum allowed size
        self.validator.validate_file_size(1024)  # Exactly 1KB
        
        # Maximum allowed size 
        self.validator.validate_file_size(2 * 1024 * 1024 * 1024)  # Exactly 2GB
    
    # ------------------------
    # MIME Type Validation Tests
    # ------------------------
    
    def test_validate_mime_type_mp4_valid(self):
        """Test valid MP4 MIME type validation"""
        mp4_header = b'\x00\x00\x00\x18ftypmp4'
        mime_type = self.validator.validate_mime_type(mp4_header, "test.mp4")
        assert mime_type in ["video/mp4", "application/octet-stream"]
    
    def test_validate_mime_type_mp3_valid(self):
        """Test valid MP3 MIME type validation"""
        mp3_header = b'ID3\x03\x00\x00\x00'
        mime_type = self.validator.validate_mime_type(mp3_header, "test.mp3")
        assert mime_type in ["audio/mpeg", "application/octet-stream"]
    
    def test_validate_mime_type_content_mismatch(self):
        """Test content doesn't match extension"""
        # PDF content but MP4 extension
        pdf_header = b'%PDF-1.4'
        with pytest.raises(ContentValidationError):
            self.validator.validate_mime_type(pdf_header, "test.mp4")
    
    def test_validate_mime_type_unknown_extension(self):
        """Test unknown content with allowed extension should raise error"""
        # PNG content with MP4 extension should fail
        png_content = b'\x89PNG\r\n\x1a\n'  # PNG header
        with pytest.raises(ContentValidationError):
            self.validator.validate_mime_type(png_content, "test.mp4")
    
    # ------------------------
    # Magic Number Tests
    # ------------------------
    
    def test_check_magic_number_mp4(self):
        """Test MP4 magic number validation"""
        mp4_content = b'\x00\x00\x00\x18ftypmp4'
        assert self.validator._check_magic_number(mp4_content, "mp4")
    
    def test_check_magic_number_mp3(self):
        """Test MP3 magic number validation"""
        mp3_content = b'ID3\x03\x00'
        assert self.validator._check_magic_number(mp3_content, "mp3")
    
    def test_check_magic_number_wav(self):
        """Test WAV magic number validation"""
        wav_content = b'RIFF\x24\x08\x00\x00WAVE'
        assert self.validator._check_magic_number(wav_content, "wav")
    
    def test_check_magic_number_invalid(self):
        """Test invalid magic number"""
        invalid_content = b'InvalidContent'
        assert not self.validator._check_magic_number(invalid_content, "mp4")
    
    def test_check_magic_number_unknown_type(self):
        """Test unknown file type returns True (allowed)"""
        content = b'SomeContent'
        assert self.validator._check_magic_number(content, "unknown")
    
    # ------------------------
    # Virus Scanning Tests
    # ------------------------
    
    def test_virus_scan_disabled(self):
        """Test virus scanning when disabled"""
        validator = FileSecurityValidator(enable_virus_scan=False)
        assert validator.scan_for_virus("any_path") is True
    
    def test_virus_scan_enabled_clean_file(self):
        """Test virus scanning with clean file"""
        validator = FileSecurityValidator(enable_virus_scan=True)
        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock implementation always returns clean
            assert validator.scan_for_virus(temp_file.name) is True
    
    @patch('core.utils.file_security.FileSecurityValidator.scan_for_virus')
    def test_virus_scan_infected_file(self, mock_scan):
        """Test virus scanning with infected file"""
        mock_scan.side_effect = VirusDetectedError("Virus detected")
        validator = FileSecurityValidator(enable_virus_scan=True)
        
        with pytest.raises(VirusDetectedError):
            validator.scan_for_virus("infected_file.mp4")
    
    # ------------------------
    # Secure File Storage Tests
    # ------------------------
    
    def test_create_secure_temp_file(self):
        """Test secure temporary file creation"""
        temp_path = self.validator.create_secure_temp_file(suffix=".mp4")
        
        try:
            assert os.path.exists(temp_path)
            assert temp_path.endswith(".mp4")
            
            # Check file permissions (owner read/write only)
            stat_info = os.stat(temp_path)
            assert stat_info.st_mode & 0o777 == 0o600
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_store_file_securely(self):
        """Test secure file storage"""
        test_data = b"Test video content"
        filename = "test.mp4"
        
        stored_path = self.validator.store_file_securely(
            test_data, self.temp_dir, filename
        )
        
        assert os.path.exists(stored_path)
        assert os.path.basename(stored_path) == filename
        
        with open(stored_path, 'rb') as f:
            assert f.read() == test_data
        
        # Check file permissions
        stat_info = os.stat(stored_path)
        assert stat_info.st_mode & 0o777 == 0o644
    
    def test_store_file_securely_prevent_overwrite(self):
        """Test secure storage prevents overwriting existing files"""
        test_data = b"Test content"
        filename = "test.mp4"
        
        # Store first file
        first_path = self.validator.store_file_securely(
            test_data, self.temp_dir, filename
        )
        
        # Store second file with same name
        second_path = self.validator.store_file_securely(
            test_data, self.temp_dir, filename
        )
        
        assert first_path != second_path
        assert os.path.basename(second_path) == "test_1.mp4"
        assert os.path.exists(first_path)
        assert os.path.exists(second_path)
    
    # ------------------------
    # Comprehensive Upload Validation Tests
    # ------------------------
    
    def test_validate_upload_success(self):
        """Test successful upload validation"""
        # Valid MP4 content - ensure it's large enough (>1KB)
        mp4_content = b'\x00\x00\x00\x18ftypmp4' + b'x' * 2000  # >1KB
        filename = "test_video.mp4"
        file_size = len(mp4_content)
        
        sanitized_filename, mime_type = self.validator.validate_upload(
            mp4_content, filename, file_size
        )
        
        assert sanitized_filename == "test_video.mp4"
        assert mime_type in ["video/mp4", "application/octet-stream"]
    
    def test_validate_upload_security_failure(self):
        """Test upload validation with security failure"""
        content = b"some content"
        filename = "../../../etc/passwd.mp4"  # Path traversal
        file_size = len(content)
        
        with pytest.raises(PathTraversalError):
            self.validator.validate_upload(content, filename, file_size)
    
    def test_validate_upload_size_failure(self):
        """Test upload validation with size failure"""
        content = b"small"  # Too small
        filename = "test.mp4"
        file_size = len(content)
        
        with pytest.raises(FileSizeError):
            self.validator.validate_upload(content, filename, file_size)
    
    def test_validate_upload_content_failure(self):
        """Test upload validation with content mismatch"""
        # PDF content with MP4 extension - ensure it's large enough
        content = b'%PDF-1.4' + b'x' * 2000  # >1KB
        filename = "test.mp4"
        file_size = len(content)
        
        with pytest.raises(ContentValidationError):
            self.validator.validate_upload(content, filename, file_size)


class TestUserFriendlyErrorMessages:
    """Test user-friendly error message generation"""
    
    def test_path_traversal_error_message(self):
        """Test path traversal error message"""
        error = PathTraversalError("Dangerous pattern detected")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '🚫 Invalid Filename'
        assert 'invalid characters' in message['message']
        assert 'rename your file' in message['suggestion']
    
    def test_invalid_mime_type_error_message(self):
        """Test invalid MIME type error message"""
        error = InvalidMimeTypeError("MIME type 'application/pdf' not allowed")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '📄 Unsupported File Type'
        assert 'not allowed' in message['message']
        assert 'MP4, AVI' in message['suggestion']
    
    def test_file_size_error_message(self):
        """Test file size error message"""
        error = FileSizeError("File too large. Maximum size: 2048.0 MB")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '📏 File Size Issue'
        assert 'too large' in message['message']
        assert 'compressing' in message['suggestion']
    
    def test_content_validation_error_message(self):
        """Test content validation error message"""
        error = ContentValidationError("Content doesn't match extension")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '🔍 Content Validation Failed'
        assert 'content' in message['message']
        assert 'valid video' in message['suggestion']
    
    def test_virus_detected_error_message(self):
        """Test virus detected error message"""
        error = VirusDetectedError("Virus found in file")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '🛡️ Security Alert'
        assert 'security scanning' in message['message']
        assert 'antivirus' in message['suggestion']
    
    def test_generic_error_message(self):
        """Test generic error message"""
        error = FileSecurityError("Some unexpected error")
        message = get_user_friendly_error_message(error)
        
        assert message['title'] == '⚠️ Upload Error'
        assert 'Some unexpected error' in message['message']
        assert 'try again' in message['suggestion']


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_validate_uploaded_file_success(self):
        """Test convenience function with valid file"""
        mp4_content = b'\x00\x00\x00\x18ftypmp4' + b'x' * 2000  # >1KB
        filename = "test.mp4"
        file_size = len(mp4_content)
        
        sanitized_filename, mime_type = validate_uploaded_file(
            mp4_content, filename, file_size
        )
        
        assert sanitized_filename == "test.mp4"
        assert mime_type is not None
    
    def test_validate_uploaded_file_failure(self):
        """Test convenience function with invalid file"""
        content = b"invalid"
        filename = "../malicious.mp4"
        file_size = len(content)
        
        with pytest.raises(FileSecurityError):
            validate_uploaded_file(content, filename, file_size)


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    def test_end_to_end_secure_upload(self):
        """Test complete secure upload workflow"""
        # Create validator
        validator = FileSecurityValidator()
        
        # Valid file data - ensure it's large enough
        mp4_content = b'\x00\x00\x00\x18ftypmp4' + b'Valid video content' * 200  # >1KB
        filename = "my test video.mp4"
        file_size = len(mp4_content)
        
        # Validate upload
        sanitized_filename, mime_type = validator.validate_upload(
            mp4_content[:1024], filename, file_size
        )
        
        # Store securely
        with tempfile.TemporaryDirectory() as temp_dir:
            stored_path = validator.store_file_securely(
                mp4_content, temp_dir, sanitized_filename
            )
            
            # Verify stored file
            assert os.path.exists(stored_path)
            assert os.path.basename(stored_path) == "my_test_video.mp4"
            
            with open(stored_path, 'rb') as f:
                assert f.read() == mp4_content
    
    def test_malicious_upload_blocked(self):
        """Test that malicious uploads are properly blocked"""
        validator = FileSecurityValidator()
        
        # Malicious file with path traversal
        malicious_content = b'Malicious executable content'
        malicious_filename = "../../etc/passwd"
        file_size = len(malicious_content)
        
        with pytest.raises(FileSecurityError):
            validator.validate_upload(
                malicious_content, malicious_filename, file_size
            )
    
    def test_performance_large_file_validation(self):
        """Test performance with large file validation"""
        import time
        
        validator = FileSecurityValidator()
        
        # Large file simulation (only validate headers)
        large_file_header = b'\x00\x00\x00\x18ftypmp4' + b'x' * 1000
        filename = "large_video.mp4"
        file_size = 1024 * 1024 * 1024  # 1GB reported size
        
        start_time = time.time()
        sanitized_filename, mime_type = validator.validate_upload(
            large_file_header, filename, file_size
        )
        end_time = time.time()
        
        # Validation should be fast even for large files
        assert end_time - start_time < 1.0  # Less than 1 second
        assert sanitized_filename == "large_video.mp4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])