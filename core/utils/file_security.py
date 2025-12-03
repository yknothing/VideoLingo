"""
VideoLingo File Upload Security Module
=====================================

Comprehensive security module for safe file upload handling with:
- Path traversal prevention
- MIME type validation with magic number verification
- File size limits and content validation
- Filename sanitization
- Virus scanning hooks
- Security logging

Author: VideoLingo Security Team
Last Updated: 2025-08-21
"""

import os
import re
import hashlib
import logging
import mimetypes
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import shutil

# Configure security logger
security_logger = logging.getLogger('videolingo.security')
if not security_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    security_logger.addHandler(handler)
    security_logger.setLevel(logging.INFO)


class FileSecurityError(Exception):
    """Base exception for file security violations"""
    pass


class PathTraversalError(FileSecurityError):
    """Raised when path traversal attempt is detected"""
    pass


class InvalidMimeTypeError(FileSecurityError):
    """Raised when file MIME type is not allowed"""
    pass


class FileSizeError(FileSecurityError):
    """Raised when file size exceeds limits"""
    pass


class ContentValidationError(FileSecurityError):
    """Raised when file content validation fails"""
    pass


class VirusDetectedError(FileSecurityError):
    """Raised when virus is detected in file"""
    pass


class FileSecurityValidator:
    """
    Comprehensive file security validator for VideoLingo uploads
    
    Features:
    - Magic number validation
    - MIME type verification
    - File size limits
    - Filename sanitization
    - Path traversal prevention
    - Content signature verification
    - Virus scanning hooks
    """
    
    # Magic number signatures for supported file types
    MAGIC_NUMBERS = {
        # Video formats
        'mp4': [
            b'\x00\x00\x00\x18ftypmp4',    # MP4 container
            b'\x00\x00\x00\x20ftypmp4',    # MP4 container variant
            b'\x00\x00\x00\x1cftypmp4',    # MP4 container variant
            b'ftypmp4',                     # MP4 simplified check
            b'ftypisom',                    # ISO Base Media
        ],
        'avi': [b'RIFF', b'AVI '],
        'mov': [b'ftypqt  ', b'moov', b'free', b'mdat'],
        'mkv': [b'\x1a\x45\xdf\xa3'],      # EBML header
        'webm': [b'\x1a\x45\xdf\xa3'],     # WebM (same as MKV)
        'flv': [b'FLV'],
        'wmv': [b'\x30\x26\xb2\x75\x8e\x66\xcf\x11'],
        
        # Audio formats  
        'mp3': [
            b'ID3',                         # MP3 with ID3 tag
            b'\xff\xfb',                    # MP3 frame sync
            b'\xff\xf3',                    # MP3 frame sync
            b'\xff\xf2',                    # MP3 frame sync
        ],
        'wav': [b'RIFF', b'WAVE'],
        'flac': [b'fLaC'],
        'aac': [b'\xff\xf1', b'\xff\xf9'],
        'm4a': [b'ftypM4A ', b'ftypm4a'],
        'ogg': [b'OggS'],
        'wma': [b'\x30\x26\xb2\x75\x8e\x66\xcf\x11'],
    }
    
    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        # Video MIME types
        'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo',
        'video/x-ms-wmv', 'video/webm', 'video/x-flv', 'video/x-matroska',
        
        # Audio MIME types  
        'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac',
        'audio/aac', 'audio/mp4', 'audio/ogg', 'audio/x-ms-wma',
        'audio/vnd.wav', 'audio/wave',
    }
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB default
    MIN_FILE_SIZE = 1024  # 1KB minimum
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv',
        '.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg', '.wma'
    }
    
    # Patterns that should cause rejection (security violations)
    REJECTION_PATTERNS = [
        r'\.\.[\\/]',           # Path traversal
        r'^[\\/]',              # Absolute paths
        r'\x00',                # Null bytes
        r'(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)',  # Windows reserved names
    ]
    
    # Characters that should be sanitized (replaced with underscores)
    SANITIZE_PATTERN = r'[<>:"|?*]'  # Windows forbidden chars
    
    def __init__(self, max_file_size: Optional[int] = None, 
                 custom_allowed_extensions: Optional[List[str]] = None,
                 enable_virus_scan: bool = False):
        """
        Initialize the file security validator
        
        Args:
            max_file_size: Maximum allowed file size in bytes
            custom_allowed_extensions: Custom list of allowed extensions
            enable_virus_scan: Whether to enable virus scanning
        """
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE
        self.allowed_extensions = set(custom_allowed_extensions) if custom_allowed_extensions else self.ALLOWED_EXTENSIONS
        self.enable_virus_scan = enable_virus_scan
        
        security_logger.info(f"FileSecurityValidator initialized: max_size={self.max_file_size}, virus_scan={enable_virus_scan}")
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize filename to prevent security issues
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            PathTraversalError: If path traversal attempt detected
            FileSecurityError: If filename is invalid
        """
        if not filename or not filename.strip():
            raise FileSecurityError("Filename cannot be empty")
            
        # Check for patterns that should cause rejection
        for pattern in self.REJECTION_PATTERNS:
            if re.search(pattern, filename):
                security_logger.warning(f"Dangerous filename pattern detected: {filename}")
                raise PathTraversalError(f"Dangerous filename pattern detected: {pattern}")
        
        # Sanitize filename
        # Remove path components
        filename = os.path.basename(filename)
        
        # Replace spaces with underscores
        sanitized = filename.replace(' ', '_')
        
        # Sanitize dangerous characters
        sanitized = re.sub(self.SANITIZE_PATTERN, '_', sanitized)
        
        # Replace any remaining non-word characters (except dots, hyphens, underscores)
        sanitized = re.sub(r'[^\w\-_\.]', '_', sanitized)
        
        # Ensure extension is lowercase
        name, ext = os.path.splitext(sanitized)
        ext = ext.lower()
        
        # Validate extension
        if ext not in self.allowed_extensions:
            raise InvalidMimeTypeError(f"File extension '{ext}' not allowed. Allowed: {', '.join(self.allowed_extensions)}")
        
        # Prevent empty names
        if not name or name == '_':
            name = f"upload_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
        
        # Limit filename length
        max_name_length = 100
        if len(name) > max_name_length:
            name = name[:max_name_length]
        
        sanitized_filename = f"{name}{ext}"
        
        if sanitized_filename != filename:
            security_logger.info(f"Filename sanitized: '{filename}' -> '{sanitized_filename}'")
        
        return sanitized_filename
    
    def validate_file_size(self, file_size: int, filename: str = "unknown") -> None:
        """
        Validate file size is within acceptable limits
        
        Args:
            file_size: Size of file in bytes
            filename: Filename for logging
            
        Raises:
            FileSizeError: If file size is invalid
        """
        if file_size < self.MIN_FILE_SIZE:
            security_logger.warning(f"File too small: {filename} ({file_size} bytes)")
            raise FileSizeError(f"File too small. Minimum size: {self.MIN_FILE_SIZE} bytes")
        
        if file_size > self.max_file_size:
            security_logger.warning(f"File too large: {filename} ({file_size} bytes)")
            raise FileSizeError(f"File too large. Maximum size: {self.max_file_size / 1024 / 1024:.1f} MB")
    
    def validate_mime_type(self, file_content: bytes, filename: str) -> str:
        """
        Validate MIME type using both filename and content analysis
        
        Args:
            file_content: First few bytes of file content
            filename: Original filename
            
        Returns:
            Detected MIME type
            
        Raises:
            InvalidMimeTypeError: If MIME type is not allowed
        """
        # Get MIME type from filename
        mime_type, _ = mimetypes.guess_type(filename)
        
        # Validate based on file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in self.ALLOWED_EXTENSIONS:
            # Verify magic number matches extension
            if not self._check_magic_number(file_content, ext[1:]):  # Remove dot from extension
                security_logger.warning(f"Magic number mismatch for {filename} (claimed: {ext})")
                raise ContentValidationError(f"File content doesn't match extension {ext}")
        
        # Validate MIME type
        if mime_type and mime_type not in self.ALLOWED_MIME_TYPES:
            security_logger.warning(f"Invalid MIME type: {mime_type} for {filename}")
            raise InvalidMimeTypeError(f"MIME type '{mime_type}' not allowed")
        
        return mime_type or "application/octet-stream"
    
    def _check_magic_number(self, file_content: bytes, file_type: str) -> bool:
        """
        Check if file content matches expected magic numbers
        
        Args:
            file_content: First few bytes of file
            file_type: Expected file type (without dot)
            
        Returns:
            True if magic number matches, False otherwise
        """
        if file_type not in self.MAGIC_NUMBERS:
            return True  # Allow unknown types if extension is in allowed list
        
        expected_signatures = self.MAGIC_NUMBERS[file_type]
        
        for signature in expected_signatures:
            if file_content.startswith(signature):
                return True
            
            # Check if signature appears within first 32 bytes (for some formats)
            if signature in file_content[:32]:
                return True
        
        return False
    
    def scan_for_virus(self, file_path: str) -> bool:
        """
        Placeholder for virus scanning integration
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            True if file is clean, False if virus detected
            
        Raises:
            VirusDetectedError: If virus is detected
        """
        if not self.enable_virus_scan:
            return True
        
        # TODO: Integrate with actual virus scanner
        # Example integrations:
        # - ClamAV
        # - Windows Defender API
        # - Commercial AV solutions
        
        security_logger.info(f"Virus scan placeholder for: {file_path}")
        
        # Placeholder implementation - always returns clean
        # In production, implement actual virus scanning here
        return True
    
    def create_secure_temp_file(self, suffix: str = None) -> str:
        """
        Create a secure temporary file with restricted permissions
        
        Args:
            suffix: File suffix/extension
            
        Returns:
            Path to secure temporary file
        """
        # Create temp file with restrictive permissions (owner read/write only)
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='videolingo_secure_')
        
        # Set restrictive permissions (0o600 = owner read/write only)
        os.chmod(temp_path, 0o600)
        os.close(fd)
        
        security_logger.info(f"Created secure temp file: {temp_path}")
        return temp_path
    
    def validate_upload(self, uploaded_file_data: bytes, filename: str, 
                       file_size: int) -> Tuple[str, str]:
        """
        Comprehensive validation of uploaded file
        
        Args:
            uploaded_file_data: First chunk of uploaded file data
            filename: Original filename
            file_size: Total file size
            
        Returns:
            Tuple of (sanitized_filename, mime_type)
            
        Raises:
            FileSecurityError: If any validation fails
        """
        security_logger.info(f"Validating upload: {filename} ({file_size} bytes)")
        
        # 1. Validate filename and sanitize
        sanitized_filename = self.validate_filename(filename)
        
        # 2. Validate file size
        self.validate_file_size(file_size, filename)
        
        # 3. Validate MIME type and content
        mime_type = self.validate_mime_type(uploaded_file_data, sanitized_filename)
        
        security_logger.info(f"Upload validation passed: {sanitized_filename}")
        return sanitized_filename, mime_type
    
    def store_file_securely(self, uploaded_file_data: bytes, target_directory: str, 
                           sanitized_filename: str) -> str:
        """
        Store uploaded file securely with proper permissions
        
        Args:
            uploaded_file_data: Complete file data
            target_directory: Target directory for storage
            sanitized_filename: Sanitized filename
            
        Returns:
            Path to stored file
            
        Raises:
            FileSecurityError: If storage fails
        """
        # Ensure target directory exists and is secure
        os.makedirs(target_directory, mode=0o755, exist_ok=True)
        
        # Create final file path
        final_path = os.path.join(target_directory, sanitized_filename)
        
        # Prevent overwriting existing files
        counter = 1
        base_name, ext = os.path.splitext(sanitized_filename)
        while os.path.exists(final_path):
            new_filename = f"{base_name}_{counter}{ext}"
            final_path = os.path.join(target_directory, new_filename)
            counter += 1
        
        # Write file with secure permissions
        try:
            with open(final_path, 'wb') as f:
                f.write(uploaded_file_data)
            
            # Set restrictive permissions (owner read/write, group read)
            os.chmod(final_path, 0o644)
            
            security_logger.info(f"File stored securely: {final_path}")
            return final_path
            
        except Exception as e:
            security_logger.error(f"Failed to store file securely: {e}")
            raise FileSecurityError(f"Failed to store file: {e}")


def get_user_friendly_error_message(error: FileSecurityError) -> Dict[str, str]:
    """
    Convert security errors to user-friendly messages
    
    Args:
        error: FileSecurityError instance
        
    Returns:
        Dict with 'title', 'message', and 'suggestion' keys
    """
    if isinstance(error, PathTraversalError):
        return {
            'title': '🚫 Invalid Filename',
            'message': 'The filename contains invalid characters or patterns.',
            'suggestion': 'Please rename your file to use only letters, numbers, hyphens, and underscores.'
        }
    
    elif isinstance(error, InvalidMimeTypeError):
        return {
            'title': '📄 Unsupported File Type',
            'message': str(error),
            'suggestion': 'Please upload a video file (MP4, AVI, MOV, MKV, WebM) or audio file (MP3, WAV, FLAC, AAC).'
        }
    
    elif isinstance(error, FileSizeError):
        return {
            'title': '📏 File Size Issue',
            'message': str(error),
            'suggestion': 'Please choose a file within the size limits. For large files, consider compressing the video first.'
        }
    
    elif isinstance(error, ContentValidationError):
        return {
            'title': '🔍 Content Validation Failed',
            'message': 'The file content doesn\'t match the expected format.',
            'suggestion': 'Please ensure your file is a valid video or audio file and hasn\'t been corrupted.'
        }
    
    elif isinstance(error, VirusDetectedError):
        return {
            'title': '🛡️ Security Alert',
            'message': 'The uploaded file failed security scanning.',
            'suggestion': 'Please scan your file with antivirus software and try uploading again.'
        }
    
    else:
        return {
            'title': '⚠️ Upload Error',
            'message': str(error),
            'suggestion': 'Please try again or contact support if the problem persists.'
        }


# Global validator instance
_default_validator = None


def get_default_validator() -> FileSecurityValidator:
    """Get the default file security validator instance"""
    global _default_validator
    if _default_validator is None:
        _default_validator = FileSecurityValidator()
    return _default_validator


def validate_uploaded_file(uploaded_file_data: bytes, filename: str, 
                          file_size: int) -> Tuple[str, str]:
    """
    Convenience function for validating uploaded files
    
    Args:
        uploaded_file_data: File data bytes
        filename: Original filename  
        file_size: File size in bytes
        
    Returns:
        Tuple of (sanitized_filename, mime_type)
    """
    validator = get_default_validator()
    return validator.validate_upload(uploaded_file_data, filename, file_size)