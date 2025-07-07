"""
Video file validation utilities
Single responsibility: Validate video files and handle file integrity checks
"""

import os
import subprocess
import glob
from typing import Tuple, List, Optional


class VideoFileValidator:
    """Handle video file validation and integrity checks"""
    
    def __init__(self, min_size_mb: float = 1.0):
        self.min_size_mb = min_size_mb
        
    def validate_video_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate video file integrity and size
        
        Args:
            file_path (str): Path to video file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb < self.min_size_mb:
            return False, f"File too small ({file_size_mb:.1f}MB), likely incomplete"
        
        # Check if file is a partial download (common extensions)
        if file_path.endswith(('.part', '.tmp', '.download')):
            return False, "File appears to be a partial download"
        
        # Basic file format validation
        if not self._validate_format_with_ffprobe(file_path):
            return False, "File format validation failed"
        
        return True, "File is valid"
    
    def _validate_format_with_ffprobe(self, file_path: str) -> bool:
        """Validate file format using ffprobe if available"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                 '-show_format', file_path], 
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If ffprobe not available, skip detailed validation
            return True
    
    def find_best_video_file(self, save_path: str, allowed_formats: List[str]) -> Optional[str]:
        """
        Find the best video file when multiple files exist
        Priority: largest valid file with proper format
        
        Args:
            save_path (str): Directory to search
            allowed_formats (list): Allowed file extensions
            
        Returns:
            str or None: Path to best video file
        """
        candidates = []
        
        for file_path in glob.glob(os.path.join(save_path, "*")):
            if not os.path.isfile(file_path):
                continue
                
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext not in allowed_formats:
                continue
                
            # Validate file
            is_valid, error_msg = self.validate_video_file(file_path)
            if not is_valid:
                continue
                
            file_size = os.path.getsize(file_path)
            candidates.append((file_path, file_size))
        
        if not candidates:
            return None
        
        # Return largest valid file
        best_file = max(candidates, key=lambda x: x[1])[0]
        return best_file
    
    def find_most_recent_video_file(self, save_path: str, allowed_formats: List[str]) -> Optional[str]:
        """
        Find the most recently created video file in the directory
        
        Args:
            save_path (str): Directory to search
            allowed_formats (list): Allowed file extensions
            
        Returns:
            str or None: Path to most recent video file
        """
        video_files = []
        
        for file_path in glob.glob(os.path.join(save_path, "*")):
            if not os.path.isfile(file_path):
                continue
                
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext not in allowed_formats:
                continue
                
            # Skip obvious non-video files
            if file_path.endswith(('.jpg', '.jpeg', '.png', '.txt', '.json', '.part', '.tmp')):
                continue
                
            # Validate file
            is_valid, error_msg = self.validate_video_file(file_path)
            if not is_valid:
                continue
                
            # Get file modification time
            mtime = os.path.getmtime(file_path)
            video_files.append((file_path, mtime))
        
        if not video_files:
            return None
        
        # Return the most recently modified file
        most_recent = max(video_files, key=lambda x: x[1])[0]
        return most_recent


class PartialDownloadCleaner:
    """Handle cleanup of partial downloads and temporary files"""
    
    def __init__(self):
        self.partial_patterns = ['*.part', '*.tmp', '*.download', '*.f*', '*.ytdl']
        
    def cleanup_partial_downloads(self, save_path: str) -> List[str]:
        """
        Clean up partial downloads and temporary files
        
        Args:
            save_path (str): Directory to clean
            
        Returns:
            list: List of cleaned files
        """
        cleaned_files = []
        
        for pattern in self.partial_patterns:
            for file_path in glob.glob(os.path.join(save_path, pattern)):
                try:
                    os.remove(file_path)
                    cleaned_files.append(os.path.basename(file_path))
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
        
        return cleaned_files