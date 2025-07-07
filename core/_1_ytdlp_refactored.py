"""
Refactored video download module
Replaces the monolithic _1_ytdlp.py with clean, modular architecture

This module maintains backward compatibility while using the new modular design
"""

import os
import sys
from typing import Optional, Callable

# Import the new modular components
from core.download import (
    DownloadManager, 
    DownloadConfig,
    sanitize_filename,
    VideoFileValidator,
    VideoFormatSelector
)


# Maintain backward compatibility with existing code
def download_video_ytdlp(
    url: str, 
    save_path: Optional[str] = None, 
    resolution: Optional[str] = None, 
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Main download function - refactored with improved architecture
    Maintains backward compatibility with existing VideoLingo code
    
    Args:
        url (str): Video URL to download
        save_path (str, optional): Directory to save video
        resolution (str, optional): Target resolution
        progress_callback (callable, optional): Progress callback function
        
    Returns:
        str: Path to downloaded video file
    """
    # Load configuration from VideoLingo config if available
    config = _create_download_config(save_path, resolution)
    
    # Create download manager with configuration
    manager = DownloadManager(config)
    
    # Execute download
    return manager.download_video(url, progress_callback)


def find_video_files(save_path: Optional[str] = None) -> str:
    """
    Find video files in the specified directory
    Maintains backward compatibility
    
    Args:
        save_path (str, optional): Directory to search
        
    Returns:
        str: Path to found video file
        
    Raises:
        ValueError: If no valid video files found
    """
    if save_path is None:
        try:
            from core.utils.config_utils import get_storage_paths
            paths = get_storage_paths()
            save_path = paths['input']
        except (ImportError, KeyError) as e:
            print(f"Config not available, using default path: {e}")
            save_path = os.path.join(os.getcwd(), 'input')
    
    # Get allowed formats from config
    try:
        from core.utils.config_utils import load_key
        allowed_formats = load_key("allowed_video_formats")
    except (ImportError, KeyError) as e:
        print(f"Config not available, using default formats: {e}")
        allowed_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
    
    # Use validator to find files
    validator = VideoFileValidator()
    
    # First try to find the best file
    best_file = validator.find_best_video_file(save_path, allowed_formats)
    if best_file:
        return best_file
    
    # Then try most recent
    recent_file = validator.find_most_recent_video_file(save_path, allowed_formats)
    if recent_file:
        return recent_file
    
    raise ValueError("No video files found. Please check if the download was successful.")


def get_optimal_format(resolution: str) -> str:
    """
    Get optimal format for given resolution
    Maintains backward compatibility
    
    Args:
        resolution (str): Target resolution
        
    Returns:
        str: yt-dlp format string
    """
    selector = VideoFormatSelector()
    return selector.get_optimal_format(resolution)


def validate_video_file(file_path: str, expected_min_size_mb: float = 1.0) -> tuple:
    """
    Validate video file
    Maintains backward compatibility
    
    Args:
        file_path (str): Path to video file
        expected_min_size_mb (float): Minimum expected file size in MB
        
    Returns:
        tuple: (is_valid, error_message)
    """
    validator = VideoFileValidator(min_size_mb=expected_min_size_mb)
    return validator.validate_video_file(file_path)


def categorize_download_error(error_msg: str) -> tuple:
    """
    Categorize download error
    Maintains backward compatibility
    
    Args:
        error_msg (str): Error message
        
    Returns:
        tuple: (category, is_retryable, suggested_wait_time)
    """
    from core.download.error_handler import DownloadErrorHandler
    
    handler = DownloadErrorHandler()
    category, is_retryable, wait_time = handler.categorize_download_error(error_msg)
    return category.value, is_retryable, wait_time


def intelligent_retry_download(download_func: Callable, max_retries: int = 3, initial_wait: int = 5):
    """
    Intelligent retry mechanism
    Maintains backward compatibility
    
    Args:
        download_func (callable): Function to retry
        max_retries (int): Maximum retries
        initial_wait (int): Initial wait time
        
    Returns:
        Any: Result of download function
    """
    from core.download.error_handler import DownloadErrorHandler
    
    handler = DownloadErrorHandler()
    return handler.intelligent_retry_download(download_func, max_retries, initial_wait)


def cleanup_partial_downloads(save_path: str) -> list:
    """
    Clean up partial downloads
    Maintains backward compatibility
    
    Args:
        save_path (str): Directory to clean
        
    Returns:
        list: List of cleaned files
    """
    from core.download.video_validator import PartialDownloadCleaner
    
    cleaner = PartialDownloadCleaner()
    return cleaner.cleanup_partial_downloads(save_path)


def find_most_recent_video_file(save_path: str) -> Optional[str]:
    """
    Find most recent video file
    Maintains backward compatibility
    
    Args:
        save_path (str): Directory to search
        
    Returns:
        str or None: Path to most recent video file
    """
    try:
        from core.utils.config_utils import load_key
        allowed_formats = load_key("allowed_video_formats")
    except (ImportError, KeyError) as e:
        print(f"Config not available, using default formats: {e}")
        allowed_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
    
    validator = VideoFileValidator()
    return validator.find_most_recent_video_file(save_path, allowed_formats)


def find_best_video_file(save_path: str, allowed_formats: list) -> Optional[str]:
    """
    Find best video file
    Maintains backward compatibility
    
    Args:
        save_path (str): Directory to search
        allowed_formats (list): Allowed file extensions
        
    Returns:
        str or None: Path to best video file
    """
    validator = VideoFileValidator()
    return validator.find_best_video_file(save_path, allowed_formats)


def update_ytdlp():
    """
    Update yt-dlp package
    Maintains backward compatibility
    """
    import subprocess
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        print("yt-dlp updated successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to update yt-dlp: {e}")
    
    try:
        from yt_dlp import YoutubeDL
        return YoutubeDL
    except ImportError as e:
        raise Exception(f"yt-dlp not available after update attempt: {e}")


def _create_download_config(save_path: Optional[str], resolution: Optional[str]) -> DownloadConfig:
    """
    Create download configuration from VideoLingo settings
    
    Args:
        save_path (str, optional): Override save path
        resolution (str, optional): Override resolution
        
    Returns:
        DownloadConfig: Configuration object
    """
    config = DownloadConfig()
    
    # Set save path
    if save_path:
        config.save_path = save_path
    else:
        try:
            from core.utils.config_utils import get_storage_paths
            paths = get_storage_paths()
            config.save_path = paths['input']
        except (ImportError, KeyError) as e:
            print(f"Config not available, using default save path: {e}")
            config.save_path = os.path.join(os.getcwd(), 'input')
    
    # Set resolution
    if resolution:
        config.resolution = resolution
    else:
        try:
            from core.utils.config_utils import load_key
            config.resolution = str(load_key('ytb_resolution'))
        except (ImportError, KeyError) as e:
            print(f"Config not available, using default resolution: {e}")
            config.resolution = '1080'
    
    # Set proxy if available
    try:
        from core.utils.config_utils import load_key
        proxy_url = load_key("youtube.proxy")
        if proxy_url:
            config.proxy_url = proxy_url
    except (ImportError, KeyError) as e:
        print(f"Proxy config not available, using defaults: {e}")
        # Continue without proxy configuration
    
    return config


# Legacy function aliases for maximum compatibility
download_via_python_api = download_video_ytdlp
download_via_command = download_video_ytdlp


# Main entry point for testing
if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = resolution if resolution.isdigit() else '1080'
    
    try:
        downloaded_file = download_video_ytdlp(url, resolution=resolution)
        print(f"🎥 Video has been downloaded to {downloaded_file}")
    except Exception as e:
        print(f"❌ Download failed: {e}")