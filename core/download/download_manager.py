"""
Main download manager - Orchestrates the download process
Single responsibility: Coordinate download operations using other specialized components
"""

import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .filename_utils import sanitize_filename, generate_safe_filename
from .video_validator import VideoFileValidator, PartialDownloadCleaner
from .format_selector import VideoFormatSelector
from .error_handler import DownloadErrorHandler


@dataclass
class DownloadConfig:
    """Configuration for download operations"""
    save_path: Optional[str] = None
    resolution: str = "1080"
    max_retries: int = 3
    initial_wait: int = 5
    min_file_size_mb: float = 1.0
    use_cookies: bool = True
    proxy_url: Optional[str] = None
    max_filename_length: int = 200


class DownloadManager:
    """
    Main download manager that orchestrates video downloads
    Uses composition instead of inheritance to avoid god object pattern
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        self.config = config or DownloadConfig()
        
        # Composed components following SRP
        self.validator = VideoFileValidator(min_size_mb=self.config.min_file_size_mb)
        self.cleaner = PartialDownloadCleaner()
        self.format_selector = VideoFormatSelector()
        self.error_handler = DownloadErrorHandler()
        
    def download_video(
        self, 
        url: str, 
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Download video with comprehensive error handling and validation
        
        Args:
            url (str): Video URL to download
            progress_callback (callable, optional): Progress callback function
            
        Returns:
            str: Path to downloaded video file
            
        Raises:
            Exception: If download fails after all retries
        """
        # Ensure directories exist
        save_path = self._get_save_path()
        os.makedirs(save_path, exist_ok=True)
        
        # Clean up any existing partial downloads
        self.cleaner.cleanup_partial_downloads(save_path)
        
        # Define download function for retry mechanism
        def attempt_download():
            try:
                return self._download_via_command(url, save_path, progress_callback)
            except Exception as e:
                print(f"External yt-dlp failed: {e}. Trying Python API...")
                return self._download_via_python_api(url, save_path, progress_callback)
        
        # Execute download with intelligent retry
        downloaded_file = self.error_handler.intelligent_retry_download(
            attempt_download, 
            max_retries=self.config.max_retries,
            initial_wait=self.config.initial_wait
        )
        
        # Enhanced validation and fallback logic
        if not downloaded_file or not os.path.exists(downloaded_file):
            print("Download function returned invalid path, searching for downloaded file...")
            downloaded_file = self._find_downloaded_file(save_path)
            
            if not downloaded_file:
                raise Exception("No video file found after download completion")
        
        # Final validation
        is_valid, validation_msg = self.validator.validate_video_file(downloaded_file)
        if not is_valid:
            print(f"Warning: Downloaded file validation failed: {validation_msg}")
        
        # Update file timestamp
        self._update_file_timestamp(downloaded_file)
        
        file_size = os.path.getsize(downloaded_file) / (1024 * 1024)
        print(f"Download completed successfully: {os.path.basename(downloaded_file)} ({file_size:.1f}MB)")
        
        return downloaded_file
    
    def _get_save_path(self) -> str:
        """Get the save path for downloads"""
        if self.config.save_path:
            return self.config.save_path
        
        # Try to get from config utils if available
        try:
            from core.utils.config_utils import get_storage_paths
            paths = get_storage_paths()
            return paths['input']
        except ImportError:
            # Fallback to current directory
            return os.path.join(os.getcwd(), 'input')
    
    def _find_downloaded_file(self, save_path: str) -> Optional[str]:
        """Find downloaded file using fallback logic"""
        try:
            from core.utils.config_utils import load_key
            allowed_formats = load_key("allowed_video_formats")
        except (ImportError, KeyError):
            allowed_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
        
        # First try to find most recent file
        recent_file = self.validator.find_most_recent_video_file(save_path, allowed_formats)
        if recent_file:
            return recent_file
        
        # Then try to find best file
        return self.validator.find_best_video_file(save_path, allowed_formats)
    
    def _download_via_command(
        self, 
        url: str, 
        save_path: str, 
        progress_callback: Optional[Callable]
    ) -> str:
        """Download using external yt-dlp command"""
        format_string = self.format_selector.get_optimal_format(self.config.resolution)
        
        cmd = [
            'yt-dlp',
            '--format', format_string,
            '--output', f'{save_path}/%(title).{self.config.max_filename_length}s.%(ext)s',
            '--restrict-filenames',
            '--no-playlist',
            '--merge-output-format', 'mp4',
            '--no-part',
            '--no-mtime',
            url
        ]
        
        # Add cookies if enabled
        if self.config.use_cookies:
            cmd.extend(['--cookies-from-browser', 'chrome'])
        
        # Add proxy if configured
        if self.config.proxy_url:
            cmd.extend(['--proxy', self.config.proxy_url])
        elif 'HTTPS_PROXY' in os.environ:
            cmd.extend(['--proxy', os.environ['HTTPS_PROXY']])
        
        return self._execute_command(cmd, save_path, progress_callback)
    
    def _download_via_python_api(
        self, 
        url: str, 
        save_path: str, 
        progress_callback: Optional[Callable]
    ) -> str:
        """Download using Python yt-dlp API"""
        try:
            from yt_dlp import YoutubeDL
        except ImportError:
            raise Exception("yt-dlp not available for Python API download")
        
        downloaded_holder = {'path': None}
        
        def progress_hook(d):
            if progress_callback and d['status'] == 'downloading':
                self._handle_progress_callback(d, progress_callback)
            elif progress_callback and d['status'] == 'finished':
                progress_callback({'progress': 1.0, 'status': 'finished'})
                downloaded_holder['path'] = d.get('filename')
        
        format_string = self.format_selector.get_optimal_format(self.config.resolution)
        
        ydl_opts = {
            'format': format_string,
            'outtmpl': f'{save_path}/%(title).{self.config.max_filename_length}s.%(ext)s',
            'restrictfilenames': True,
            'windowsfilenames': True,
            'noplaylist': True,
            'progress_hooks': [progress_hook] if progress_callback else [],
            'merge_output_format': 'mp4'
        }
        
        # Add network resilience options
        ydl_opts.update({
            'retries': 15,
            'fragment_retries': 15,
            'file_access_retries': 15,
            'continuedl': True,
            'socket_timeout': 30,
            'sleep_interval': 1,
            'max_sleep_interval': 5
        })
        
        # Add cookies and proxy settings
        self._configure_network_options(ydl_opts)
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Get final path
        final_path = downloaded_holder['path']
        if not final_path or not os.path.exists(final_path):
            final_path = self._find_downloaded_file(save_path)
        
        if not final_path:
            raise Exception("Download completed but could not locate the downloaded file")
        
        return final_path
    
    def _execute_command(
        self, 
        cmd: list, 
        save_path: str, 
        progress_callback: Optional[Callable]
    ) -> str:
        """Execute yt-dlp command and parse output"""
        import re
        
        dest_file_holder = {'path': None}
        
        def parse_progress_line(line: str):
            """Parse yt-dlp progress line and call callback"""
            if progress_callback and '[download]' in line:
                progress_match = re.search(r'\[download\]\s+(\d+\.?\d*)%\s+of\s+([\d\.]+\w+)', line)
                if progress_match:
                    percent = float(progress_match.group(1)) / 100.0
                    total_size = progress_match.group(2)
                    
                    speed_match = re.search(r'at\s+([\d\.]+\w+/s)', line)
                    eta_match = re.search(r'ETA\s+(\d+:\d+)', line)
                    
                    progress_callback({
                        'progress': percent,
                        'total_size_str': total_size,
                        'speed_str': speed_match.group(1) if speed_match else '',
                        'eta_str': eta_match.group(1) if eta_match else '',
                        'status': 'downloading'
                    })
            
            # Enhanced destination file detection
            self._parse_destination_file(line, dest_file_holder, save_path)
        
        # Run process with real-time output parsing
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            universal_newlines=True
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line)
            parse_progress_line(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            error_output = ''.join(output_lines)
            raise Exception(f"yt-dlp command failed: {error_output}")
        
        if progress_callback:
            progress_callback({'progress': 1.0, 'status': 'finished'})
        
        # Get final path
        final_path = dest_file_holder['path']
        if not final_path or not os.path.exists(final_path):
            final_path = self._find_downloaded_file(save_path)
        
        if not final_path:
            raise Exception("Download completed but could not locate the downloaded file")
        
        return final_path
    
    def _parse_destination_file(self, line: str, dest_holder: dict, save_path: str):
        """Parse line to extract destination file path"""
        import re
        
        if dest_holder['path'] is not None:
            return
        
        # Multiple patterns to detect destination file
        patterns = [
            r'\[download\]\s+Destination:\s+(.+)',
            r'\[Merger\]\s+Merging formats into\s+"([^"]+)"',
            r'\[download\]\s+(.+)\s+has already been downloaded',
            r'\[ffmpeg\]\s+Merging formats into\s+"([^"]+)"',
            r'\[download\]\s+100%\s+of\s+([^\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                detected_path = match.group(1).strip()
                if os.path.isabs(detected_path) or '/' in detected_path:
                    dest_holder['path'] = detected_path
                else:
                    dest_holder['path'] = os.path.join(save_path, detected_path)
                break
    
    def _handle_progress_callback(self, progress_data: dict, callback: Callable):
        """Handle progress callback for Python API"""
        try:
            downloaded = progress_data.get('downloaded_bytes', 0)
            total = progress_data.get('total_bytes') or progress_data.get('total_bytes_estimate')
            
            if total and total > 0:
                progress = downloaded / total
                speed = progress_data.get('speed', 0) or 0
                eta = progress_data.get('eta', 0) or 0
                
                callback({
                    'progress': progress,
                    'downloaded': downloaded,
                    'total': total,
                    'speed': speed,
                    'eta': eta,
                    'status': 'downloading'
                })
        except Exception:
            # Ignore progress callback errors
            pass
    
    def _configure_network_options(self, ydl_opts: dict):
        """Configure network options for yt-dlp"""
        # Try browser cookies first
        try:
            if self.config.use_cookies:
                ydl_opts["cookiesfrombrowser"] = ("chrome",)
        except Exception:
            # If browser cookies fail, continue without
            pass
        
        # Add proxy if configured
        if self.config.proxy_url:
            ydl_opts["proxy"] = self.config.proxy_url
        elif "HTTPS_PROXY" in os.environ:
            ydl_opts["proxy"] = os.environ["HTTPS_PROXY"]
        elif "HTTP_PROXY" in os.environ:
            ydl_opts["proxy"] = os.environ["HTTP_PROXY"]
    
    def _update_file_timestamp(self, file_path: str):
        """Update file timestamp to current time"""
        if file_path and os.path.exists(file_path):
            import time
            current_time = time.time()
            os.utime(file_path, (current_time, current_time))
    
    def get_supported_resolutions(self) -> Dict[str, str]:
        """Get supported resolutions"""
        return self.format_selector.get_supported_resolutions()
    
    def validate_url(self, url: str) -> bool:
        """Basic URL validation"""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None