"""
Video format selection utilities
Single responsibility: Handle video format selection and optimization
"""

from typing import Dict, Optional


class VideoFormatSelector:
    """Handle video format selection based on resolution and codec preferences"""
    
    def __init__(self):
        # Preferred codecs in order of preference
        self.video_codecs = ['avc1', 'h264', 'vp9', 'av01']
        self.audio_codecs = ['mp4a', 'aac', 'opus']
        
        # Resolution priority for "best" selection
        self.resolution_priority = [2160, 1440, 1080, 720, 480, 360]
        
    def get_optimal_format(self, resolution: str) -> str:
        """
        Get optimal format string for yt-dlp based on VideoLingo's processing needs
        Prioritizes H.264+AAC combination for best processing performance
        
        Args:
            resolution (str): Target resolution ('best', '1080', '720', etc.)
            
        Returns:
            str: yt-dlp format selector string
        """
        if resolution == 'best':
            return self._get_best_quality_format()
        else:
            return self._get_resolution_specific_format(resolution)
    
    def _get_best_quality_format(self) -> str:
        """Generate format string for best quality with codec preferences"""
        format_parts = []
        
        # Build format selectors for each resolution in priority order
        for res in self.resolution_priority:
            for codec in self.video_codecs:
                format_parts.append(f"bestvideo[height<={res}][vcodec^={codec}]")
        
        # Add general fallbacks
        format_parts.extend([
            "bestvideo[ext=mp4]",
            "bestvideo"
        ])
        
        # Audio preferences
        audio_parts = []
        for codec in self.audio_codecs:
            audio_parts.append(f"bestaudio[acodec^={codec}]")
        audio_parts.extend(["bestaudio[ext=m4a]", "bestaudio"])
        
        # Combined format with fallbacks
        video_selector = "/".join(format_parts)
        audio_selector = "/".join(audio_parts)
        
        # Build final format string with fallbacks
        return f"({video_selector})+({audio_selector})/best[vcodec^=avc1]/best[vcodec^=h264]/best[ext=mp4]/best"
    
    def _get_resolution_specific_format(self, resolution: str) -> str:
        """Generate format string for specific resolution"""
        height_filter = f"[height<={resolution}]"
        
        # Video format preferences
        video_parts = []
        for codec in self.video_codecs:
            video_parts.append(f"bestvideo{height_filter}[vcodec^={codec}]")
        video_parts.append(f"bestvideo{height_filter}[ext=mp4]")
        video_parts.append(f"bestvideo{height_filter}")
        
        # Audio format preferences  
        audio_parts = []
        for codec in self.audio_codecs:
            audio_parts.append(f"bestaudio[acodec^={codec}]")
        audio_parts.extend(["bestaudio[ext=m4a]", "bestaudio"])
        
        # Fallback formats
        fallback_parts = []
        for codec in self.video_codecs:
            fallback_parts.append(f"best{height_filter}[vcodec^={codec}]")
        fallback_parts.extend([
            f"best{height_filter}[ext=mp4]",
            f"best{height_filter}"
        ])
        
        video_selector = "/".join(video_parts)
        audio_selector = "/".join(audio_parts)
        fallback_selector = "/".join(fallback_parts)
        
        return f"({video_selector})+({audio_selector})/({fallback_selector})"
    
    def get_supported_resolutions(self) -> Dict[str, str]:
        """
        Get mapping of display names to resolution values
        
        Returns:
            dict: Mapping of display names to internal resolution values
        """
        return {
            "360p": "360",
            "720p": "720", 
            "1080p": "1080",
            "1440p": "1440",
            "2160p (4K)": "2160",
            "Best Quality": "best"
        }
    
    def validate_resolution(self, resolution: str) -> bool:
        """
        Validate if resolution is supported
        
        Args:
            resolution (str): Resolution to validate
            
        Returns:
            bool: True if resolution is supported
        """
        supported_values = set(self.get_supported_resolutions().values())
        return resolution in supported_values
    
    def get_format_info(self, resolution: str) -> Dict[str, str]:
        """
        Get detailed information about format selection for given resolution
        
        Args:
            resolution (str): Target resolution
            
        Returns:
            dict: Format information including codecs, containers, etc.
        """
        info = {
            'resolution': resolution,
            'video_codecs': self.video_codecs,
            'audio_codecs': self.audio_codecs,
            'format_string': self.get_optimal_format(resolution)
        }
        
        if resolution == 'best':
            info['description'] = 'Highest available quality with optimal codec selection'
            info['max_resolution'] = max(self.resolution_priority)
        else:
            info['description'] = f'Best quality up to {resolution}p with optimal codecs'
            info['max_resolution'] = int(resolution)
            
        return info