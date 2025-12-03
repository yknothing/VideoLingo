# Comprehensive Unit Tests for VideoFileManager
# Tests core/utils/video_manager.py with 80%+ coverage target

import pytest
import os
import tempfile
import shutil
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

from core.utils.video_manager import VideoFileManager, video_manager


class TestVideoFileManager:
    """Comprehensive test suite for VideoFileManager"""

    @pytest.fixture
    def mock_paths(self):
        """Mock storage paths for testing"""
        return {
            'input': '/tmp/test_input',
            'temp': '/tmp/test_temp',
            'output': '/tmp/test_output'
        }

    @pytest.fixture
    def temp_video_manager(self, mock_paths):
        """Create VideoFileManager with mocked paths"""
        with patch('core.utils.video_manager.get_storage_paths', return_value=mock_paths):
            with patch('os.makedirs'):
                manager = VideoFileManager()
                return manager

    @pytest.fixture
    def sample_video_file(self, tmp_path):
        """Create a sample video file for testing"""
        video_file = tmp_path / "test_video.mp4"
        # Create a file with some content for hash generation
        video_file.write_bytes(b"fake video content for testing" * 1000)
        return str(video_file)

    def test_init_creates_directory_structure(self):
        """Test that VideoFileManager initializes and creates directories"""
        mock_paths = {
            'input': '/tmp/input',
            'temp': '/tmp/temp',
            'output': '/tmp/output'
        }
        
        with patch('core.utils.video_manager.get_storage_paths', return_value=mock_paths):
            with patch('os.makedirs') as mock_makedirs:
                manager = VideoFileManager()
                
                # Should call makedirs for each path
                expected_calls = [call(path, exist_ok=True) for path in mock_paths.values()]
                mock_makedirs.assert_has_calls(expected_calls, any_order=True)

    def test_check_disk_space_sufficient(self, temp_video_manager):
        """Test disk space check with sufficient space"""
        with patch('shutil.disk_usage') as mock_usage:
            # Mock 5GB available space
            mock_usage.return_value = MagicMock(free=5 * 1024 * 1024 * 1024)
            
            result = temp_video_manager.check_disk_space(1000)  # Need 1GB
            
            assert result['available'] is True
            assert result['free_mb'] == 5 * 1024
            assert result['required_mb'] == 1000

    def test_generate_video_id_success(self, temp_video_manager, sample_video_file):
        """Test video ID generation from file"""
        video_id = temp_video_manager.generate_video_id(sample_video_file)
        
        # Should contain hash_name_timestamp pattern
        parts = video_id.split('_')
        assert len(parts) == 3
        assert len(parts[0]) == 8  # file hash (8 chars)
        assert len(parts[1]) == 4  # name hash (4 chars)
        assert len(parts[2]) == 6  # timestamp (6 chars)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
