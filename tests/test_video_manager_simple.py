# Simple isolated test for video_manager core functions

import pytest
import os
import sys
import importlib.util
from unittest.mock import patch, mock_open, MagicMock

# Import video_manager directly
video_manager_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'utils', 'video_manager.py')
spec = importlib.util.spec_from_file_location("video_manager", video_manager_path)
video_manager = importlib.util.module_from_spec(spec)
sys.modules["video_manager"] = video_manager
spec.loader.exec_module(video_manager)

class TestVideoManagerSimple:
    """Simple test for core video_manager functions"""
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    def test_initialization(self, mock_paths, mock_ensure):
        """Test basic initialization"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        assert manager.base_path == '/test'
        assert mock_ensure.called
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('builtins.open', mock_open(read_data=b"test content"))
    @patch('os.path.getsize', return_value=100)
    def test_generate_video_id(self, mock_size, mock_paths, mock_ensure):
        """Test video ID generation"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        video_id = manager._generate_video_id('/test/video.mp4')
        
        assert isinstance(video_id, str)
        assert len(video_id) > 10
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('os.path.exists', return_value=True)
    @patch('json.load', return_value={'video_123': {'original_path': '/test/video.mp4', 'filename': 'video.mp4'}})
    @patch('builtins.open', mock_open())
    def test_get_video_paths(self, mock_paths, mock_ensure):
        """Test getting video paths"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        paths = manager.get_video_paths('video_123')
        
        assert 'input' in paths
        assert 'temp_dir' in paths
        assert 'output_dir' in paths
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('os.listdir', return_value=['video.mp4'])
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.exists', return_value=True)
    @patch('json.load', return_value={'video_123': {'original_path': '/input/video.mp4', 'filename': 'video.mp4'}})
    @patch('builtins.open', mock_open())
    def test_get_current_video_id(self, mock_paths, mock_ensure):
        """Test getting current video ID"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        current_id = manager.get_current_video_id()
        
        assert current_id == 'video_123'
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('os.listdir', return_value=['video1.mp4', 'video2.avi', 'text.txt'])
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.exists', return_value=True)
    def test_list_video_files(self, mock_paths, mock_ensure):
        """Test listing video files"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        video_files = manager.list_video_files()
        
        # Should return only video files
        assert len(video_files) == 2
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('json.load', return_value={})
    @patch('builtins.open', mock_open())
    @patch('os.path.exists', return_value=True)
    def test_invalid_video_id(self, mock_paths, mock_ensure):
        """Test error handling for invalid video ID"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        
        with pytest.raises(ValueError, match="Video ID .* not found"):
            manager.get_video_paths('nonexistent')
    
    @patch.object(video_manager, 'get_storage_paths')
    def test_singleton_pattern(self, mock_paths):
        """Test singleton pattern"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        with patch.object(video_manager.VideoFileManager, 'ensure_directory_structure'):
            manager1 = video_manager.get_video_manager()
            manager2 = video_manager.get_video_manager()
            
            assert manager1 is manager2
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('os.listdir', return_value=[])
    @patch('os.path.exists', return_value=True)
    def test_empty_directory(self, mock_paths, mock_ensure):
        """Test empty directory handling"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        video_files = manager.list_video_files()
        
        assert video_files == []
    
    @patch.object(video_manager.VideoFileManager, 'ensure_directory_structure')
    @patch.object(video_manager, 'get_storage_paths')
    @patch('builtins.open', side_effect=FileNotFoundError("No metadata"))
    @patch('os.path.exists', return_value=True)
    def test_metadata_error(self, mock_paths, mock_ensure):
        """Test metadata error handling"""
        mock_paths.return_value = {
            'base': '/test', 'input': '/input', 'temp': '/temp', 'output': '/output'
        }
        
        manager = video_manager.VideoFileManager()
        current_id = manager.get_current_video_id()
        
        assert current_id is None