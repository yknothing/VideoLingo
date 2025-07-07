# Isolated test for video_manager to avoid torch import conflicts

import pytest
import os
import sys
import importlib.util
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import video_manager directly without going through core/__init__.py
video_manager_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'utils', 'video_manager.py')
spec = importlib.util.spec_from_file_location("video_manager", video_manager_path)
video_manager = importlib.util.module_from_spec(spec)
sys.modules["video_manager"] = video_manager
spec.loader.exec_module(video_manager)

class TestVideoManagerIsolated:
    """Test video_manager module in isolation"""
    
    def test_video_file_manager_initialization(self):
        """Test VideoFileManager initialization"""
        with patch('os.makedirs') as mock_makedirs, \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test/base',
                 'input': '/test/input', 
                 'temp': '/test/temp',
                 'output': '/test/output'
             }), \
             patch.object(video_manager.VideoFileManager, 'ensure_directory_structure'):
            
            manager = video_manager.VideoFileManager()
            
            # Should set paths correctly
            assert manager.base_path == '/test/base'
            assert manager.temp_path == '/test/temp'
            assert manager.output_path == '/test/output'
    
    def test_generate_video_id(self):
        """Test video ID generation"""
        test_content = b"test video content"
        
        with patch('builtins.open', mock_open(read_data=test_content)), \
             patch('os.path.getsize', return_value=len(test_content)), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }), \
             patch.object(video_manager.VideoFileManager, 'ensure_directory_structure'):
            
            manager = video_manager.VideoFileManager()
            video_id = manager._generate_video_id('/test/video.mp4')
            
            # Should be a string with timestamp and hash
            assert isinstance(video_id, str)
            assert len(video_id) > 10
    
    def test_register_video_new(self):
        """Test registering a new video"""
        test_metadata = {}
        
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1000), \
             patch('builtins.open', mock_open(read_data=b"test content")), \
             patch('json.dump') as mock_json_dump, \
             patch('json.load', return_value=test_metadata), \
             patch('shutil.copy2') as mock_copy, \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            video_id = manager.register_video('/test/input/video.mp4')
            
            # Should return a video ID
            assert isinstance(video_id, str)
            # Should save metadata
            assert mock_json_dump.called
    
    def test_get_video_paths(self):
        """Test getting video paths for specific ID"""
        test_metadata = {
            'video_123': {
                'original_path': '/test/input/video.mp4',
                'filename': 'video.mp4'
            }
        }
        
        with patch('json.load', return_value=test_metadata), \
             patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            paths = manager.get_video_paths('video_123')
            
            assert 'input' in paths
            assert 'temp_dir' in paths
            assert 'output_dir' in paths
            assert paths['input'] == '/test/input/video.mp4'
    
    def test_get_current_video_id(self):
        """Test getting current video ID"""
        with patch('os.listdir', return_value=['video.mp4']), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            
            # Mock metadata with current video
            test_metadata = {
                'video_123': {
                    'original_path': '/test/input/video.mp4',
                    'filename': 'video.mp4'
                }
            }
            
            with patch('json.load', return_value=test_metadata), \
                 patch('builtins.open', mock_open()):
                
                current_id = manager.get_current_video_id()
                assert current_id == 'video_123'
    
    def test_list_video_files(self):
        """Test listing video files"""
        with patch('os.listdir', return_value=['video1.mp4', 'video2.avi', 'text.txt']), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            video_files = manager.list_video_files()
            
            # Should return only video files
            assert len(video_files) == 2
            assert 'video1.mp4' in [os.path.basename(f) for f in video_files]
            assert 'video2.avi' in [os.path.basename(f) for f in video_files]
    
    def test_safe_overwrite_with_backup(self):
        """Test safe overwrite operation with backup"""
        with patch('os.path.exists', return_value=True), \
             patch('shutil.copy2') as mock_copy, \
             patch('os.makedirs') as mock_makedirs, \
             patch('builtins.print') as mock_print, \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            manager.safe_overwrite_temp_dir('video_123')
            
            # Should create backup and log operation
            assert mock_makedirs.called
            assert mock_print.called
    
    def test_error_handling_invalid_video_id(self):
        """Test error handling for invalid video ID"""
        with patch('json.load', return_value={}), \
             patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            
            with pytest.raises(ValueError, match="Video ID .* not found"):
                manager.get_video_paths('nonexistent_video')
    
    def test_get_video_manager_singleton(self):
        """Test singleton video manager"""
        with patch.object(video_manager, 'get_storage_paths', return_value={
             'base': '/test', 'input': '/test/input',
             'temp': '/test/temp', 'output': '/test/output'
         }):
            
            manager1 = video_manager.get_video_manager()
            manager2 = video_manager.get_video_manager()
            
            # Should return same instance
            assert manager1 is manager2
    
    def test_edge_case_empty_directory(self):
        """Test edge case with empty directory"""
        with patch('os.listdir', return_value=[]), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            video_files = manager.list_video_files()
            
            # Should return empty list
            assert video_files == []
    
    def test_metadata_error_handling(self):
        """Test metadata file error handling"""
        with patch('builtins.open', side_effect=FileNotFoundError("Metadata not found")), \
             patch('os.path.exists', return_value=True), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            current_id = manager.get_current_video_id()
            
            # Should handle missing metadata gracefully
            assert current_id is None
    
    def test_complex_video_id_generation(self):
        """Test video ID generation with various file characteristics"""
        test_cases = [
            (b"small", "small.mp4"),
            (b"x" * 1000, "medium.avi"),
            (b"y" * 10000, "large.mkv")
        ]
        
        with patch.object(video_manager, 'get_storage_paths', return_value={
             'base': '/test', 'input': '/test/input',
             'temp': '/test/temp', 'output': '/test/output'
         }):
            
            manager = video_manager.VideoFileManager()
            
            for content, filename in test_cases:
                with patch('builtins.open', mock_open(read_data=content)), \
                     patch('os.path.getsize', return_value=len(content)):
                    
                    video_id = manager._generate_video_id(f'/test/{filename}')
                    
                    # Should generate valid ID for all file sizes
                    assert isinstance(video_id, str)
                    assert len(video_id) > 10
    
    def test_video_registration_with_copy(self):
        """Test video registration requiring file copy"""
        with patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1000), \
             patch('builtins.open', mock_open(read_data=b"content")), \
             patch('shutil.copy2') as mock_copy, \
             patch('json.dump'), \
             patch('json.load', return_value={}), \
             patch.object(video_manager, 'get_storage_paths', return_value={
                 'base': '/test', 'input': '/test/input',
                 'temp': '/test/temp', 'output': '/test/output'
             }):
            
            manager = video_manager.VideoFileManager()
            video_id = manager.register_video('/external/path/video.mp4')
            
            # Should copy file from external path
            assert mock_copy.called
            assert isinstance(video_id, str)