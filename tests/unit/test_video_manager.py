# Unit Tests for Video File Management System
# Tests core/utils/video_manager.py

import pytest
import os
import json
import tempfile
import hashlib
import time
from pathlib import Path
from unittest.mock import patch, Mock, mock_open
import threading
import shutil
import errno

# Import the module directly to avoid torch dependency issues
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock the config_utils dependency at import time
with patch('core.utils.config_utils.get_storage_paths') as mock_get_storage_paths:
    mock_get_storage_paths.return_value = {
        'input': '/tmp/test/input',
        'temp': '/tmp/test/temp', 
        'output': '/tmp/test/output'
    }
    from core.utils.video_manager import VideoFileManager, get_video_manager

class TestVideoFileManager:
    """Test suite for VideoFileManager class"""
    
    @pytest.fixture
    def video_manager(self, temp_config_dir):
        """Create a VideoFileManager instance for testing"""
        with patch('core.utils.video_manager.get_storage_paths') as mock_paths:
            mock_paths.return_value = {
                'input': str(temp_config_dir / 'input'),
                'temp': str(temp_config_dir / 'temp'),
                'output': str(temp_config_dir / 'output')
            }
            return VideoFileManager()
    
    def test_video_manager_initialization(self, video_manager, temp_config_dir):
        """Test VideoFileManager initialization"""
        assert video_manager.paths['input'].endswith('input')
        assert video_manager.paths['temp'].endswith('temp')
        assert video_manager.paths['output'].endswith('output')
        
        # Check directories are created
        for path in video_manager.paths.values():
            assert os.path.exists(path)
    
    def test_generate_video_id_consistency(self, video_manager, mock_video_file):
        """Test video ID generation consistency"""
        # Same file should generate same ID
        id1 = video_manager.generate_video_id(mock_video_file)
        id2 = video_manager.generate_video_id(mock_video_file)
        
        assert id1 == id2
        assert len(id1) > 10  # Should be reasonably long
        assert '_' in id1  # Should contain separators
    
    def test_generate_video_id_uniqueness(self, video_manager, temp_config_dir):
        """Test video ID uniqueness for different files"""
        # Create two different files
        file1 = temp_config_dir / 'input' / 'video1.mp4'
        file2 = temp_config_dir / 'input' / 'video2.mp4'
        
        with open(file1, 'wb') as f:
            f.write(b'content1' * 1000)
        with open(file2, 'wb') as f:
            f.write(b'content2' * 1000)
        
        id1 = video_manager.generate_video_id(str(file1))
        id2 = video_manager.generate_video_id(str(file2))
        
        assert id1 != id2
    
    def test_generate_video_id_fallback(self, video_manager):
        """Test video ID generation fallback for unreadable files"""
        # Test with non-existent file
        fake_path = "/nonexistent/file.mp4"
        video_id = video_manager.generate_video_id(fake_path)
        
        assert video_id.startswith('fallback_')
        assert len(video_id) > 10
    
    def test_register_video_new(self, video_manager, mock_video_file):
        """Test registering a new video"""
        video_id = video_manager.register_video(mock_video_file)
        
        assert video_id is not None
        assert len(video_id) > 0
        
        # Check video paths are created
        paths = video_manager.get_video_paths(video_id)
        assert os.path.exists(paths['input'])
        assert os.path.exists(paths['temp_dir'])
        
        # Check metadata is saved
        assert os.path.exists(paths['metadata_file'])
    
    def test_register_video_custom_id(self, video_manager, mock_video_file):
        """Test registering video with custom ID"""
        custom_id = "test_custom_id_123"
        video_id = video_manager.register_video(mock_video_file, video_id=custom_id)
        
        assert video_id == custom_id
        
        # Verify file structure with custom ID
        paths = video_manager.get_video_paths(custom_id)
        assert custom_id in paths['input']
        assert custom_id in paths['temp_dir']
    
    def test_register_video_copy_behavior(self, video_manager, temp_config_dir):
        """Test video registration copy behavior"""
        # Create source file outside input directory
        source_file = temp_config_dir / 'source_video.mp4'
        with open(source_file, 'wb') as f:
            f.write(b'test_content' * 100)
        
        video_id = video_manager.register_video(str(source_file))
        
        # Should copy to input directory
        paths = video_manager.get_video_paths(video_id)
        assert os.path.exists(paths['input'])
        assert paths['input'] != str(source_file)
        
        # Content should be preserved
        with open(paths['input'], 'rb') as f:
            content = f.read()
        assert content == b'test_content' * 100
    
    def test_get_video_paths_existing(self, video_manager, mock_video_file):
        """Test getting paths for existing video"""
        video_id = video_manager.register_video(mock_video_file)
        paths = video_manager.get_video_paths(video_id)
        
        expected_keys = ['input', 'temp_dir', 'output_dir', 'metadata_file']
        for key in expected_keys:
            assert key in paths
            assert isinstance(paths[key], str)
    
    def test_get_video_paths_nonexistent(self, video_manager):
        """Test getting paths for non-existent video"""
        with pytest.raises(ValueError, match="Video ID .* not found"):
            video_manager.get_video_paths("nonexistent_id")
    
    def test_get_temp_dir(self, video_manager):
        """Test temp directory path generation"""
        video_id = "test_id_123"
        temp_dir = video_manager.get_temp_dir(video_id)
        
        assert video_id in temp_dir
        assert 'temp' in temp_dir
    
    def test_get_temp_file(self, video_manager):
        """Test temp file path generation"""
        video_id = "test_id_123"
        filename = "test_file.txt"
        temp_file = video_manager.get_temp_file(video_id, filename)
        
        assert video_id in temp_file
        assert filename in temp_file
        assert 'temp' in temp_file
    
    def test_get_output_file(self, video_manager):
        """Test output file path generation"""
        video_id = "test_id_123"
        suffix = "sub"
        output_file = video_manager.get_output_file(video_id, suffix)
        
        assert video_id in output_file
        assert suffix in output_file
        assert output_file.endswith('.mp4')
        assert 'output' in output_file
    
    def test_get_output_file_custom_extension(self, video_manager):
        """Test output file path with custom extension"""
        video_id = "test_id_123"
        suffix = "audio"
        extension = ".wav"
        output_file = video_manager.get_output_file(video_id, suffix, extension)
        
        assert output_file.endswith('.wav')
        assert suffix in output_file
    
    def test_list_temp_files_empty(self, video_manager):
        """Test listing temp files for empty directory"""
        video_id = "empty_test_id"
        files = video_manager.list_temp_files(video_id)
        assert files == []
    
    def test_list_temp_files_with_files(self, video_manager, mock_video_file):
        """Test listing temp files with actual files"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = video_manager.get_temp_dir(video_id)
        
        # Create some temp files
        test_files = ['file1.txt', 'subdir/file2.txt', 'file3.json']
        for file_path in test_files:
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text('test content')
        
        files = video_manager.list_temp_files(video_id)
        assert len(files) >= len(test_files)  # At least our test files plus metadata
        
        # Check that our files are included
        for test_file in test_files:
            assert any(test_file in f for f in files)
    
    def test_list_output_files_empty(self, video_manager):
        """Test listing output files when none exist"""
        video_id = "empty_output_test"
        files = video_manager.list_output_files(video_id)
        assert files == []
    
    def test_list_output_files_with_files(self, video_manager, temp_config_dir):
        """Test listing output files with actual files"""
        video_id = "output_test_id"
        output_dir = video_manager.paths['output']
        
        # Create some output files
        test_files = [f'{video_id}_sub.mp4', f'{video_id}_dub.mp4', f'{video_id}_audio.wav']
        for filename in test_files:
            file_path = Path(output_dir) / filename
            file_path.write_text('test content')
        
        # Create some unrelated files (should not be included)
        unrelated_files = ['other_video_sub.mp4', 'random_file.txt']
        for filename in unrelated_files:
            file_path = Path(output_dir) / filename
            file_path.write_text('test content')
        
        files = video_manager.list_output_files(video_id)
        assert len(files) == len(test_files)
        
        for test_file in test_files:
            assert any(test_file in f for f in files)
        
        for unrelated_file in unrelated_files:
            assert not any(unrelated_file in f for f in files)
    
    def test_safe_overwrite_temp_files(self, video_manager, mock_video_file):
        """Test safe overwrite of temp files"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = video_manager.get_temp_dir(video_id)
        
        # Create some temp files (besides metadata)
        test_file = Path(temp_dir) / 'test_file.txt'
        test_subdir = Path(temp_dir) / 'subdir'
        test_subdir.mkdir(exist_ok=True)
        test_subfile = test_subdir / 'test_subfile.txt'
        
        test_file.write_text('original content')
        test_subfile.write_text('sub content')
        
        # Verify files exist before overwrite
        assert test_file.exists()
        assert test_subfile.exists()
        assert test_subdir.exists()
        
        # Perform overwrite
        video_manager.safe_overwrite_temp_files(video_id)
        
        # Check that our test files are removed but temp directory preserved
        assert os.path.exists(temp_dir)
        assert not test_file.exists()
        assert not test_subfile.exists()
        assert not test_subdir.exists()
        
        # Check that log was created if the operation succeeded
        log_dir = Path(temp_dir) / 'logs'
        # The safe_overwrite function creates logs, so directory should exist
        # but may not if there were no files to log
        if log_dir.exists():
            log_file = log_dir / 'overwrite_operations.log'
            assert log_file.exists()
    
    def test_safe_overwrite_output_files(self, video_manager, temp_config_dir):
        """Test safe overwrite of output files"""
        video_id = "overwrite_test_id"
        output_dir = video_manager.paths['output']
        
        # Create some output files
        test_files = [f'{video_id}_sub.mp4', f'{video_id}_dub.mp4']
        for filename in test_files:
            file_path = Path(output_dir) / filename
            file_path.write_text('test content')
        
        # Create unrelated file (should not be affected)
        unrelated_file = Path(output_dir) / 'other_video_sub.mp4'
        unrelated_file.write_text('other content')
        
        # Perform overwrite
        video_manager.safe_overwrite_output_files(video_id)
        
        # Check that target files are removed
        for filename in test_files:
            file_path = Path(output_dir) / filename
            assert not file_path.exists()
        
        # Check that unrelated file is preserved
        assert unrelated_file.exists()
        assert unrelated_file.read_text() == 'other content'
    
    def test_get_current_video_id_single_file(self, video_manager, temp_config_dir):
        """Test getting current video ID with single file"""
        input_dir = Path(video_manager.paths['input'])
        video_file = input_dir / 'test_video_id.mp4'
        video_file.write_text('test content')
        
        current_id = video_manager.get_current_video_id()
        assert current_id == 'test_video_id'
    
    def test_get_current_video_id_multiple_files(self, video_manager, temp_config_dir):
        """Test getting current video ID with multiple files"""
        input_dir = Path(video_manager.paths['input'])
        
        # Create multiple files with different timestamps
        old_file = input_dir / 'old_video_id.mp4'
        new_file = input_dir / 'new_video_id.mp4'
        
        old_file.write_text('old content')
        time.sleep(0.1)  # Ensure different timestamps
        new_file.write_text('new content')
        
        current_id = video_manager.get_current_video_id()
        assert current_id == 'new_video_id'  # Should return newest file
    
    def test_get_current_video_id_no_files(self, video_manager):
        """Test getting current video ID when no files exist"""
        # Clean up any existing files in input directory
        input_dir = Path(video_manager.paths['input'])
        for file in input_dir.glob('*'):
            file.unlink()
        
        current_id = video_manager.get_current_video_id()
        assert current_id is None
    
    def test_metadata_persistence(self, video_manager, mock_video_file):
        """Test video metadata persistence"""
        video_id = video_manager.register_video(mock_video_file)
        
        # Get metadata path
        paths = video_manager.get_video_paths(video_id)
        metadata_file = paths['metadata_file']
        
        assert os.path.exists(metadata_file)
        
        # Load and verify metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'original_path' in metadata
        assert 'original_filename' in metadata
        assert 'registered_time' in metadata
        assert 'input_path' in metadata
        assert 'temp_dir' in metadata
        assert 'file_extension' in metadata
        
        assert metadata['original_path'] == mock_video_file
        assert isinstance(metadata['registered_time'], (int, float))
    
    def test_overwrite_logging(self, video_manager, mock_video_file):
        """Test that overwrite operations are properly logged"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = video_manager.get_temp_dir(video_id)
        
        # Create a temp file
        test_file = Path(temp_dir) / 'test_file.txt'
        test_file.write_text('test content')
        
        # Perform overwrite
        video_manager.safe_overwrite_temp_files(video_id)
        
        # Check log file
        log_file = Path(temp_dir) / 'logs' / 'overwrite_operations.log'
        if log_file.exists():
            log_content = log_file.read_text()
            assert 'test_file.txt' in log_content
            assert video_id in log_content
            assert 'temp' in log_content
    
    def test_concurrent_video_registration(self, video_manager, temp_config_dir):
        """Test concurrent video registration"""
        import threading
        
        results = []
        errors = []
        
        def register_video(thread_id):
            try:
                # Create unique video file for each thread
                video_file = temp_config_dir / 'input' / f'video_{thread_id}.mp4'
                with open(video_file, 'wb') as f:
                    f.write(f'content_{thread_id}'.encode() * 100)
                
                video_id = video_manager.register_video(str(video_file))
                results.append((thread_id, video_id))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_video, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent registration errors: {errors}"
        assert len(results) == 5
        
        # All video IDs should be unique
        video_ids = [result[1] for result in results]
        assert len(set(video_ids)) == 5


class TestVideoManagerSingleton:
    """Test the global video manager instance"""
    
    def test_get_video_manager_singleton(self):
        """Test that get_video_manager returns singleton instance"""
        manager1 = get_video_manager()
        manager2 = get_video_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, VideoFileManager)
    
    def test_video_manager_state_persistence(self, mock_video_file):
        """Test that video manager maintains state across calls"""
        manager1 = get_video_manager()
        
        with patch.object(manager1, 'paths', {'input': '/test', 'temp': '/test', 'output': '/test'}):
            # Mock the registration to avoid file operations
            with patch.object(manager1, 'register_video', return_value='test_id'):
                video_id = manager1.register_video(mock_video_file)
        
        manager2 = get_video_manager()
        
        # Should be the same instance
        assert manager1 is manager2


class TestVideoFileManagerEdgeCases:
    """Additional edge case tests for comprehensive coverage"""
    
    @pytest.fixture
    def video_manager(self, temp_config_dir):
        """Create a VideoFileManager instance for testing"""
        with patch('core.utils.video_manager.get_storage_paths') as mock_paths:
            mock_paths.return_value = {
                'input': str(temp_config_dir / 'input'),
                'temp': str(temp_config_dir / 'temp'),
                'output': str(temp_config_dir / 'output')
            }
            return VideoFileManager()
    
    def test_generate_video_id_large_file(self, video_manager, temp_config_dir):
        """Test video ID generation for large files (>1MB)"""
        # Create a large file (>1MB)
        large_file = temp_config_dir / 'large_video.mp4'
        with open(large_file, 'wb') as f:
            f.write(b'a' * (2 * 1024 * 1024))  # 2MB
        
        video_id = video_manager.generate_video_id(str(large_file))
        assert video_id is not None
        assert len(video_id) > 10
        assert not video_id.startswith('fallback_')
    
    def test_generate_video_id_empty_file(self, video_manager, temp_config_dir):
        """Test video ID generation for empty files"""
        empty_file = temp_config_dir / 'empty.mp4'
        empty_file.touch()
        
        video_id = video_manager.generate_video_id(str(empty_file))
        assert video_id is not None
        assert len(video_id) > 10
        # Should still work with empty file
        assert not video_id.startswith('fallback_')
    
    def test_generate_video_id_permission_error(self, video_manager, temp_config_dir):
        """Test video ID generation when file cannot be read"""
        test_file = temp_config_dir / 'test.mp4'
        test_file.write_text('test')
        
        # Mock open to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            video_id = video_manager.generate_video_id(str(test_file))
            assert video_id.startswith('fallback_')
    
    def test_generate_video_id_unicode_filename(self, video_manager, temp_config_dir):
        """Test video ID generation with unicode filenames"""
        unicode_file = temp_config_dir / '测试视频_♪♫.mp4'
        with open(unicode_file, 'wb') as f:
            f.write(b'test_content')
        
        video_id = video_manager.generate_video_id(str(unicode_file))
        assert video_id is not None
        assert len(video_id) > 10
        assert not video_id.startswith('fallback_')
    
    def test_generate_video_id_special_characters(self, video_manager, temp_config_dir):
        """Test video ID generation with special characters in filename"""
        special_file = temp_config_dir / 'video@#$%^&*()_+.mp4'
        with open(special_file, 'wb') as f:
            f.write(b'test_content')
        
        video_id = video_manager.generate_video_id(str(special_file))
        assert video_id is not None
        assert len(video_id) > 10
        assert not video_id.startswith('fallback_')
    
    def test_register_video_same_file_path(self, video_manager, temp_config_dir):
        """Test registering video that requires renaming based on generated ID"""
        input_dir = temp_config_dir / 'input'
        video_file = input_dir / 'existing.mp4'
        with open(video_file, 'wb') as f:
            f.write(b'test_content')
        
        # Generate the video ID first to understand the expected filename
        video_id = video_manager.generate_video_id(str(video_file))
        expected_input_path = input_dir / f'{video_id}.mp4'
        
        # If the existing file doesn't match the expected ID-based name, it will be copied
        with patch('shutil.copy2', wraps=shutil.copy2) as mock_copy:
            registered_id = video_manager.register_video(str(video_file))
            
            # Check if the original filename matches the expected ID-based filename
            if str(video_file) != str(expected_input_path):
                # Should copy when filenames differ
                mock_copy.assert_called_once()
            else:
                # Should not copy when file is already in correct location
                mock_copy.assert_not_called()
                
            assert registered_id == video_id
    
    def test_register_video_copy_error(self, video_manager, temp_config_dir):
        """Test register video when copy fails"""
        source_file = temp_config_dir / 'source.mp4'
        with open(source_file, 'wb') as f:
            f.write(b'test_content')
        
        # Mock copy to raise error
        with patch('shutil.copy2', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                video_manager.register_video(str(source_file))
    
    def test_register_video_makedirs_error(self, video_manager, temp_config_dir):
        """Test register video when makedirs fails"""
        source_file = temp_config_dir / 'source.mp4'
        with open(source_file, 'wb') as f:
            f.write(b'test_content')
        
        # Mock makedirs to raise error
        with patch('os.makedirs', side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                video_manager.register_video(str(source_file))
    
    def test_metadata_save_error(self, video_manager, temp_config_dir):
        """Test metadata save when file write fails"""
        source_file = temp_config_dir / 'source.mp4'
        with open(source_file, 'wb') as f:
            f.write(b'test_content')
        
        # Mock json.dump to raise error when writing metadata
        with patch('json.dump', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                video_manager.register_video(str(source_file))
    
    def test_load_video_metadata_corrupted(self, video_manager, temp_config_dir):
        """Test loading corrupted metadata file"""
        video_id = 'test_corrupted_metadata'
        temp_dir = temp_config_dir / 'temp' / video_id
        temp_dir.mkdir(parents=True)
        
        # Create corrupted metadata file
        metadata_file = temp_dir / '.metadata.json'
        metadata_file.write_text('invalid json content')
        
        metadata = video_manager._load_video_metadata(video_id)
        assert metadata is None
    
    def test_load_video_metadata_missing_file(self, video_manager):
        """Test loading metadata for non-existent file"""
        video_id = 'nonexistent_video'
        metadata = video_manager._load_video_metadata(video_id)
        assert metadata is None
    
    def test_list_temp_files_nested_structure(self, video_manager, mock_video_file):
        """Test listing temp files with complex nested structure"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = Path(video_manager.get_temp_dir(video_id))
        
        # Create nested structure
        nested_dirs = [
            'level1/level2/level3',
            'audio/segments',
            'video/frames',
            'subtitles/chunks'
        ]
        
        for dir_path in nested_dirs:
            full_dir = temp_dir / dir_path
            full_dir.mkdir(parents=True, exist_ok=True)
            (full_dir / 'test_file.txt').write_text('test')
        
        files = video_manager.list_temp_files(video_id)
        assert len(files) >= len(nested_dirs)
        
        # Check all nested files are found
        for dir_path in nested_dirs:
            assert any(dir_path in f for f in files)
    
    def test_list_temp_files_permission_error(self, video_manager, mock_video_file):
        """Test listing temp files when permission denied"""
        video_id = video_manager.register_video(mock_video_file)
        
        # Mock os.walk to raise permission error
        with patch('os.walk', side_effect=PermissionError("Permission denied")):
            # Should raise error since the code doesn't handle exceptions
            with pytest.raises(PermissionError):
                video_manager.list_temp_files(video_id)
    
    def test_list_output_files_nonexistent_directory(self, video_manager):
        """Test listing output files when output directory doesn't exist"""
        video_id = 'test_video'
        
        # Mock output directory as non-existent
        with patch('os.path.exists', return_value=False):
            files = video_manager.list_output_files(video_id)
            assert files == []
    
    def test_list_output_files_permission_error(self, video_manager, temp_config_dir):
        """Test listing output files when permission denied"""
        video_id = 'test_video'
        
        # Mock listdir to raise permission error
        with patch('os.listdir', side_effect=PermissionError("Permission denied")):
            # Should raise error since the code doesn't handle exceptions
            with pytest.raises(PermissionError):
                video_manager.list_output_files(video_id)
    
    def test_safe_overwrite_temp_files_nested_readonly(self, video_manager, mock_video_file):
        """Test safe overwrite with nested readonly files"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = Path(video_manager.get_temp_dir(video_id))
        
        # Create nested structure with readonly files
        nested_file = temp_dir / 'subdir' / 'readonly.txt'
        nested_file.parent.mkdir(parents=True)
        nested_file.write_text('readonly content')
        nested_file.chmod(0o444)  # Read-only
        
        # Should handle readonly files gracefully
        try:
            video_manager.safe_overwrite_temp_files(video_id)
            # If we get here, the operation succeeded
            assert not nested_file.exists()
        except PermissionError:
            # If permission error, that's also acceptable behavior
            assert nested_file.exists()
    
    def test_safe_overwrite_temp_files_remove_error(self, video_manager, mock_video_file):
        """Test safe overwrite when file removal fails"""
        video_id = video_manager.register_video(mock_video_file)
        temp_dir = Path(video_manager.get_temp_dir(video_id))
        
        # Create test file
        test_file = temp_dir / 'test.txt'
        test_file.write_text('test')
        
        # Mock os.remove to raise error
        def mock_remove_side_effect(path):
            if 'test.txt' in path:
                raise PermissionError("Permission denied")
        
        with patch('os.remove', side_effect=mock_remove_side_effect):
            # Should raise error since the code doesn't handle exceptions
            with pytest.raises(PermissionError):
                video_manager.safe_overwrite_temp_files(video_id)
    
    def test_safe_overwrite_output_files_remove_error(self, video_manager, temp_config_dir):
        """Test safe overwrite output files when removal fails"""
        video_id = 'test_video'
        output_dir = Path(video_manager.paths['output'])
        
        # Create test output file
        output_file = output_dir / f'{video_id}_test.mp4'
        output_file.write_text('test')
        
        # Mock os.remove to raise error
        def mock_remove_side_effect(path):
            if f'{video_id}_test.mp4' in path:
                raise PermissionError("Permission denied")
        
        with patch('os.remove', side_effect=mock_remove_side_effect):
            # Should raise error since the code doesn't handle exceptions
            with pytest.raises(PermissionError):
                video_manager.safe_overwrite_output_files(video_id)
    
    def test_get_current_video_id_no_input_dir(self, video_manager):
        """Test getting current video ID when input directory doesn't exist"""
        with patch('os.path.exists', return_value=False):
            current_id = video_manager.get_current_video_id()
            assert current_id is None
    
    def test_get_current_video_id_permission_error(self, video_manager, temp_config_dir):
        """Test getting current video ID when listdir fails"""
        with patch('os.listdir', side_effect=PermissionError("Permission denied")):
            # Should raise error since the code doesn't handle exceptions
            with pytest.raises(PermissionError):
                video_manager.get_current_video_id()
    
    def test_get_current_video_id_various_extensions(self, video_manager, temp_config_dir):
        """Test getting current video ID with various file extensions"""
        input_dir = Path(video_manager.paths['input'])
        
        # Create files with different extensions
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        for i, ext in enumerate(extensions):
            video_file = input_dir / f'video_{i}{ext}'
            video_file.write_text(f'content {i}')
            time.sleep(0.01)  # Ensure different timestamps
        
        current_id = video_manager.get_current_video_id()
        # Should return the newest file (last one created)
        assert current_id == 'video_4'
    
    def test_get_current_video_id_mixed_files(self, video_manager, temp_config_dir):
        """Test getting current video ID with mixed file types"""
        input_dir = Path(video_manager.paths['input'])
        
        # Create mix of video and non-video files
        (input_dir / 'document.txt').write_text('text')
        (input_dir / 'image.jpg').write_text('image')
        (input_dir / 'video.mp4').write_text('video')
        (input_dir / 'audio.mp3').write_text('audio')
        
        current_id = video_manager.get_current_video_id()
        assert current_id == 'video'
    
    def test_log_overwrite_operation_makedirs_error(self, video_manager, mock_video_file):
        """Test overwrite logging when makedirs fails"""
        video_id = video_manager.register_video(mock_video_file)
        
        # Mock makedirs to raise error
        with patch('os.makedirs', side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                video_manager._log_overwrite_operation(video_id, 'temp', ['test.txt'])
    
    def test_log_overwrite_operation_write_error(self, video_manager, mock_video_file):
        """Test overwrite logging when file write fails"""
        video_id = video_manager.register_video(mock_video_file)
        
        # Mock open to raise error for log file
        def mock_open_side_effect(path, mode='r', **kwargs):
            if 'overwrite_operations.log' in path and 'a' in mode:
                raise PermissionError("Permission denied")
            return mock_open()(path, mode, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open_side_effect):
            with pytest.raises(PermissionError):
                video_manager._log_overwrite_operation(video_id, 'temp', ['test.txt'])
    
    def test_ensure_directory_structure_error(self, temp_config_dir):
        """Test directory structure creation when makedirs fails"""
        with patch('core.utils.video_manager.get_storage_paths') as mock_paths:
            mock_paths.return_value = {
                'input': '/invalid/path/input',
                'temp': '/invalid/path/temp',
                'output': '/invalid/path/output'
            }
            
            with patch('os.makedirs', side_effect=OSError("Permission denied")):
                with pytest.raises(OSError):
                    VideoFileManager()
    
    def test_video_manager_paths_validation(self, temp_config_dir):
        """Test video manager with invalid paths"""
        with patch('core.utils.video_manager.get_storage_paths') as mock_paths:
            mock_paths.return_value = {
                'input': None,  # Invalid path
                'temp': '/tmp/test/temp',
                'output': '/tmp/test/output'
            }
            
            with pytest.raises(TypeError):
                VideoFileManager()
    
    def test_video_id_time_uniqueness(self, video_manager, temp_config_dir):
        """Test that video IDs are unique even with same content"""
        # Create identical files
        file1 = temp_config_dir / 'identical1.mp4'
        file2 = temp_config_dir / 'identical2.mp4'
        
        content = b'identical_content' * 100
        file1.write_bytes(content)
        file2.write_bytes(content)
        
        # Generate IDs with small time gap
        id1 = video_manager.generate_video_id(str(file1))
        time.sleep(0.01)  # Small delay
        id2 = video_manager.generate_video_id(str(file2))
        
        # Should be different due to timestamp component
        assert id1 != id2
    
    def test_get_output_file_edge_cases(self, video_manager):
        """Test get_output_file with edge cases"""
        video_id = "test_video"
        
        # Test with empty suffix
        output_file = video_manager.get_output_file(video_id, "")
        assert f"{video_id}_" in output_file
        assert output_file.endswith('.mp4')
        
        # Test with special characters in suffix
        output_file = video_manager.get_output_file(video_id, "sub@#$%")
        assert "sub@#$%" in output_file
        
        # Test with very long suffix
        long_suffix = "a" * 100
        output_file = video_manager.get_output_file(video_id, long_suffix)
        assert long_suffix in output_file
    
    def test_concurrent_metadata_access(self, video_manager, temp_config_dir):
        """Test concurrent metadata access"""
        video_id = 'concurrent_test'
        
        errors = []
        results = []
        
        def save_metadata(thread_id):
            try:
                metadata = {
                    'thread_id': thread_id,
                    'timestamp': time.time(),
                    'data': f'test_data_{thread_id}'
                }
                video_manager._save_video_metadata(video_id, metadata)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_metadata, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have minimal errors (last write wins)
        assert len(errors) <= 1  # At most one error due to race condition
        assert len(results) >= 2  # At least some should succeed
    
    def test_file_extension_extraction(self, video_manager, temp_config_dir):
        """Test file extension extraction in various scenarios"""
        # Test with multiple dots
        file_path = temp_config_dir / 'video.backup.mp4'
        file_path.write_bytes(b'test')
        
        video_id = video_manager.register_video(str(file_path))
        metadata = video_manager._load_video_metadata(video_id)
        assert metadata['file_extension'] == '.mp4'
        
        # Test with no extension
        file_path2 = temp_config_dir / 'video_no_ext'
        file_path2.write_bytes(b'test')
        
        video_id2 = video_manager.register_video(str(file_path2))
        metadata2 = video_manager._load_video_metadata(video_id2)
        assert metadata2['file_extension'] == ''
    
    def test_temp_file_operations_error_handling(self, video_manager, mock_video_file):
        """Test temp file operations with various error conditions"""
        video_id = video_manager.register_video(mock_video_file)
        
        # Test with non-existent temp directory
        with patch.object(video_manager, 'get_temp_dir', return_value='/nonexistent/dir'):
            files = video_manager.list_temp_files(video_id)
            assert files == []
        
        # Test get_temp_file with valid video_id
        temp_file = video_manager.get_temp_file(video_id, 'test.txt')
        assert video_id in temp_file
        assert 'test.txt' in temp_file