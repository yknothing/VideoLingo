# Comprehensive test for config_utils to maximize coverage

import pytest
import os
import tempfile
import sys
import importlib.util
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Import config_utils directly 
config_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'utils', 'config_utils.py')
spec = importlib.util.spec_from_file_location("config_utils", config_utils_path)
config_utils = importlib.util.module_from_spec(spec)
sys.modules["config_utils"] = config_utils
spec.loader.exec_module(config_utils)

class TestConfigUtilsComprehensive:
    """Comprehensive test coverage for config_utils"""
    
    def test_get_joiner_function(self):
        """Test get_joiner function with all branches"""
        yaml_content = """
language_split_with_space:
  - en
  - fr
  - de
language_split_without_space:
  - zh
  - ja
  - ko
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test language with space
            assert config_utils.get_joiner('en') == " "
            assert config_utils.get_joiner('fr') == " "
            
            # Test language without space
            assert config_utils.get_joiner('zh') == ""
            assert config_utils.get_joiner('ja') == ""
            
            # Test unsupported language
            with pytest.raises(ValueError, match="Unsupported language code: xx"):
                config_utils.get_joiner('xx')
    
    def test_get_storage_paths_with_config(self):
        """Test get_storage_paths when config exists"""
        yaml_content = """
video_storage:
  base_path: /test/base
  input_dir: input_custom
  temp_dir: temp_custom
  output_dir: output_custom
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            paths = config_utils.get_storage_paths()
            
            assert paths['base'] == '/test/base'
            assert paths['input'] == '/test/base/input_custom'
            assert paths['temp'] == '/test/base/temp_custom'
            assert paths['output'] == '/test/base/output_custom'
    
    def test_get_storage_paths_fallback(self):
        """Test get_storage_paths fallback when config missing"""
        yaml_content = """
other_config: value
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            paths = config_utils.get_storage_paths()
            
            # Should use fallback values
            assert paths['base'] == 'output'
            assert paths['input'] == 'input'
            assert paths['temp'] == 'temp'
            assert paths['output'] == 'output'
    
    def test_ensure_storage_dirs(self):
        """Test ensure_storage_dirs function"""
        mock_makedirs = MagicMock()
        
        with patch('os.makedirs', mock_makedirs), \
             patch.object(config_utils, 'get_storage_paths', return_value={
                 'base': '/test/base',
                 'input': '/test/input',
                 'temp': '/test/temp',
                 'output': '/test/output'
             }):
            
            config_utils.ensure_storage_dirs()
            
            # Should create main directories
            expected_calls = [
                (('/test/base',), {'exist_ok': True}),
                (('/test/input',), {'exist_ok': True}),
                (('/test/temp',), {'exist_ok': True}),
                (('/test/output',), {'exist_ok': True})
            ]
            
            # Should also create temp subdirectories
            temp_subdirs = ['log', 'gpt_log', 'audio', 'audio/refers', 'audio/segs', 'audio/tmp']
            for subdir in temp_subdirs:
                expected_calls.append((('/test/temp/' + subdir,), {'exist_ok': True}))
            
            # Check that makedirs was called for all expected directories
            assert mock_makedirs.call_count >= len(expected_calls)
    
    def test_is_protected_directory(self):
        """Test is_protected_directory function"""
        with patch.object(config_utils, 'get_storage_paths', return_value={
            'input': '/test/input',
            'temp': '/test/temp',
            'output': '/test/output'
        }), patch('os.path.abspath', side_effect=lambda x: x):
            
            # Test protected directories
            assert config_utils.is_protected_directory('/test/input') is True
            assert config_utils.is_protected_directory('/test/temp') is True
            assert config_utils.is_protected_directory('/test/output') is True
            
            # Test non-protected directory
            assert config_utils.is_protected_directory('/other/path') is False
    
    def test_clean_directory_contents_files_only(self):
        """Test clean_directory_contents with files only"""
        mock_listdir = MagicMock(return_value=['file1.txt', 'file2.txt'])
        mock_isfile = MagicMock(side_effect=lambda x: True)
        mock_isdir = MagicMock(side_effect=lambda x: False)
        mock_remove = MagicMock()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', mock_listdir), \
             patch('os.path.isfile', mock_isfile), \
             patch('os.path.isdir', mock_isdir), \
             patch('os.remove', mock_remove):
            
            config_utils.clean_directory_contents('/test/dir')
            
            # Should remove both files
            assert mock_remove.call_count == 2
    
    def test_clean_directory_contents_with_subdirs(self):
        """Test clean_directory_contents with subdirectories"""
        mock_listdir = MagicMock(return_value=['file1.txt', 'subdir'])
        mock_isfile = MagicMock(side_effect=lambda x: 'file' in x)
        mock_isdir = MagicMock(side_effect=lambda x: 'subdir' in x)
        mock_remove = MagicMock()
        mock_rmtree = MagicMock()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', mock_listdir), \
             patch('os.path.isfile', mock_isfile), \
             patch('os.path.isdir', mock_isdir), \
             patch('os.remove', mock_remove), \
             patch('shutil.rmtree', mock_rmtree), \
             patch.object(config_utils, 'is_protected_directory', return_value=False):
            
            config_utils.clean_directory_contents('/test/dir', preserve_structure=False)
            
            # Should remove file and directory
            assert mock_remove.call_count == 1
            assert mock_rmtree.call_count == 1
    
    def test_clean_directory_contents_preserve_structure(self):
        """Test clean_directory_contents with preserve_structure=True"""
        mock_listdir = MagicMock(return_value=['subdir'])
        mock_isfile = MagicMock(side_effect=lambda x: False)
        mock_isdir = MagicMock(side_effect=lambda x: True)
        mock_rmtree = MagicMock()
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', mock_listdir), \
             patch('os.path.isfile', mock_isfile), \
             patch('os.path.isdir', mock_isdir), \
             patch('shutil.rmtree', mock_rmtree), \
             patch.object(config_utils, 'is_protected_directory', return_value=True), \
             patch.object(config_utils, 'clean_directory_contents') as mock_recursive:
            
            config_utils.clean_directory_contents('/test/dir', preserve_structure=True)
            
            # Should not remove directory but recursively clean it
            assert mock_rmtree.call_count == 0
            assert mock_recursive.call_count == 2  # Original call + recursive call
    
    def test_clean_directory_nonexistent(self):
        """Test clean_directory_contents with non-existent directory"""
        with patch('os.path.exists', return_value=False):
            # Should not raise error for non-existent directory
            config_utils.clean_directory_contents('/nonexistent/dir')
    
    def test_safe_clean_processing_directories(self):
        """Test deprecated safe_clean_processing_directories function"""
        mock_paths = {
            'input': '/test/input',
            'temp': '/test/temp',
            'output': '/test/output'
        }
        
        mock_listdir = MagicMock(return_value=['file1.txt', 'subdir'])
        mock_isfile = MagicMock(side_effect=lambda x: 'file' in x)
        mock_isdir = MagicMock(side_effect=lambda x: 'subdir' in x)
        mock_remove = MagicMock()
        mock_rmtree = MagicMock()
        
        with patch.object(config_utils, 'get_storage_paths', return_value=mock_paths), \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', mock_listdir), \
             patch('os.path.isfile', mock_isfile), \
             patch('os.path.isdir', mock_isdir), \
             patch('os.remove', mock_remove), \
             patch('shutil.rmtree', mock_rmtree), \
             patch('builtins.print') as mock_print:
            
            config_utils.safe_clean_processing_directories(exclude_input=True)
            
            # Should print deprecation warning
            mock_print.assert_called()
            warning_call = mock_print.call_args_list[0][0][0]
            assert "deprecated" in warning_call
            
            # Should clean temp and output but not input
            # Called for both temp and output directories
            assert mock_remove.call_count >= 2  # At least one file per directory
            assert mock_rmtree.call_count >= 2  # At least one subdir per directory
    
    def test_safe_clean_processing_directories_with_exceptions(self):
        """Test safe_clean_processing_directories with file removal exceptions"""
        mock_paths = {
            'input': '/test/input',
            'temp': '/test/temp',
            'output': '/test/output'
        }
        
        mock_listdir = MagicMock(return_value=['file1.txt'])
        mock_isfile = MagicMock(return_value=True)
        mock_remove = MagicMock(side_effect=PermissionError("Permission denied"))
        
        with patch.object(config_utils, 'get_storage_paths', return_value=mock_paths), \
             patch('os.path.exists', return_value=True), \
             patch('os.listdir', mock_listdir), \
             patch('os.path.isfile', mock_isfile), \
             patch('os.remove', mock_remove), \
             patch('builtins.print') as mock_print:
            
            # Should not raise exception even if file removal fails
            config_utils.safe_clean_processing_directories()
            
            # Should print warning about failed removal
            warning_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("Could not remove" in call for call in warning_calls)
    
    def test_get_video_specific_paths(self):
        """Test get_video_specific_paths function"""
        mock_manager = MagicMock()
        mock_manager.get_video_paths.return_value = {
            'input': '/test/video123.mp4',
            'temp_dir': '/test/temp/video123',
            'output_dir': '/test/output'
        }
        
        with patch('core.utils.video_manager.get_video_manager', return_value=mock_manager):
            paths = config_utils.get_video_specific_paths('video123')
            
            assert paths['input'] == '/test/video123.mp4'
            assert paths['temp_dir'] == '/test/temp/video123'
            mock_manager.get_video_paths.assert_called_once_with('video123')
    
    def test_get_or_create_video_id_existing(self):
        """Test get_or_create_video_id with existing video"""
        mock_manager = MagicMock()
        mock_manager.get_current_video_id.return_value = 'existing_video_123'
        
        with patch('core.utils.video_manager.get_video_manager', return_value=mock_manager):
            video_id = config_utils.get_or_create_video_id()
            
            assert video_id == 'existing_video_123'
            mock_manager.get_current_video_id.assert_called_once()
    
    def test_get_or_create_video_id_new_video(self):
        """Test get_or_create_video_id with new video"""
        mock_manager = MagicMock()
        mock_manager.get_current_video_id.return_value = None
        mock_manager.register_video.return_value = 'new_video_456'
        
        with patch('core.utils.video_manager.get_video_manager', return_value=mock_manager), \
             patch('os.path.exists', return_value=True):
            
            video_id = config_utils.get_or_create_video_id('/test/video.mp4')
            
            assert video_id == 'new_video_456'
            mock_manager.register_video.assert_called_once_with('/test/video.mp4')
    
    def test_get_or_create_video_id_no_video(self):
        """Test get_or_create_video_id with no video"""
        mock_manager = MagicMock()
        mock_manager.get_current_video_id.return_value = None
        
        with patch('core.utils.video_manager.get_video_manager', return_value=mock_manager):
            with pytest.raises(ValueError, match="No current video found"):
                config_utils.get_or_create_video_id()
    
    def test_get_or_create_video_id_nonexistent_path(self):
        """Test get_or_create_video_id with non-existent video path"""
        mock_manager = MagicMock()
        mock_manager.get_current_video_id.return_value = None
        
        with patch('core.utils.video_manager.get_video_manager', return_value=mock_manager), \
             patch('os.path.exists', return_value=False):
            
            with pytest.raises(ValueError, match="No current video found"):
                config_utils.get_or_create_video_id('/nonexistent/video.mp4')
    
    def test_main_execution(self):
        """Test __main__ execution branch"""
        yaml_content = """
language_split_with_space:
  - en
  - fr
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test the main block behavior by calling load_key directly
            # This simulates what happens in the __main__ block
            result = config_utils.load_key('language_split_with_space')
            assert result == ['en', 'fr']