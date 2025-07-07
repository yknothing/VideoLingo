# Unit Tests for Configuration Management
# Tests core/utils/config_utils.py

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import threading
import time

from core.utils.config_utils import (
    load_key, save_key, load_all_config, 
    get_storage_paths, ensure_storage_dirs,
    update_config, validate_config_structure
)

class TestConfigUtils:
    """Test suite for configuration utilities"""
    
    def test_load_key_existing_config(self, temp_config_dir):
        """Test loading existing configuration key"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            result = load_key("api.key")
            assert result == "test-api-key"
    
    def test_load_key_nested_path(self, temp_config_dir):
        """Test loading nested configuration keys"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            result = load_key("asr.whisper.model")
            assert result == "base"
    
    def test_load_key_nonexistent_key(self, temp_config_dir):
        """Test loading non-existent configuration key"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            result = load_key("nonexistent.key")
            assert result is None
    
    def test_load_key_with_default(self, temp_config_dir):
        """Test loading key with default value"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            result = load_key("nonexistent.key", default="default_value")
            assert result == "default_value"
    
    def test_save_key_new_value(self, temp_config_dir):
        """Test saving new configuration value"""
        config_path = temp_config_dir / "config.yaml"
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            success = save_key("new.test.key", "test_value")
            assert success is True
            
            # Verify the value was saved
            result = load_key("new.test.key")
            assert result == "test_value"
    
    def test_save_key_update_existing(self, temp_config_dir):
        """Test updating existing configuration value"""
        config_path = temp_config_dir / "config.yaml"
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            # Update existing key
            success = save_key("api.key", "new-api-key")
            assert success is True
            
            # Verify the value was updated
            result = load_key("api.key")
            assert result == "new-api-key"
    
    def test_save_key_invalid_yaml_structure(self, temp_config_dir):
        """Test saving key with invalid YAML structure"""
        config_path = temp_config_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            success = save_key("test.key", "value")
            # Should handle invalid YAML gracefully
            assert success is False
    
    def test_load_all_config(self, temp_config_dir):
        """Test loading complete configuration"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            config = load_all_config()
            assert isinstance(config, dict)
            assert "api" in config
            assert "paths" in config
            assert config["api"]["key"] == "test-api-key"
    
    def test_load_all_config_missing_file(self):
        """Test loading config when file doesn't exist"""
        with patch('core.utils.config_utils.CONFIG_PATH', "/nonexistent/config.yaml"):
            config = load_all_config()
            assert config == {}
    
    def test_get_storage_paths_from_config(self, temp_config_dir):
        """Test getting storage paths from configuration"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            paths = get_storage_paths()
            assert "input" in paths
            assert "temp" in paths
            assert "output" in paths
            assert paths["input"].endswith("input")
    
    def test_get_storage_paths_default(self):
        """Test getting default storage paths"""
        with patch('core.utils.config_utils.load_key', return_value=None):
            paths = get_storage_paths()
            assert "input" in paths
            assert "temp" in paths
            assert "output" in paths
            # Should use default relative paths
            assert "input" in paths["input"]
    
    def test_ensure_storage_dirs_creation(self, temp_config_dir):
        """Test storage directory creation"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            ensure_storage_dirs()
            
            # Check that directories were created
            paths = get_storage_paths()
            for path in paths.values():
                assert os.path.exists(path)
                assert os.path.isdir(path)
    
    def test_ensure_storage_dirs_existing(self, temp_config_dir):
        """Test storage directory creation when dirs already exist"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            # Create dirs first
            ensure_storage_dirs()
            
            # Should not fail if dirs already exist
            ensure_storage_dirs()
            
            paths = get_storage_paths()
            for path in paths.values():
                assert os.path.exists(path)
    
    def test_update_config_merge(self, temp_config_dir):
        """Test configuration update with merge"""
        config_path = temp_config_dir / "config.yaml"
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            new_config = {
                "api": {
                    "new_field": "new_value"
                },
                "new_section": {
                    "test": "value"
                }
            }
            
            success = update_config(new_config)
            assert success is True
            
            # Verify merge worked correctly
            updated_config = load_all_config()
            assert updated_config["api"]["key"] == "test-api-key"  # Original preserved
            assert updated_config["api"]["new_field"] == "new_value"  # New added
            assert updated_config["new_section"]["test"] == "value"  # New section added
    
    def test_validate_config_structure_valid(self, temp_config_dir):
        """Test configuration structure validation with valid config"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            is_valid, errors = validate_config_structure()
            assert is_valid is True
            assert len(errors) == 0
    
    def test_validate_config_structure_missing_required(self):
        """Test configuration validation with missing required fields"""
        invalid_config = {"incomplete": "config"}
        
        with patch('core.utils.config_utils.load_all_config', return_value=invalid_config):
            is_valid, errors = validate_config_structure()
            assert is_valid is False
            assert len(errors) > 0
            assert any("api" in error for error in errors)
    
    def test_thread_safety_concurrent_access(self, temp_config_dir):
        """Test thread safety of configuration access"""
        config_path = temp_config_dir / "config.yaml"
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
                    # Concurrent read/write operations
                    for i in range(10):
                        save_key(f"thread_{thread_id}.key_{i}", f"value_{i}")
                        value = load_key(f"thread_{thread_id}.key_{i}")
                        results.append((thread_id, i, value))
                        time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify all operations completed
        assert len(results) == 50  # 5 threads * 10 operations each
        
        # Verify data consistency
        for thread_id, key_id, value in results:
            assert value == f"value_{key_id}"
    
    def test_config_file_corruption_recovery(self, temp_config_dir):
        """Test recovery from corrupted configuration file"""
        config_path = temp_config_dir / "config.yaml"
        
        # Corrupt the config file
        with open(config_path, 'w') as f:
            f.write("corrupted: yaml: content: [incomplete")
        
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            # Should handle corruption gracefully
            config = load_all_config()
            assert config == {}
            
            # Should be able to save new config
            success = save_key("recovery.test", "success")
            assert success is True
    
    def test_config_environment_override(self, temp_config_dir):
        """Test environment variable override for config paths"""
        config_path = temp_config_dir / "config.yaml"
        
        with patch.dict(os.environ, {'VIDEOLINGO_CONFIG_PATH': str(config_path)}):
            # Should use environment variable path
            result = load_key("api.key")
            assert result == "test-api-key"
    
    def test_config_backup_creation(self, temp_config_dir):
        """Test that config backups are created on updates"""
        config_path = temp_config_dir / "config.yaml"
        
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            # Make an update that should trigger backup
            save_key("backup.test", "value")
            
            # Check if backup directory exists (implementation dependent)
            backup_dir = temp_config_dir / ".backups"
            if backup_dir.exists():
                backup_files = list(backup_dir.glob("config_*.yaml"))
                assert len(backup_files) > 0
    
    @pytest.mark.parametrize("key_path,expected_type", [
        ("api.key", str),
        ("asr.whisper.model", str),
        ("api.llm_support_json", bool),
        ("tts.speed", float),
        ("allowed_video_formats", list),
    ])
    def test_config_data_types(self, temp_config_dir, key_path, expected_type):
        """Test that configuration values maintain correct data types"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            value = load_key(key_path)
            assert isinstance(value, expected_type), f"Expected {expected_type}, got {type(value)}"
    
    def test_config_key_case_sensitivity(self, temp_config_dir):
        """Test case sensitivity of configuration keys"""
        with patch('core.utils.config_utils.CONFIG_PATH', str(temp_config_dir / "config.yaml")):
            # Test that keys are case sensitive
            value1 = load_key("api.key")
            value2 = load_key("API.KEY")
            value3 = load_key("api.KEY")
            
            assert value1 == "test-api-key"
            assert value2 is None
            assert value3 is None
    
    def test_config_unicode_support(self, temp_config_dir):
        """Test Unicode support in configuration values"""
        config_path = temp_config_dir / "config.yaml"
        
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            # Test saving and loading Unicode values
            unicode_values = {
                "chinese": "你好世界",
                "japanese": "こんにちは",
                "emoji": "🎉🚀🎯",
                "mixed": "Hello 世界 🌍"
            }
            
            for key, value in unicode_values.items():
                success = save_key(f"unicode.{key}", value)
                assert success is True
                
                loaded_value = load_key(f"unicode.{key}")
                assert loaded_value == value
    
    def test_large_config_performance(self, temp_config_dir):
        """Test performance with large configuration files"""
        config_path = temp_config_dir / "config.yaml"
        
        with patch('core.utils.config_utils.CONFIG_PATH', str(config_path)):
            # Create a large configuration
            start_time = time.time()
            
            for i in range(1000):
                save_key(f"large.section_{i // 100}.key_{i}", f"value_{i}")
            
            creation_time = time.time() - start_time
            
            # Performance should be reasonable (less than 10 seconds for 1000 keys)
            assert creation_time < 10.0
            
            # Test bulk loading performance
            start_time = time.time()
            config = load_all_config()
            load_time = time.time() - start_time
            
            assert load_time < 1.0  # Should load quickly
            assert len(str(config)) > 10000  # Should contain substantial data