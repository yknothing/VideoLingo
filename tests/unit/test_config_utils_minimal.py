# Minimal Working Tests for Configuration Management
# Tests core/utils/config_utils.py - Essential functions only

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from core.utils.config_utils import (
    load_key,
    save_key,
    get_storage_paths,
    ensure_storage_dirs,
    load_all_config,
)


class TestConfigUtilsMinimal:
    """Minimal test suite for configuration utilities"""

    def test_load_key_existing_config(self, temp_config_dir):
        """Test loading existing configuration key"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            result = load_key("api.key")
            assert result == "test-api-key"

    def test_load_key_nested_path(self, temp_config_dir):
        """Test loading nested configuration keys"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            result = load_key("asr.whisper.model")
            assert result == "base"

    def test_load_key_nonexistent_key(self, temp_config_dir):
        """Test loading non-existent configuration key"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            result = load_key("nonexistent.key")
            assert result is None

    def test_load_key_with_default(self, temp_config_dir):
        """Test loading key with default value"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            result = load_key("nonexistent.key", default="default_value")
            assert result == "default_value"

    def test_save_key_new_value(self, temp_config_dir):
        """Test saving new configuration value"""
        config_path = temp_config_dir / "config.yaml"
        with patch("core.utils.config_utils.CONFIG_PATH", str(config_path)):
            success = save_key("new.test.key", "test_value")
            assert success is True

            # Verify the value was saved
            result = load_key("new.test.key")
            assert result == "test_value"

    def test_load_all_config(self, temp_config_dir):
        """Test loading complete configuration"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            config = load_all_config()
            assert isinstance(config, dict)
            assert "api" in config
            assert "paths" in config
            assert config["api"]["key"] == "test-api-key"

    def test_get_storage_paths_from_config(self, temp_config_dir):
        """Test getting storage paths from configuration"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            paths = get_storage_paths()
            assert "input" in paths
            assert "temp" in paths
            assert "output" in paths
            assert paths["input"].endswith("input")

    def test_ensure_storage_dirs_creation(self, temp_config_dir):
        """Test storage directory creation"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", str(temp_config_dir / "config.yaml")
        ):
            ensure_storage_dirs()

            # Check that directories were created
            paths = get_storage_paths()
            for path in paths.values():
                assert os.path.exists(path)
                assert os.path.isdir(path)
