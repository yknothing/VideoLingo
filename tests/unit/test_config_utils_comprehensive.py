# Comprehensive Test Suite for Configuration Management
# Tests core/utils/config_utils.py - Targeting 85% branch coverage

import pytest
import os
import tempfile
import threading
import time
import json
from pathlib import Path
from unittest.mock import patch, mock_open, Mock, call
from ruamel.yaml import YAML

try:
    from core.utils.config_utils import (
        load_key,
        update_key,
        load_all_config,
        update_config,
        validate_config_structure,
        get_storage_paths,
        ensure_storage_dirs,
        get_joiner,
        is_protected_directory,
        clean_directory_contents,
        safe_clean_processing_directories,
        get_video_specific_paths,
        get_or_create_video_id,
        save_key,
        load_secret_key,
        load_key_with_path,
        _resolve_keys_path,
        _read_kv_file,
        _write_kv_file,
        ENV_MAPPINGS,
        CONFIG_PATH,
        lock,
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestEnvironmentMapping:
    """Test environment variable mapping and priority"""

    def test_load_key_environment_priority(self):
        """Test environment variable takes priority over config file"""
        config_content = """
api:
  key: "config-key"
  base_url: "https://config.example.com"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path), patch.dict(
                os.environ, {"API_KEY": "env-key"}
            ):
                # Should return environment variable value
                result = load_key("api.key")
                assert result == "env-key"

        finally:
            os.unlink(config_path)

    def test_load_key_config_fallback(self):
        """Test fallback to config when env var not set"""
        config_content = """
api:
  key: "config-key"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path), patch.dict(
                os.environ, {}, clear=True
            ):
                result = load_key("api.key")
                assert result == "config-key"

        finally:
            os.unlink(config_path)

    def test_load_key_empty_config_value_env_fallback(self):
        """Test empty config value falls back to environment"""
        config_content = """
api:
  key: ""
  base_url: "https://config.example.com"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path), patch.dict(
                os.environ, {"API_KEY": "env-fallback-key"}
            ):
                result = load_key("api.key")
                assert result == "env-fallback-key"

        finally:
            os.unlink(config_path)

    def test_load_key_no_env_mapping(self):
        """Test keys without environment mapping"""
        config_content = """
custom:
  setting: "custom-value"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                result = load_key("custom.setting")
                assert result == "custom-value"

        finally:
            os.unlink(config_path)


class TestYAMLOperations:
    """Test YAML loading, parsing, and thread safety"""

    def test_load_key_nested_path(self):
        """Test loading deeply nested configuration keys"""
        config_content = """
level1:
  level2:
    level3:
      deep_value: "found"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                result = load_key("level1.level2.level3.deep_value")
                assert result == "found"

        finally:
            os.unlink(config_path)

    def test_load_key_path_not_exists(self):
        """Test loading non-existent nested paths"""
        config_content = """
existing:
  key: "value"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                # Non-existent top level
                assert load_key("nonexistent.key") is None

                # Non-existent nested level
                assert load_key("existing.nonexistent") is None

                # Path through non-dict value
                assert load_key("existing.key.invalid") is None

        finally:
            os.unlink(config_path)

    def test_load_key_default_value(self):
        """Test default value handling"""
        config_content = """
existing:
  key: "value"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                result = load_key("nonexistent.key", default="default_value")
                assert result == "default_value"

                result = load_key("existing.key", default="default_value")
                assert result == "value"

        finally:
            os.unlink(config_path)

    def test_update_key_success(self):
        """Test successful key update"""
        config_content = """
api:
  key: "old-key"
  model: "gpt-3.5"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                success = update_key("api.key", "new-key")
                assert success is True

                # Verify update
                result = load_key("api.key")
                assert result == "new-key"

                # Verify other values unchanged
                result = load_key("api.model")
                assert result == "gpt-3.5"

        finally:
            os.unlink(config_path)

    def test_update_key_nested_success(self):
        """Test nested key update"""
        config_content = """
level1:
  level2:
    key: "old-value"
    other: "unchanged"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                success = update_key("level1.level2.key", "new-value")
                assert success is True

                result = load_key("level1.level2.key")
                assert result == "new-value"

                result = load_key("level1.level2.other")
                assert result == "unchanged"

        finally:
            os.unlink(config_path)

    def test_update_key_path_not_exists(self):
        """Test update with non-existent path"""
        config_content = """
existing:
  key: "value"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                success = update_key("nonexistent.key", "new-value")
                assert success is False

        finally:
            os.unlink(config_path)

    def test_update_key_invalid_path(self):
        """Test update with invalid path structure"""
        config_content = """
existing:
  key: "value"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                with pytest.raises(KeyError, match="Key 'nonexistent' not found"):
                    update_key("existing.nonexistent", "new-value")

        finally:
            os.unlink(config_path)

    def test_load_all_config_success(self):
        """Test loading complete configuration"""
        config_content = """
api:
  key: "test-key"
  model: "gpt-4"
paths:
  input: "input"
  output: "output"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                config = load_all_config()

                assert isinstance(config, dict)
                assert config["api"]["key"] == "test-key"
                assert config["api"]["model"] == "gpt-4"
                assert config["paths"]["input"] == "input"

        finally:
            os.unlink(config_path)

    def test_load_all_config_file_not_found(self):
        """Test loading config when file doesn't exist"""
        with patch("core.utils.config_utils.CONFIG_PATH", "/nonexistent/config.yaml"):
            config = load_all_config()
            assert config == {}

    def test_load_all_config_invalid_yaml(self):
        """Test loading config with invalid YAML"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                config = load_all_config()
                assert config == {}

        finally:
            os.unlink(config_path)


class TestConfigurationMerging:
    """Test configuration merging and updating"""

    def test_update_config_new_file(self):
        """Test updating config when file doesn't exist"""
        new_config = {"api": {"key": "new-key"}, "paths": {"input": "input"}}

        with patch(
            "core.utils.config_utils.CONFIG_PATH", "/tmp/new_config.yaml"
        ), patch("builtins.open", mock_open()) as mock_file, patch(
            "core.utils.config_utils.yaml.dump"
        ) as mock_dump:
            success = update_config(new_config)
            assert success is True

            # Verify file operations
            mock_file.assert_called()
            mock_dump.assert_called_once()

    def test_update_config_merge_existing(self):
        """Test merging new config with existing"""
        existing_config = """
api:
  key: "old-key"
  model: "gpt-3.5"
paths:
  input: "old-input"
"""

        new_config = {
            "api": {"key": "new-key", "base_url": "https://example.com"},
            "new_section": {"test": "value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(existing_config)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                success = update_config(new_config)
                assert success is True

                # Verify merge results
                merged = load_all_config()
                assert merged["api"]["key"] == "new-key"  # Updated
                assert merged["api"]["model"] == "gpt-3.5"  # Preserved
                assert merged["api"]["base_url"] == "https://example.com"  # Added
                assert merged["paths"]["input"] == "old-input"  # Preserved
                assert merged["new_section"]["test"] == "value"  # Added

        finally:
            os.unlink(config_path)

    def test_update_config_deep_merge(self):
        """Test deep merging of nested structures"""
        existing_config = """
level1:
  level2a:
    key1: "value1"
    key2: "value2"
  level2b:
    key3: "value3"
"""

        new_config = {
            "level1": {
                "level2a": {"key1": "new_value1", "key4": "value4"},
                "level2c": {"key5": "value5"},
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(existing_config)
            config_path = f.name

        try:
            with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                success = update_config(new_config)
                assert success is True

                merged = load_all_config()
                assert merged["level1"]["level2a"]["key1"] == "new_value1"  # Updated
                assert merged["level1"]["level2a"]["key2"] == "value2"  # Preserved
                assert merged["level1"]["level2a"]["key4"] == "value4"  # Added
                assert merged["level1"]["level2b"]["key3"] == "value3"  # Preserved
                assert merged["level1"]["level2c"]["key5"] == "value5"  # Added

        finally:
            os.unlink(config_path)

    def test_update_config_exception_handling(self):
        """Test exception handling during config update"""
        with patch(
            "core.utils.config_utils.CONFIG_PATH", "/readonly/config.yaml"
        ), patch("builtins.open", side_effect=PermissionError("Access denied")):
            success = update_config({"test": "value"})
            assert success is False


class TestConfigurationValidation:
    """Test configuration structure validation"""

    def test_validate_config_structure_valid(self):
        """Test validation with valid configuration"""
        valid_config = {"api": {"key": "test-key"}, "paths": {}}

        with patch(
            "core.utils.config_utils.load_all_config", return_value=valid_config
        ):
            is_valid, errors = validate_config_structure()
            assert is_valid is True
            assert len(errors) == 0

    def test_validate_config_structure_missing_sections(self):
        """Test validation with missing required sections"""
        invalid_config = {"incomplete": "config"}

        with patch(
            "core.utils.config_utils.load_all_config", return_value=invalid_config
        ):
            is_valid, errors = validate_config_structure()
            assert is_valid is False
            assert len(errors) > 0
            assert any("Missing required section: api" in error for error in errors)
            assert any("Missing required section: paths" in error for error in errors)

    def test_validate_config_structure_invalid_section_type(self):
        """Test validation with invalid section types"""
        invalid_config = {"api": "not_a_dict", "paths": []}

        with patch(
            "core.utils.config_utils.load_all_config", return_value=invalid_config
        ):
            is_valid, errors = validate_config_structure()
            assert is_valid is False
            assert any(
                "Section 'api' must be a dictionary" in error for error in errors
            )
            assert any(
                "Section 'paths' must be a dictionary" in error for error in errors
            )

    def test_validate_config_structure_missing_required_keys(self):
        """Test validation with missing required keys"""
        invalid_config = {"api": {}, "paths": {}}  # Missing required "key"

        with patch(
            "core.utils.config_utils.load_all_config", return_value=invalid_config
        ):
            is_valid, errors = validate_config_structure()
            assert is_valid is False
            assert any("Missing required key: api.key" in error for error in errors)


class TestStoragePathManagement:
    """Test storage path configuration and management"""

    def test_get_storage_paths_from_config(self):
        """Test getting storage paths from configuration"""
        config_data = {
            "video_storage": {
                "base_path": "/custom/base",
                "input_dir": "custom_input",
                "temp_dir": "custom_temp",
                "output_dir": "custom_output",
            }
        }

        with patch("core.utils.config_utils.load_key") as mock_load_key:
            mock_load_key.side_effect = lambda key: config_data.get(
                "video_storage", {}
            ).get(key.split(".")[-1])

            with patch("os.makedirs"), patch("builtins.open", mock_open()), patch(
                "os.remove"
            ):
                paths = get_storage_paths()

                assert paths["base"] == "/custom/base"
                assert paths["input"] == "/custom/base/custom_input"
                assert paths["temp"] == "/custom/base/custom_temp"
                assert paths["output"] == "/custom/base/custom_output"

    def test_get_storage_paths_empty_base_path(self):
        """Test storage paths with empty base path"""
        with patch("core.utils.config_utils.load_key") as mock_load_key:
            mock_load_key.side_effect = lambda key: {
                "video_storage.base_path": "",
                "video_storage.input_dir": "input",
                "video_storage.temp_dir": "temp",
                "video_storage.output_dir": "output",
            }.get(key)

            paths = get_storage_paths()

            assert paths["base"] == "output"
            assert paths["input"] == "output/input"

    def test_get_storage_paths_inaccessible_base_path(self):
        """Test storage paths with inaccessible base path"""
        with patch("core.utils.config_utils.load_key") as mock_load_key, patch(
            "os.makedirs", side_effect=PermissionError("Access denied")
        ), patch("print") as mock_print:
            mock_load_key.side_effect = lambda key: {
                "video_storage.base_path": "/inaccessible/path",
                "video_storage.input_dir": "input",
                "video_storage.temp_dir": "temp",
                "video_storage.output_dir": "output",
            }.get(key)

            paths = get_storage_paths()

            assert paths["base"] == "output"
            mock_print.assert_called()

    def test_get_storage_paths_key_error_fallback(self):
        """Test storage paths fallback on KeyError"""
        with patch(
            "core.utils.config_utils.load_key", side_effect=KeyError("Config not found")
        ):
            paths = get_storage_paths()

            assert paths["base"] == "output"
            assert paths["input"] == "input"
            assert paths["temp"] == "temp"
            assert paths["output"] == "output"

    def test_ensure_storage_dirs_creation(self):
        """Test storage directory creation"""
        mock_paths = {
            "base": "/test/base",
            "input": "/test/base/input",
            "temp": "/test/base/temp",
            "output": "/test/base/output",
        }

        with patch(
            "core.utils.config_utils.get_storage_paths", return_value=mock_paths
        ), patch("os.makedirs") as mock_makedirs:
            ensure_storage_dirs()

            # Verify main directories created
            expected_calls = [
                call("/test/base", exist_ok=True),
                call("/test/base/input", exist_ok=True),
                call("/test/base/temp", exist_ok=True),
                call("/test/base/output", exist_ok=True),
            ]

            # Verify subdirectories created
            expected_subdirs = [
                "/test/base/temp/log",
                "/test/base/temp/gpt_log",
                "/test/base/temp/audio",
                "/test/base/temp/audio/refers",
                "/test/base/temp/audio/segs",
                "/test/base/temp/audio/tmp",
            ]

            for subdir in expected_subdirs:
                expected_calls.append(call(subdir, exist_ok=True))

            mock_makedirs.assert_has_calls(expected_calls, any_order=True)


class TestDirectorySafety:
    """Test directory protection and safe cleanup"""

    def test_is_protected_directory_system_paths(self):
        """Test system directory protection"""
        system_paths = ["/", "/home", "/Users", "/System", "/usr", "/var", "/etc"]

        for path in system_paths:
            assert is_protected_directory(path) is True

    def test_is_protected_directory_configured_paths(self):
        """Test configured storage path protection"""
        mock_paths = {
            "base": "/project/base",
            "input": "/project/input",
            "temp": "/project/temp",
            "output": "/project/output",
        }

        with patch(
            "core.utils.config_utils.get_storage_paths", return_value=mock_paths
        ):
            for path in mock_paths.values():
                assert is_protected_directory(path) is True

    def test_is_protected_directory_parent_protection(self):
        """Test parent directory protection"""
        mock_paths = {
            "base": "/project/videolingo/base",
            "temp": "/project/videolingo/temp",
        }

        with patch(
            "core.utils.config_utils.get_storage_paths", return_value=mock_paths
        ):
            # Parent directories should be protected
            assert is_protected_directory("/project") is True
            assert is_protected_directory("/project/videolingo") is True

    def test_is_protected_directory_shallow_paths(self):
        """Test shallow directory depth protection"""
        shallow_paths = ["/usr", "/home/user", "C:\\"]

        for path in shallow_paths:
            with patch("core.utils.config_utils.get_storage_paths", return_value={}):
                assert is_protected_directory(path) is True

    def test_is_protected_directory_safe_paths(self):
        """Test non-protected paths"""
        with patch("core.utils.config_utils.get_storage_paths", return_value={}):
            safe_paths = [
                "/home/user/project/deep/directory",
                "/tmp/some/temp/directory",
                "/var/log/application/deep/logs",
            ]

            for path in safe_paths:
                assert is_protected_directory(path) is False

    def test_clean_directory_contents_validation(self):
        """Test directory cleaning input validation"""
        # Empty path
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            clean_directory_contents("")

        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            clean_directory_contents(None)

    def test_clean_directory_contents_protected_directory(self):
        """Test cleaning protected directory is prevented"""
        with pytest.raises(ValueError, match="Cannot clean protected directory"):
            clean_directory_contents("/")

    def test_clean_directory_contents_outside_project(self):
        """Test cleaning directory outside project is prevented"""
        with patch("os.path.abspath") as mock_abspath:
            mock_abspath.side_effect = ["/outside/project", "/current/project"]

            with pytest.raises(
                ValueError, match="Cannot clean directory outside project root"
            ):
                clean_directory_contents("/outside/project")

    def test_clean_directory_contents_success(self):
        """Test successful directory cleaning"""
        with patch("os.path.abspath") as mock_abspath, patch(
            "core.utils.config_utils.is_protected_directory", return_value=False
        ), patch("os.path.exists", return_value=True), patch(
            "os.listdir", return_value=["file1.txt", "subdir"]
        ), patch(
            "os.path.isfile"
        ) as mock_isfile, patch(
            "os.path.isdir"
        ) as mock_isdir, patch(
            "os.remove"
        ) as mock_remove, patch(
            "shutil.rmtree"
        ) as mock_rmtree:
            mock_abspath.side_effect = ["/project/temp", "/project"]
            mock_isfile.side_effect = lambda x: "file" in x
            mock_isdir.side_effect = lambda x: "dir" in x

            clean_directory_contents("/project/temp")

            mock_remove.assert_called_once()
            mock_rmtree.assert_called_once()

    def test_clean_directory_contents_preserve_structure(self):
        """Test cleaning with structure preservation"""
        with patch("os.path.abspath") as mock_abspath, patch(
            "core.utils.config_utils.is_protected_directory"
        ) as mock_protected, patch("os.path.exists", return_value=True), patch(
            "os.listdir", return_value=["file1.txt", "protected_subdir"]
        ), patch(
            "os.path.isfile"
        ) as mock_isfile, patch(
            "os.path.isdir"
        ) as mock_isdir, patch(
            "os.remove"
        ) as mock_remove, patch(
            "core.utils.config_utils.clean_directory_contents"
        ) as mock_recursive:
            mock_abspath.side_effect = ["/project/temp", "/project"]
            mock_isfile.side_effect = lambda x: "file" in x
            mock_isdir.side_effect = lambda x: "dir" in x
            mock_protected.side_effect = lambda x: "protected" in x

            clean_directory_contents("/project/temp", preserve_structure=True)

            mock_remove.assert_called_once()
            mock_recursive.assert_called_once()

    def test_clean_directory_contents_exception_handling(self):
        """Test exception handling during cleanup"""
        with patch("os.path.abspath") as mock_abspath, patch(
            "core.utils.config_utils.is_protected_directory", return_value=False
        ), patch("os.path.exists", return_value=True), patch(
            "os.listdir", return_value=["problematic_file"]
        ), patch(
            "os.path.isfile", return_value=True
        ), patch(
            "os.remove", side_effect=PermissionError("Access denied")
        ), patch(
            "print"
        ) as mock_print:
            mock_abspath.side_effect = ["/project/temp", "/project"]

            # Should not raise exception, just print warning
            clean_directory_contents("/project/temp")
            mock_print.assert_called()


class TestUtilityFunctions:
    """Test utility functions"""

    def test_get_joiner_with_space(self):
        """Test joiner for space-separated languages"""
        config = {"language_split_with_space": ["en", "es", "fr", "de"]}

        with patch("core.utils.config_utils.load_key") as mock_load_key:
            mock_load_key.return_value = config["language_split_with_space"]

            assert get_joiner("en") == " "
            assert get_joiner("es") == " "

    def test_get_joiner_without_space(self):
        """Test joiner for non-space-separated languages"""
        config = {
            "language_split_with_space": ["en", "es"],
            "language_split_without_space": ["zh", "ja", "ko"],
        }

        with patch("core.utils.config_utils.load_key") as mock_load_key:

            def side_effect(key):
                if "with_space" in key:
                    return config["language_split_with_space"]
                elif "without_space" in key:
                    return config["language_split_without_space"]
                return None

            mock_load_key.side_effect = side_effect

            assert get_joiner("zh") == ""
            assert get_joiner("ja") == ""

    def test_get_joiner_unsupported_language(self):
        """Test joiner for unsupported language"""
        config = {
            "language_split_with_space": ["en"],
            "language_split_without_space": ["zh"],
        }

        with patch("core.utils.config_utils.load_key") as mock_load_key:

            def side_effect(key):
                if "with_space" in key:
                    return config["language_split_with_space"]
                elif "without_space" in key:
                    return config["language_split_without_space"]
                return None

            mock_load_key.side_effect = side_effect

            with pytest.raises(ValueError, match="Unsupported language code: unknown"):
                get_joiner("unknown")


class TestVideoManagement:
    """Test video-specific path and ID management"""

    def test_get_video_specific_paths(self):
        """Test getting video-specific paths"""
        with patch("core.utils.config_utils.get_video_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_video_paths.return_value = {"input": "/path/to/video"}
            mock_get_manager.return_value = mock_manager

            result = get_video_specific_paths("test_video_id")

            assert result == {"input": "/path/to/video"}
            mock_manager.get_video_paths.assert_called_once_with("test_video_id")

    def test_get_or_create_video_id_existing(self):
        """Test getting existing video ID"""
        with patch("core.utils.config_utils.get_video_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_current_video_id.return_value = "existing_id"
            mock_get_manager.return_value = mock_manager

            result = get_or_create_video_id()

            assert result == "existing_id"

    def test_get_or_create_video_id_from_path(self):
        """Test creating video ID from path"""
        with patch(
            "core.utils.config_utils.get_video_manager"
        ) as mock_get_manager, patch("os.path.exists", return_value=True):
            mock_manager = Mock()
            mock_manager.get_current_video_id.return_value = None
            mock_manager.register_video.return_value = "new_video_id"
            mock_get_manager.return_value = mock_manager

            result = get_or_create_video_id("/path/to/video.mp4")

            assert result == "new_video_id"
            mock_manager.register_video.assert_called_once_with("/path/to/video.mp4")

    def test_get_or_create_video_id_no_path(self):
        """Test error when no current video and no path provided"""
        with patch("core.utils.config_utils.get_video_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_current_video_id.return_value = None
            mock_get_manager.return_value = mock_manager

            with pytest.raises(ValueError, match="No current video found"):
                get_or_create_video_id()

    def test_get_or_create_video_id_invalid_path(self):
        """Test error when path doesn't exist"""
        with patch(
            "core.utils.config_utils.get_video_manager"
        ) as mock_get_manager, patch("os.path.exists", return_value=False):
            mock_manager = Mock()
            mock_manager.get_current_video_id.return_value = None
            mock_get_manager.return_value = mock_manager

            with pytest.raises(ValueError, match="No current video found"):
                get_or_create_video_id("/nonexistent/path.mp4")

    def test_safe_clean_processing_directories_deprecated(self):
        """Test deprecated function warning"""
        with patch("print") as mock_print, patch(
            "core.utils.config_utils.get_storage_paths"
        ) as mock_paths, patch("os.path.exists", return_value=False):
            mock_paths.return_value = {
                "input": "/input",
                "temp": "/temp",
                "output": "/output",
            }

            safe_clean_processing_directories()

            mock_print.assert_called()
            warning_msg = mock_print.call_args[0][0]
            assert "deprecated" in warning_msg.lower()


class TestSecretKeyManagement:
    """Test secret key file-based operations"""

    def test_resolve_keys_path_explicit(self):
        """Test explicit path resolution"""
        result = _resolve_keys_path("/explicit/path/keys.ini")
        assert result == "/explicit/path/keys.ini"

    def test_resolve_keys_path_env_var(self):
        """Test path resolution with environment variable"""
        with patch.dict(os.environ, {"VIDEOLINGO_CONFIG_DIR": "/env/config"}):
            result = _resolve_keys_path(None)
            assert result == "/env/config/keys.ini"

    def test_resolve_keys_path_default(self):
        """Test default path resolution"""
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_keys_path(None)
            assert result == "./keys.ini"

    def test_read_kv_file_nonexistent(self):
        """Test reading non-existent key-value file"""
        result = _read_kv_file("/nonexistent/file.ini")
        assert result == {}

    def test_read_kv_file_success(self):
        """Test successful key-value file reading"""
        content = """# Comment line
key1=value1
key2=value with spaces
# Another comment
key3=value3

empty_line_above=value4"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = _read_kv_file(temp_path)

            expected = {
                "key1": "value1",
                "key2": "value with spaces",
                "key3": "value3",
                "empty_line_above": "value4",
            }
            assert result == expected

        finally:
            os.unlink(temp_path)

    def test_read_kv_file_malformed_lines(self):
        """Test reading file with malformed lines"""
        content = """key1=value1
malformed_line_without_equals
key2=value2
=value_without_key
key3=value3"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = _read_kv_file(temp_path)

            # Should skip malformed lines
            expected = {
                "key1": "value1",
                "key2": "value2",
                "": "value_without_key",  # Empty key
                "key3": "value3",
            }
            assert result == expected

        finally:
            os.unlink(temp_path)

    def test_write_kv_file_new_directory(self):
        """Test writing to new directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "subdir", "keys.ini")
            data = {"key1": "value1", "key2": "value2"}

            _write_kv_file(file_path, data)

            assert os.path.exists(file_path)
            result = _read_kv_file(file_path)
            assert result == data

    def test_write_kv_file_atomic_replacement(self):
        """Test atomic file replacement"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            # Write initial data
            initial_data = {"key1": "value1"}
            _write_kv_file(file_path, initial_data)

            # Update with new data
            new_data = {"key1": "updated1", "key2": "value2"}
            _write_kv_file(file_path, new_data)

            result = _read_kv_file(file_path)
            assert result == new_data

    def test_write_kv_file_permissions_posix(self):
        """Test file permissions on POSIX systems"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            with patch("os.name", "posix"), patch("os.chmod") as mock_chmod:
                _write_kv_file(file_path, {"key": "value"})
                mock_chmod.assert_called_once_with(file_path, 0o600)

    def test_write_kv_file_permissions_windows(self):
        """Test file permissions on non-POSIX systems"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            with patch("os.name", "nt"), patch("os.chmod") as mock_chmod:
                _write_kv_file(file_path, {"key": "value"})
                mock_chmod.assert_not_called()

    def test_save_key_success(self):
        """Test successful key saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            with patch(
                "core.utils.config_utils._resolve_keys_path", return_value=file_path
            ):
                result = save_key("test_key", "test_value")
                assert result is True

                # Verify saved
                saved_data = _read_kv_file(file_path)
                assert saved_data["test_key"] == "test_value"

    def test_save_key_validation(self):
        """Test key saving validation"""
        # Empty namespace
        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            save_key("", "value")

        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            save_key("   ", "value")

        # Non-string namespace
        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            save_key(123, "value")

        # Empty value
        with pytest.raises(ValueError, match="value must be a non-empty string"):
            save_key("key", "")

        # Non-string value
        with pytest.raises(ValueError, match="value must be a non-empty string"):
            save_key("key", 123)

    def test_save_key_exception_handling(self):
        """Test exception handling during save"""
        with patch(
            "core.utils.config_utils._resolve_keys_path",
            return_value="/readonly/keys.ini",
        ), patch(
            "core.utils.config_utils._write_kv_file",
            side_effect=PermissionError("Access denied"),
        ):
            result = save_key("test_key", "test_value")
            assert result is False

    def test_load_secret_key_success(self):
        """Test successful secret key loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            # Setup test data
            _write_kv_file(file_path, {"secret_key": "secret_value"})

            with patch(
                "core.utils.config_utils._resolve_keys_path", return_value=file_path
            ):
                result = load_secret_key("secret_key")
                assert result == "secret_value"

    def test_load_secret_key_not_found(self):
        """Test loading non-existent secret key"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")

            with patch(
                "core.utils.config_utils._resolve_keys_path", return_value=file_path
            ):
                with pytest.raises(KeyError, match="namespace 'nonexistent' not found"):
                    load_secret_key("nonexistent")

    def test_load_secret_key_validation(self):
        """Test secret key loading validation"""
        # Empty namespace
        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            load_secret_key("")

        # Non-string namespace
        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            load_secret_key(123)

    def test_load_key_with_path_compatibility(self):
        """Test backward compatibility function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "keys.ini")
            _write_kv_file(file_path, {"test_key": "test_value"})

            result = load_key_with_path("test_key", file_path)
            assert result == "test_value"


class TestThreadSafety:
    """Test thread safety of configuration operations"""

    def test_yaml_operations_thread_safety(self):
        """Test thread safety of YAML operations"""
        config_content = """
test:
  counter: 0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        results = []
        errors = []

        def worker(thread_id):
            try:
                with patch("core.utils.config_utils.CONFIG_PATH", config_path):
                    for i in range(10):
                        # Concurrent reads and writes
                        value = load_key("test.counter", 0)
                        update_key("test.counter", value + 1)
                        results.append((thread_id, i, value))
                        time.sleep(0.001)  # Small delay

            except Exception as e:
                errors.append((thread_id, str(e)))

        try:
            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)

            # Verify no errors occurred
            assert len(errors) == 0, f"Thread safety errors: {errors}"

        finally:
            os.unlink(config_path)

    def test_key_value_operations_thread_safety(self):
        """Test thread safety of key-value file operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            errors = []

            def worker(thread_id):
                try:
                    file_path = os.path.join(temp_dir, f"keys_{thread_id}.ini")

                    for i in range(10):
                        key = f"thread_{thread_id}_key_{i}"
                        value = f"thread_{thread_id}_value_{i}"

                        with patch(
                            "core.utils.config_utils._resolve_keys_path",
                            return_value=file_path,
                        ):
                            success = save_key(key, value)
                            assert success is True

                            loaded_value = load_secret_key(key)
                            results.append((thread_id, i, loaded_value))

                except Exception as e:
                    errors.append((thread_id, str(e)))

            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)

            # Verify no errors
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            assert len(results) == 50  # 5 threads * 10 operations
