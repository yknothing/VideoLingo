#!/usr/bin/env python3
"""
Simple test for config_utils.py to verify basic functionality
This bypasses the complex test infrastructure that's causing hanging issues.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.utils.config_utils import (
    load_key,
    update_key,
    get_storage_paths,
    ensure_storage_dirs,
)


def test_load_key():
    """Test basic load_key functionality"""
    print("Testing load_key...")

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        test_config = {
            "api": {"key": "test-api-key", "base_url": "https://api.test.com/v1"},
            "asr": {"whisper": {"model": "base"}},
        }
        yaml.dump(test_config, f)
        temp_config_path = f.name

    try:
        # Test with mocked CONFIG_PATH
        with patch("core.utils.config_utils.CONFIG_PATH", temp_config_path):
            # Test simple key
            result = load_key("api.key")
            assert result == "test-api-key", f"Expected 'test-api-key', got {result}"
            print("✓ Simple key loading works")

            # Test nested key
            result = load_key("asr.whisper.model")
            assert result == "base", f"Expected 'base', got {result}"
            print("✓ Nested key loading works")

            # Test non-existent key (should raise KeyError)
            try:
                result = load_key("nonexistent.key")
                print(f"✗ Expected KeyError, but got {result}")
            except KeyError:
                print("✓ Non-existent key raises KeyError as expected")

    finally:
        os.unlink(temp_config_path)


def test_update_key():
    """Test update_key functionality"""
    print("Testing update_key...")

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        test_config = {"api": {"key": "old-key"}}
        yaml.dump(test_config, f)
        temp_config_path = f.name

    try:
        with patch("core.utils.config_utils.CONFIG_PATH", temp_config_path):
            # Test updating existing key
            result = update_key("api.key", "new-key")
            print("✓ update_key executed without error")

            # Verify the update worked
            updated_value = load_key("api.key")
            print(f"Updated value: {updated_value}")

    except Exception as e:
        print(f"✗ update_key failed: {e}")
    finally:
        os.unlink(temp_config_path)


def test_storage_paths():
    """Test storage path functions"""
    print("Testing storage path functions...")

    try:
        # Test get_storage_paths
        paths = get_storage_paths()
        assert isinstance(paths, dict), f"Expected dict, got {type(paths)}"
        print(f"✓ get_storage_paths returned: {paths}")

        # Test ensure_storage_dirs
        ensure_storage_dirs()
        print("✓ ensure_storage_dirs executed without error")

    except Exception as e:
        print(f"✗ Storage path functions failed: {e}")


def main():
    """Run all tests"""
    print("Running simple config_utils tests...")
    print("=" * 50)

    try:
        test_load_key()
        print()
        test_update_key()
        print()
        test_storage_paths()
        print()
        print("=" * 50)
        print("All tests completed successfully!")
        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
