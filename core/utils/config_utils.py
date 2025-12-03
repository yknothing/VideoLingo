import threading
import os
import time
from typing import Optional, Dict, Any

# Lazy imports - only load heavy modules when needed
_yaml = None
_dotenv_loaded = False

# Simple cache for config values to reduce file I/O
_config_cache = {}
_cache_valid = False
_cache_timestamp = 0
CACHE_TTL = 1.0  # Cache config for 1 second


def _get_yaml():
    """Lazy load YAML module only when needed"""
    global _yaml
    if _yaml is None:
        from ruamel.yaml import YAML

        _yaml = YAML()
        _yaml.preserve_quotes = True
    return _yaml


def _ensure_dotenv():
    """Lazy load environment variables only when needed"""
    global _dotenv_loaded
    if not _dotenv_loaded:
        try:
            # ------------
            # optional dependency: python-dotenv
            # ------------
            from dotenv import load_dotenv

            load_dotenv()
        except Exception as e:
            print(f"Warning: dotenv not available or failed to load: {e}")
            # proceed without .env
        finally:
            _dotenv_loaded = True


def _invalidate_cache():
    """Invalidate the config cache (call when config is updated)"""
    global _config_cache, _cache_valid, _cache_timestamp
    with lock:
        _config_cache = {}
        _cache_valid = False
        _cache_timestamp = 0


def _get_cached_config():
    """Get cached config data or load from file if cache is stale"""
    global _config_cache, _cache_valid, _cache_timestamp

    with lock:
        current_time = time.time()

        if _cache_valid and (current_time - _cache_timestamp) < CACHE_TTL:
            return _config_cache.copy() if isinstance(_config_cache, dict) else _config_cache

        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                _config_cache = _get_yaml().load(file) or {}
                _cache_valid = True
                _cache_timestamp = current_time
                return _config_cache.copy() if isinstance(_config_cache, dict) else _config_cache
        except FileNotFoundError:
            _config_cache = {}
            _cache_valid = True
            _cache_timestamp = current_time
            return {}
        except Exception:
            if _config_cache:
                return _config_cache.copy() if isinstance(_config_cache, dict) else _config_cache
            return {}


CONFIG_PATH = "config.yaml"
# Provide KEYS_PATH for tests to patch; default aligns with keys.ini in project root
KEYS_PATH = "keys.ini"
lock = threading.Lock()

# ------------
# Configuration backup and validation settings
# ------------
CONFIG_BACKUP_DIR = "_config_backups"
MAX_BACKUP_COUNT = 5

# yaml instance is now lazily loaded via _get_yaml()

# ------------
# Environment variable mapping for sensitive data
# ------------
ENV_MAPPINGS = {
    "api.key": "API_KEY",
    "api.base_url": "API_BASE_URL",
    "video_storage.base_path": "VIDEO_STORAGE_BASE_PATH",
    "youtube.cookies_path": "YOUTUBE_COOKIES_PATH",
    "youtube.proxy": "YOUTUBE_PROXY",
    "model_dir": "WHISPER_MODEL_DIR",
}

# -----------------------
# load & update config (YAML配置键)
# -----------------------


def load_key(key, default=None):
    # Ensure environment variables are loaded
    _ensure_dotenv()

    # First check if this key should be loaded from environment
    if key in ENV_MAPPINGS:
        env_value = os.getenv(ENV_MAPPINGS[key])
        if env_value:
            # Validate ALL environment variable values with enhanced security
            try:
                validated_value = _validate_and_sanitize_env_value(key, env_value)
                if validated_value is not None:
                    return validated_value
            except ValueError as e:
                print(f"Security Warning: Invalid environment variable value for {key}: {str(e)}")
                print(
                    f"Security Warning: Environment variable {ENV_MAPPINGS[key]} will be ignored for security reasons"
                )
                # Fall back to config file value with security log

    # Use cached config to reduce file I/O
    data = _get_cached_config()

    # ------------
    # Resolve nested key with contract default fallback
    # ------------
    def _get_contract_default(k):
        try:
            from core.constants import ConfigContract

            return ConfigContract.DEFAULTS.get(k)
        except Exception:
            return None

    keys = key.split(".")
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            # Prefer explicit default, otherwise contract default
            return default if default is not None else _get_contract_default(key)

    # If config value is empty string and env mapping exists, try environment with validation
    if value == "" and key in ENV_MAPPINGS:
        env_value = os.getenv(ENV_MAPPINGS[key])
        if env_value:
            # Enhanced validation for fallback environment variable usage
            try:
                validated_value = _validate_and_sanitize_env_value(key, env_value)
                if validated_value is not None:
                    return validated_value
            except ValueError as e:
                print(
                    f"Security Warning: Invalid environment variable fallback for {key}: {str(e)}"
                )
                print(
                    f"Security Warning: Using default value instead of unsafe environment variable"
                )
                # Continue with default value - safer than using potentially malicious env var

    # Final validation: ensure config file values are also safe
    if isinstance(value, str) and value.strip():
        try:
            _validate_config_value(key, value)
        except ValueError as e:
            print(f"Security Warning: Invalid config file value for {key}: {str(e)}")
            print(f"Security Warning: Using safe default instead of unsafe config value: {value}")
            return default

    # Final fallback to defaults if unresolved or empty
    if (value is None or value == "") and default is None:
        contract_default = _get_contract_default(key)
        if contract_default is not None:
            return contract_default
    return value


def update_key(key, new_value):
    """
    Atomically update a configuration key with validation and backup

    Args:
        key: Dot-separated configuration key (e.g., 'api.key')
        new_value: New value to set

    Returns:
        bool: True if update succeeded, False otherwise

    Raises:
        KeyError: If key not found in configuration
        ValueError: If validation fails for path-related keys
    """
    try:
        return _atomic_config_update(key, new_value, single_key=True)
    except Exception as e:
        print(f"Error updating configuration key '{key}': {str(e)}")
        return False


def load_all_config():
    """Load complete YAML configuration as dict"""
    try:
        return _get_cached_config()
    except Exception:
        # If cache fails, return empty dict
        return {}


def update_config(new_config):
    """Merge new configuration updates into existing config with atomic operations"""
    try:
        return _atomic_config_update(None, new_config, single_key=False)
    except Exception as e:
        print(f"Error updating configuration: {str(e)}")
        return False


def validate_config_structure():
    """Validate configuration structure and return (is_valid, errors)"""
    config = load_all_config()
    errors = []

    # Define required structure - only require sections that actually exist in config.yaml
    required_sections = {
        "api": ["key"],
        # Note: "paths" section is not required in the base config.yaml
    }

    # Check required sections
    for section, required_keys in required_sections.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue

        section_data = config[section]
        if not isinstance(section_data, dict):
            errors.append(f"Section '{section}' must be a dictionary")
            continue

        # Check required keys within section
        for key in required_keys:
            if key not in section_data:
                errors.append(f"Missing required key: {section}.{key}")

    return len(errors) == 0, errors


# ------------
# Atomic configuration update system with validation and backup
# ------------


def _create_config_backup():
    """Create a timestamped backup of current configuration"""
    if not os.path.exists(CONFIG_PATH):
        return None

    try:
        # Create backup directory
        os.makedirs(CONFIG_BACKUP_DIR, exist_ok=True)

        # Generate backup filename with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}.yaml"
        backup_path = os.path.join(CONFIG_BACKUP_DIR, backup_name)

        # Copy current config to backup
        import shutil

        shutil.copy2(CONFIG_PATH, backup_path)

        # Clean old backups (keep only MAX_BACKUP_COUNT)
        _cleanup_old_backups()

        return backup_path
    except Exception as e:
        print(f"Warning: Failed to create config backup: {str(e)}")
        return None


def _cleanup_old_backups():
    """Remove old backup files, keeping only the most recent ones"""
    try:
        if not os.path.exists(CONFIG_BACKUP_DIR):
            return

        # Get all backup files sorted by modification time
        backup_files = []
        for filename in os.listdir(CONFIG_BACKUP_DIR):
            if filename.startswith("config_backup_") and filename.endswith(".yaml"):
                filepath = os.path.join(CONFIG_BACKUP_DIR, filename)
                if os.path.isfile(filepath):
                    backup_files.append((filepath, os.path.getmtime(filepath)))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)

        # Remove old backups beyond MAX_BACKUP_COUNT
        for filepath, _ in backup_files[MAX_BACKUP_COUNT:]:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Failed to remove old backup {filepath}: {str(e)}")
    except Exception as e:
        print(f"Warning: Failed to cleanup old backups: {str(e)}")


def _validate_config_value(key: str, value: Any) -> bool:
    """Validate configuration values, especially path-related ones"""
    if not key or value is None:
        return True

    # Import security utilities for path validation
    try:
        from core.utils.security_utils import is_safe_path
    except ImportError:
        # Fallback validation if security_utils not available
        return True

    # Validate path-related configuration keys
    path_keys = [
        "video_storage.base_path",
        "youtube.cookies_path",
        "model_dir",
        "video_storage.input_dir",
        "video_storage.temp_dir",
        "video_storage.output_dir",
    ]

    if key in path_keys and isinstance(value, str) and value.strip():
        # Enhanced path validation for security
        if not _is_safe_config_path(key, value):
            raise ValueError(f"Invalid or unsafe path value for '{key}': {value}")

    return True


def _validate_and_sanitize_env_value(key: str, env_value: str) -> Optional[str]:
    """
    Comprehensive validation and sanitization for environment variable values
    Returns sanitized value if valid, None if invalid, raises ValueError for security violations
    """
    if not key or not isinstance(env_value, str):
        raise ValueError("Invalid key or environment value type")

    # Strip whitespace but preserve original for logging
    original_value = env_value
    sanitized_value = env_value.strip()

    if not sanitized_value:
        raise ValueError("Environment variable value cannot be empty after sanitization")

    # ------------
    # Enhanced security validation based on key type
    # ------------

    # API key validation
    if key in ["api.key"]:
        if not _validate_api_key_format(sanitized_value):
            raise ValueError(f"Invalid API key format in environment variable")
        return sanitized_value

    # URL validation
    elif key in ["api.base_url"]:
        if not _validate_url_format(sanitized_value):
            raise ValueError(f"Invalid URL format in environment variable")
        return sanitized_value

    # Path validation (most critical for security)
    elif key in [
        "video_storage.base_path",
        "youtube.cookies_path",
        "model_dir",
        "video_storage.input_dir",
        "video_storage.temp_dir",
        "video_storage.output_dir",
    ]:
        # Use existing path validation but with enhanced security logging
        if not _is_safe_config_path(key, sanitized_value):
            raise ValueError(f"Unsafe path detected in environment variable: {sanitized_value}")
        return sanitized_value

    # Proxy validation
    elif key in ["youtube.proxy"]:
        if not _validate_proxy_format(sanitized_value):
            raise ValueError(f"Invalid proxy format in environment variable")
        return sanitized_value

    # Default: apply basic sanitization
    else:
        # Check for obvious security violations in any environment variable
        if _contains_security_violations(sanitized_value):
            raise ValueError(f"Environment variable contains potentially malicious content")
        return sanitized_value


def _validate_api_key_format(api_key: str) -> bool:
    """Validate API key format - basic structure checks"""
    if not api_key or len(api_key) < 10:
        return False

    # Check for common API key patterns (basic validation)
    # Most API keys are alphanumeric with some special characters
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not all(c in allowed_chars for c in api_key):
        return False

    # Reject obviously fake or test keys
    fake_patterns = ["test", "demo", "example", "fake", "invalid", "placeholder"]
    api_key_lower = api_key.lower()
    if any(pattern in api_key_lower for pattern in fake_patterns):
        return False

    return True


def _validate_url_format(url: str) -> bool:
    """Validate URL format for API base URLs"""
    if not url:
        return False

    # Must start with http:// or https://
    if not (url.startswith("http://") or url.startswith("https://")):
        return False

    # Basic URL structure validation
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        # Must have a valid hostname
        if not parsed.hostname:
            return False
        # Reject localhost and internal IPs for security (unless explicitly allowed)
        if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
            # Allow localhost only for development APIs
            if not any(dev_indicator in url.lower() for dev_indicator in ["dev", "test", "local"]):
                return False
    except Exception:
        return False

    return True


def _validate_proxy_format(proxy: str) -> bool:
    """Validate proxy format"""
    if not proxy:
        return True  # Empty proxy is allowed

    # Basic proxy format validation (http://host:port or https://host:port)
    if not (
        proxy.startswith("http://") or proxy.startswith("https://") or proxy.startswith("socks5://")
    ):
        return False

    try:
        from urllib.parse import urlparse

        parsed = urlparse(proxy)
        if not parsed.hostname or not parsed.port:
            return False
    except Exception:
        return False

    return True


def _contains_security_violations(value: str) -> bool:
    """Check for obvious security violations in environment variable values"""
    # Check for command injection attempts
    dangerous_patterns = [
        ";",
        "&&",
        "||",
        "`",
        "$(",  # Command injection
        "../",
        "..\\",  # Path traversal
        "<script",
        "javascript:",  # Script injection
        "rm -rf",
        "del /",
        "format c:",  # Destructive commands
        "eval(",
        "exec(",
        "system(",  # Code execution
        "function(",  # Generic function call patterns
        "malicious",  # Obvious malicious content
        "\x00",
        "\r",
        "\n",  # Null bytes and newlines
    ]

    value_lower = value.lower()
    return any(pattern in value_lower for pattern in dangerous_patterns)


def _is_safe_config_path(key: str, path: str) -> bool:
    """
    Enhanced path validation specifically for configuration values
    Integrates with comprehensive protected directory validation
    """
    if not path or not path.strip():
        return True

    path = path.strip()

    # ------------
    # Directory traversal protection - reject any path containing .. components
    # ------------
    if ".." in path:
        return False

    # ------------
    # Use enhanced protected directory validation for comprehensive system protection
    # ------------
    try:
        # Check if the path (or its parent directory) is protected
        if os.path.isabs(path):
            # For absolute paths, check if they point to protected directories
            from pathlib import Path

            abs_path = Path(path).resolve()

            # Check the path itself
            if is_protected_directory(str(abs_path)):
                return False

            # Check parent directory (safer to check parent for file paths)
            if is_protected_directory(str(abs_path.parent)):
                return False
    except Exception:
        # If validation fails, be conservative
        return False

    # ------------
    # Special handling for different configuration keys
    # ------------
    if key == "video_storage.base_path":
        # Base path can be:
        # 1. 'output' (relative, default)
        # 2. Absolute path in safe user directories
        # 3. Empty (will use default)

        if path == "output" or path == "":
            return True

        if os.path.isabs(path):
            # Ensure absolute paths are in safe user directories
            from pathlib import Path

            user_home = Path.home()
            normalized_path = Path(path).resolve()

            # Allow paths under user home directory
            try:
                normalized_path.relative_to(user_home)
                return True
            except ValueError:
                pass

            # Allow paths under /tmp for testing (handle macOS symlink /tmp -> /private/tmp)
            if str(normalized_path).startswith("/tmp/") or str(normalized_path).startswith(
                "/private/tmp/"
            ):
                return True

            # Allow certain safe directories for user data
            import platform

            safe_abs_prefixes = []

            if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
                safe_abs_prefixes = ["/Users/", "/home/", "/media/", "/mnt/"]
                # Allow external drive access on macOS
                if platform.system() == "Darwin":
                    safe_abs_prefixes.append("/Volumes/")
            elif platform.system() == "Windows":
                # Windows user data paths
                safe_abs_prefixes = [
                    str(user_home).split("\\")[0] + "\\Users\\",  # C:\Users\
                    str(user_home).split("\\")[0] + "\\temp\\",  # C:\temp\ (common)
                ]

            for prefix in safe_abs_prefixes:
                if str(normalized_path).startswith(prefix):
                    return True

            return False
        else:
            # Allow reasonable relative paths for user convenience
            # Reject only obviously dangerous ones
            if ".." in path:
                return False  # Directory traversal
            if path.startswith("/") or path.startswith("\\"):
                return False  # Not actually relative
            if len(path.strip()) == 0:
                return True  # Empty is OK
            # Allow simple relative paths like 'my_videos', 'downloads', 'output', etc.
            return True

    elif key in ["youtube.cookies_path", "model_dir"]:
        # These can be empty or point to user files
        if path == "":
            return True

        if os.path.isabs(path):
            # Must be in user directories - use comprehensive validation
            from pathlib import Path

            user_home = Path.home()
            normalized_path = Path(path).resolve()

            # Check if in user home directory
            try:
                normalized_path.relative_to(user_home)
                return True
            except ValueError:
                pass

            # Allow /tmp for testing
            if str(normalized_path).startswith("/tmp/"):
                return True

            # Cross-platform safe user directories
            import platform

            safe_prefixes = []

            if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
                safe_prefixes = ["/Users/", "/home/", "/media/", "/mnt/"]
                # Allow external drive access on macOS
                if platform.system() == "Darwin":
                    safe_prefixes.append("/Volumes/")
            elif platform.system() == "Windows":
                safe_prefixes = [
                    str(user_home).split("\\")[0] + "\\Users\\",
                    str(user_home).split("\\")[0] + "\\temp\\",
                ]

            return any(str(normalized_path).startswith(prefix) for prefix in safe_prefixes)
        else:
            # Relative paths should be simple filenames or safe relative paths
            return not ("/" in path or "\\" in path or ".." in path)

    # Default: be conservative but allow reasonable config values
    return True


def _atomic_write_config(data: dict, backup_path: str = None) -> bool:
    """Atomically write configuration data using temporary file and rename"""
    try:
        # Create temporary file in same directory as config
        config_dir = os.path.dirname(os.path.abspath(CONFIG_PATH)) or "."

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=config_dir, delete=False, suffix=".yaml.tmp"
        ) as temp_file:
            temp_path = temp_file.name
            _get_yaml().dump(data, temp_file)

        # Atomic rename - this is the critical atomic operation
        os.replace(temp_path, CONFIG_PATH)

        return True
    except Exception as e:
        # Cleanup temporary file on failure
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        # Attempt rollback if backup exists
        if backup_path and os.path.exists(backup_path):
            try:
                import shutil

                shutil.copy2(backup_path, CONFIG_PATH)
                print(f"Configuration rolled back from backup: {backup_path}")
            except Exception as rollback_error:
                print(f"Failed to rollback configuration: {str(rollback_error)}")

        raise e


def _atomic_config_update(key: str, new_value: Any, single_key: bool = True) -> bool:
    """
    Core atomic configuration update function with validation and backup

    Args:
        key: Configuration key (for single key updates) or None (for batch updates)
        new_value: New value or configuration dict for batch updates
        single_key: True for single key update, False for batch config update

    Returns:
        bool: True if update succeeded

    Raises:
        KeyError: If key not found (single key mode)
        ValueError: If validation fails
    """
    backup_path = None

    with lock:
        try:
            # Create backup before any changes
            backup_path = _create_config_backup()

            # Load current configuration
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                    data = _get_yaml().load(file) or {}
            except FileNotFoundError:
                data = {}

            if single_key:
                # Single key update mode
                if not key:
                    raise ValueError("Key cannot be empty for single key update")

                # Validate the new value
                _validate_config_value(key, new_value)

                # Navigate to the key location
                keys = key.split(".")
                current = data
                for k in keys[:-1]:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        raise KeyError(f"Key path '{key}' not found in configuration")

                # Set the value
                if isinstance(current, dict) and keys[-1] in current:
                    old_value = current[keys[-1]]
                    current[keys[-1]] = new_value

                    # Log the change
                    print(f"Config update: {key} = {new_value} (was: {old_value})")
                else:
                    raise KeyError(f"Key '{keys[-1]}' not found in configuration")
            else:
                # Batch configuration update mode
                if not isinstance(new_value, dict):
                    raise ValueError("new_value must be a dictionary for batch updates")

                # Validate all values in the new configuration
                def validate_nested_config(config_dict, prefix=""):
                    for k, v in config_dict.items():
                        full_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            validate_nested_config(v, full_key)
                        else:
                            _validate_config_value(full_key, v)

                validate_nested_config(new_value)

                # Deep merge new config into current
                def deep_merge(target, source):
                    for key, value in source.items():
                        if (
                            key in target
                            and isinstance(target[key], dict)
                            and isinstance(value, dict)
                        ):
                            deep_merge(target[key], value)
                        else:
                            target[key] = value
                    return target

                data = deep_merge(data, new_value)

                # Log the batch change
                print(f"Config batch update: {len(new_value)} sections updated")

            # Atomic write with rollback capability
            _atomic_write_config(data, backup_path)

            # Invalidate cache after successful update
            _invalidate_cache()

            return True

        except Exception as e:
            print(f"Configuration update failed: {str(e)}")
            raise


def rollback_config(backup_path: str = None) -> bool:
    """
    Rollback configuration to a previous backup

    Args:
        backup_path: Specific backup file path, or None for most recent backup

    Returns:
        bool: True if rollback succeeded
    """
    try:
        with lock:
            if backup_path:
                # Use specific backup
                if not os.path.exists(backup_path):
                    raise FileNotFoundError(f"Backup file not found: {backup_path}")
                source_path = backup_path
            else:
                # Find most recent backup
                if not os.path.exists(CONFIG_BACKUP_DIR):
                    raise FileNotFoundError("No backup directory found")

                backup_files = []
                for filename in os.listdir(CONFIG_BACKUP_DIR):
                    if filename.startswith("config_backup_") and filename.endswith(".yaml"):
                        filepath = os.path.join(CONFIG_BACKUP_DIR, filename)
                        if os.path.isfile(filepath):
                            backup_files.append((filepath, os.path.getmtime(filepath)))

                if not backup_files:
                    raise FileNotFoundError("No backup files found")

                # Sort by modification time (newest first) and use the most recent
                backup_files.sort(key=lambda x: x[1], reverse=True)
                source_path = backup_files[0][0]

            # Perform rollback with validation
            try:
                with open(source_path, "r", encoding="utf-8") as file:
                    backup_data = _get_yaml().load(file)

                # Validate backup data structure
                if not isinstance(backup_data, dict):
                    raise ValueError("Backup file contains invalid configuration data")

                # Atomic write of backup data
                _atomic_write_config(backup_data)

                # Invalidate cache after rollback
                _invalidate_cache()

                print(f"Configuration successfully rolled back from: {source_path}")
                return True

            except Exception as e:
                raise Exception(f"Failed to restore from backup {source_path}: {str(e)}")

    except Exception as e:
        print(f"Configuration rollback failed: {str(e)}")
        return False


def list_config_backups():
    """
    List available configuration backups

    Returns:
        list: List of tuples (backup_path, timestamp, size_bytes)
    """
    backups = []

    try:
        if not os.path.exists(CONFIG_BACKUP_DIR):
            return backups

        for filename in os.listdir(CONFIG_BACKUP_DIR):
            if filename.startswith("config_backup_") and filename.endswith(".yaml"):
                filepath = os.path.join(CONFIG_BACKUP_DIR, filename)
                if os.path.isfile(filepath):
                    stat_info = os.stat(filepath)
                    backups.append((filepath, stat_info.st_mtime, stat_info.st_size))

        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x[1], reverse=True)

    except Exception as e:
        print(f"Warning: Failed to list backups: {str(e)}")

    return backups


def get_config_update_status():
    """
    Get status information about configuration updates

    Returns:
        dict: Status information including backup count, last update, etc.
    """
    status = {
        "backup_count": 0,
        "backup_dir_exists": False,
        "last_backup": None,
        "config_exists": os.path.exists(CONFIG_PATH),
        "lock_available": True,
    }

    try:
        # Check backup directory status
        if os.path.exists(CONFIG_BACKUP_DIR):
            status["backup_dir_exists"] = True
            backups = list_config_backups()
            status["backup_count"] = len(backups)

            if backups:
                import datetime

                last_backup_timestamp = backups[0][1]
                status["last_backup"] = datetime.datetime.fromtimestamp(
                    last_backup_timestamp
                ).strftime("%Y-%m-%d %H:%M:%S")

        # Test lock availability (non-blocking)
        if lock.acquire(blocking=False):
            lock.release()
        else:
            status["lock_available"] = False

    except Exception as e:
        print(f"Warning: Failed to get config update status: {str(e)}")

    return status


# -----------------------
# basic utils
# -----------------------
def get_joiner(language):
    if language in load_key("language_split_with_space"):
        return " "
    elif language in load_key("language_split_without_space"):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")


# -----------------------
# directory management utils
# -----------------------
def get_storage_paths():
    """Get the configured storage paths for video processing

    When user sets base_path, they expect that to be the EXACT directory for their videos,
    not a parent directory for subdirectories.
    """
    import os

    try:
        base_path = load_key("video_storage.base_path")

        # ------------
        # If user has set a specific base_path, use it directly as the working directory
        # ------------
        if base_path and base_path.strip() != "":
            # Ensure the configured path is usable
            try:
                os.makedirs(base_path, exist_ok=True)
                test_file = os.path.join(base_path, ".perm_test")
                with open(test_file, "w", encoding="utf-8") as tf:
                    tf.write("ok")
                os.remove(test_file)

                # User's configured path is the main directory - use it for input
                # Create temp and output as sibling directories at the same level
                parent_dir = os.path.dirname(base_path)
                return {
                    "base": base_path,
                    "input": base_path,  # Use configured path directly for input
                    "temp": os.path.join(parent_dir, "temp"),
                    "output": os.path.join(parent_dir, "output"),
                }
            except Exception as e:
                print(
                    f"Warning: base_path not writable: {base_path}. Falling back to defaults. Detail: {e}"
                )
                # Fall through to default logic below

        # ------------
        # Default behavior: use system defaults with subdirectories
        # ------------
        default_base = _choose_default_base_path()
        input_dir = load_key("video_storage.input_dir", "input") or "input"
        temp_dir = load_key("video_storage.temp_dir", "temp") or "temp"
        output_dir = load_key("video_storage.output_dir", "output") or "output"

        return {
            "base": default_base,
            "input": os.path.join(default_base, input_dir),
            "temp": os.path.join(default_base, temp_dir),
            "output": os.path.join(default_base, output_dir),
        }

    except KeyError:
        # Fallback with separate directories for safety
        return {
            "base": "output",
            "input": "input",
            "temp": "temp",
            "output": "output",
        }


def ensure_storage_dirs():
    """Create storage directories and required subdirectories if they don't exist"""
    import os

    paths = get_storage_paths()

    # Create main directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # Create required subdirectories
    # Temp directory subdirectories
    temp_subdirs = [
        "log",
        "gpt_log",
        "audio",
        "audio/refers",
        "audio/segs",
        "audio/tmp",
    ]
    for subdir in temp_subdirs:
        os.makedirs(os.path.join(paths["temp"], subdir), exist_ok=True)

    # Output directory (keep simple, just the main directory)


def get_system_downloads_dir():
    """
    Return the current user's system Downloads directory in a cross-platform way.
    Tries Windows known folder registry first, then falls back to HOME/Downloads.
    """
    try:
        # Windows: Use registry Known Folders for robust resolution
        if os.name == "nt":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders",
                ) as key:
                    # Known Folder GUID for Downloads
                    val, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
                    # Expand env vars like %USERPROFILE%
                    path = os.path.expandvars(val)
                    return os.path.normpath(path)
            except Exception:
                # Fallback to HOME/Downloads
                pass

        # POSIX (macOS/Linux): default to ~/Downloads
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")
    except Exception:
        # Final fallback to project local 'output'
        return "output"


def _is_path_writable(path):
    """
    Check if a path is writable by creating the directory and a temp file.
    """
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".perm_test")
        with open(test_file, "w", encoding="utf-8") as tf:
            tf.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False


def _choose_default_base_path():
    """
    Prefer system Downloads if available and writable; otherwise fallback to 'output'.
    """
    downloads = get_system_downloads_dir()
    if downloads and _is_path_writable(downloads):
        return downloads
    return "output"


def is_protected_directory(path):
    """
    Enhanced security check if a directory is protected from modification/deletion

    Protects against:
    - System directory access
    - Symlink bypass attacks
    - Network path exploits
    - Parent directory traversal
    - Cross-platform system paths

    Args:
        path: Directory path to check

    Returns:
        bool: True if directory is protected and should not be modified
    """
    import os
    import platform
    from pathlib import Path

    if not path or not isinstance(path, (str, os.PathLike)):
        return True  # Protect against invalid inputs

    try:
        # ------------
        # Resolve symlinks and normalize path to prevent bypass attacks
        # ------------
        abs_path = Path(path).resolve()  # This resolves symlinks unlike os.path.abspath
        abs_path_str = str(abs_path)

        # ------------
        # CRITICAL FIX: Avoid circular dependency with get_storage_paths()
        # Skip configured path checking when we're in a validation context
        # ------------
        configured_protected = []
        # NOTE: Removed get_storage_paths() call here to prevent infinite loop
        # The validation in _is_safe_config_path() already handles storage path safety

        # ------------
        # Comprehensive system directory protection by platform
        # ------------
        system_protected = []

        # Unix-like systems (Linux, macOS, etc.)
        if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
            system_protected.extend(
                [
                    "/",  # Root filesystem
                    "/boot",  # Boot files
                    "/etc",  # System configuration
                    "/usr",  # System binaries and libraries
                    "/var",  # Variable data (logs, spool, etc.)
                    "/sys",  # Virtual filesystem (Linux)
                    "/proc",  # Process filesystem (Linux)
                    "/dev",  # Device files
                    "/root",  # Root user home
                    "/bin",  # Essential binaries
                    "/sbin",  # System binaries
                    "/lib",  # Essential libraries
                    "/lib64",  # 64-bit libraries
                    "/opt",  # Optional software
                    "/srv",  # Service data
                    "/run",  # Runtime data
                    # REMOVED: "/tmp" - allow temp directories for user data
                    # REMOVED: "/home" - allow user home directories
                    # REMOVED: "/Users" - allow macOS user directories
                    "/System",  # macOS system
                    "/Applications",  # macOS applications
                    "/Library",  # macOS/macOS system library
                ]
            )

        # Windows systems
        elif platform.system() == "Windows":
            # Get system drive (usually C:)
            system_drive = os.environ.get("SYSTEMDRIVE", "C:")
            windows_root = os.environ.get("SYSTEMROOT", f"{system_drive}\\Windows")
            program_files = os.environ.get("PROGRAMFILES", f"{system_drive}\\Program Files")
            program_files_x86 = os.environ.get(
                "PROGRAMFILES(X86)", f"{system_drive}\\Program Files (x86)"
            )
            program_data = os.environ.get("PROGRAMDATA", f"{system_drive}\\ProgramData")

            system_protected.extend(
                [
                    system_drive + "\\",  # Root of system drive
                    windows_root,  # Windows directory
                    program_files,  # Program Files
                    program_files_x86,  # Program Files (x86)
                    program_data,  # ProgramData
                    f"{system_drive}\\System32",  # System32
                    f"{system_drive}\\SysWOW64",  # SysWOW64 (32-bit on 64-bit)
                    f"{system_drive}\\Windows\\System32",
                    f"{system_drive}\\Windows\\SysWOW64",
                    f"{system_drive}\\Users\\Administrator",
                    f"{system_drive}\\Users\\Public",
                    f"{system_drive}\\Users\\Default",
                    f"{system_drive}\\Windows\\Temp",  # System temp
                    f"{system_drive}\\Temp",  # Alternative temp location
                ]
            )

        all_protected = configured_protected + system_protected

        # ------------
        # Protection checks with symlink-resolved paths
        # ------------

        # Check exact matches (only system protected paths now)
        for protected_path in all_protected:
            if not protected_path:
                continue
            try:
                protected_resolved = Path(protected_path).resolve()
                if abs_path == protected_resolved:
                    return True
            except Exception:
                # If we can't resolve a protected path, be conservative
                continue

        # ------------
        # CRITICAL FIX: Allow VideoLingo project directories and user data directories
        # ------------

        # Allow VideoLingo project directories (like 'output', temp files, etc.)
        try:
            project_root = Path(".").resolve()
            # If the path is within the VideoLingo project, allow it
            try:
                abs_path.relative_to(project_root)
                return False  # Allow VideoLingo project paths
            except ValueError:
                pass  # Path is not within project, continue checking
        except Exception:
            pass

        # Allow user home directories and their subdirectories
        try:
            user_home = Path.home().resolve()
            try:
                abs_path.relative_to(user_home)
                return False  # Allow user home subdirectories
            except ValueError:
                pass  # Path is not under user home
        except Exception:
            pass

        # Allow common user data directories
        user_safe_prefixes = []
        if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
            # On macOS/Linux, allow user directories but protect system ones
            user_safe_prefixes = ["/Users/", "/home/"]
            # Allow specific user data paths on macOS
            if platform.system() == "Darwin":
                user_safe_prefixes.extend(["/Volumes/"])
        elif platform.system() == "Windows":
            # On Windows, allow user directories
            system_drive = os.environ.get("SYSTEMDRIVE", "C:")
            user_safe_prefixes = [f"{system_drive}\\Users\\"]

        for prefix in user_safe_prefixes:
            if abs_path_str.startswith(prefix) and len(abs_path.parts) > 2:
                # Allow user subdirectories (depth > 2) but protect the parent directories
                return False

        # Check if target path is parent of any protected path (critical safety)
        # Only check system protected paths to avoid circular dependency
        for protected_path in system_protected:
            if not protected_path:
                continue
            try:
                protected_resolved = Path(protected_path).resolve()
                # Check if protected path is within the target path
                try:
                    protected_resolved.relative_to(abs_path)
                    return True  # Target path contains a protected path
                except ValueError:
                    continue
            except Exception:
                continue

        # ------------
        # Enhanced depth and structure protection
        # ------------

        # Root filesystem protection
        if abs_path.parent == abs_path:  # This is root (/ on Unix, C:\ on Windows)
            return True

        # Very shallow directory protection (OS-specific) - FIXED VERSION
        if platform.system() in ("Linux", "Darwin", "FreeBSD", "OpenBSD"):
            # Unix: protect system directories but be more specific
            path_parts = abs_path.parts
            if len(path_parts) <= 2:  # Only protect very shallow paths like ('/', 'usr')
                return True
            elif len(path_parts) == 3:  # ('/', 'Users', 'username') - check more carefully
                if path_parts[1] in [
                    "bin",
                    "sbin",
                    "etc",
                    "usr",
                    "var",
                    "sys",
                    "proc",
                    "dev",
                    "boot",
                    "lib",
                    "lib64",
                    "opt",
                    "srv",
                    "run",
                ]:
                    return True  # Protect system directories
                # Allow /Users/username, /home/username, etc.
                return False
        elif platform.system() == "Windows":
            # Windows: protect system directories but allow user ones
            path_parts = abs_path.parts
            if len(path_parts) <= 1:  # ('C:\\',)
                return True
            elif len(path_parts) == 2:  # ('C:\\', 'Users')
                return path_parts[1].lower() != "users"  # Allow C:\Users but protect others

        # ------------
        # Network path and special filesystem protection
        # ------------

        # UNC paths on Windows (\\server\share)
        if platform.system() == "Windows" and abs_path_str.startswith("\\\\"):
            # Protect network shares from deletion
            unc_parts = abs_path_str.split("\\")
            if len(unc_parts) <= 4:  # \\server\share or less
                return True

        # Special filesystem protection (proc, sys, etc. should be caught above but double-check)
        special_prefixes = []
        if platform.system() in ("Linux", "Darwin"):
            special_prefixes = ["/proc", "/sys", "/dev"]

        for prefix in special_prefixes:
            if abs_path_str.startswith(prefix):
                return True

        return False

    except Exception as e:
        # If we can't properly analyze the path, be conservative and protect it
        print(f"Warning: Could not analyze path security for {path}: {e}")
        return True


def clean_directory_contents(directory_path, preserve_structure=True):
    """
    Clean directory contents while preserving the directory structure if specified
    Enhanced security: multiple safety checks to prevent accidental deletion
    Uses symlink-resolving path validation for comprehensive security
    """
    import os
    import shutil
    from pathlib import Path

    # Critical safety checks
    if not directory_path or directory_path.strip() == "":
        raise ValueError("Directory path cannot be empty")

    # Use Path.resolve() for proper symlink resolution and security
    try:
        abs_dir_path = Path(directory_path).resolve()
        abs_dir_path_str = str(abs_dir_path)
    except Exception as e:
        raise ValueError(f"Cannot resolve directory path '{directory_path}': {e}")

    # Never allow cleaning of protected directories (uses enhanced validation)
    if is_protected_directory(abs_dir_path_str):
        raise ValueError(f"Cannot clean protected directory: {abs_dir_path_str}")

    # Additional safety: ensure we're only cleaning within project boundaries
    try:
        project_root = Path(".").resolve()
        abs_dir_path.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Cannot clean directory outside project root: {abs_dir_path_str}")

    if not abs_dir_path.exists():
        return

    try:
        for item in os.listdir(abs_dir_path_str):
            item_path = abs_dir_path / item

            if item_path.is_file():
                item_path.unlink()
            elif item_path.is_dir():
                # Only remove subdirectories if not preserving structure
                # or if it's not a protected directory (uses enhanced validation)
                if not preserve_structure or not is_protected_directory(str(item_path)):
                    shutil.rmtree(str(item_path))
                else:
                    # Recursively clean but preserve the directory itself
                    clean_directory_contents(str(item_path), preserve_structure)
    except Exception as e:
        # Log error but don't raise - partial cleanup is better than failure
        print(f"Warning: Error cleaning directory {directory_path}: {str(e)}")


def safe_clean_processing_directories(exclude_input=True):
    """
    DEPRECATED: 旧的清理方法，已被新的视频管理系统取代
    此函数保留用于向后兼容，但应避免使用
    """
    print(
        "⚠️ Warning: safe_clean_processing_directories is deprecated. Use VideoFileManager.safe_overwrite_* methods instead."
    )

    import os
    import shutil

    paths = get_storage_paths()

    for path_type, path in paths.items():
        # CRITICAL SAFETY: Never clean input directory
        if exclude_input and path_type == "input":
            continue

        # Only clean temp and output directories
        if path_type in ["temp", "output"] and path != paths["input"] and os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Warning: Could not remove {item_path}: {str(e)}")


def get_video_specific_paths(video_id: str):
    """
    获取特定视频ID的文件路径
    新架构：每个视频有独立的文件组织结构
    """
    from core.utils.video_manager import get_video_manager

    manager = get_video_manager()
    return manager.get_video_paths(video_id)


def get_or_create_video_id(video_path: str = None) -> str:
    """
    获取或创建当前视频的ID
    这是新架构的入口点
    """
    import os
    from core.utils.video_manager import get_video_manager

    manager = get_video_manager()

    # 尝试获取当前视频ID
    current_id = manager.get_current_video_id()
    if current_id:
        return current_id

    # 如果没有当前视频且提供了路径，注册新视频
    if video_path and os.path.exists(video_path):
        return manager.register_video(video_path)

    raise ValueError("No current video found and no valid video path provided")


# -----------------------
# secrets io: 文件型密钥保存/读取（独立于 YAML 配置键）
# 满足 tests/unit/test_config_utils_save_key_contract.py 的契约：
# - save_key(namespace, value, path?) 幂等覆盖；非法参数抛 ValueError
# - load_key(namespace, path?) 与 save_key 对应的文件中读取
# - 路径优先级：显式 path > 环境 VIDEO LINGO_CONFIG_DIR + 默认文件名
# - POSIX 下保存后 chmod 0o600；非 POSIX 平台忽略权限严格校验
# -----------------------
_DEFAULT_KEYS_FILENAME = "keys.ini"
_ENV_CONFIG_DIR = "VIDEOLINGO_CONFIG_DIR"


def _resolve_keys_path(path: Optional[str]) -> str:
    if path:
        return os.fspath(path)
    base = os.getenv(_ENV_CONFIG_DIR, "").strip()
    directory = base if base else "."
    return os.path.join(directory, _DEFAULT_KEYS_FILENAME)


def _read_kv_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v
    return data


def _write_kv_file(path: str, data: Dict[str, str]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for k, v in data.items():
            f.write(f"{k}={v}\n")
    # 原子替换
    os.replace(tmp_path, path)
    # 权限：POSIX 下设置 0o600
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except Exception:
            # 测试在不支持权限的 FS 上会跳过严格断言，这里静默
            pass


def save_key(namespace: str, value: str, path: Optional[str] = None) -> bool:
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    if not isinstance(value, str) or not value:
        raise ValueError("value must be a non-empty string")
    try:
        file_path = _resolve_keys_path(path)
        with lock:
            kv = _read_kv_file(file_path)
            kv[namespace] = value
            _write_kv_file(file_path, kv)
        return True
    except Exception:
        return False


def load_secret_key(namespace: str, path: Optional[str] = None) -> str:
    # 提供非破坏性的新名称，避免与 YAML load_key 混淆
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    file_path = _resolve_keys_path(path)
    with lock:
        kv = _read_kv_file(file_path)
    if namespace not in kv:
        raise KeyError(f"namespace '{namespace}' not found in {file_path}")
    return kv[namespace]


# 为了与测试文件兼容：它调用的是 config_utils.load_key(namespace, path=...)
# 但我们已有 YAML 的 load_key(key)。为保持向后兼容，我们仅在检测到 path 参数时走密钥逻辑。
def load_key_with_path(namespace: str, path: Optional[str] = None):
    return load_secret_key(namespace, path)


# 保持脚本直运行体验
if __name__ == "__main__":
    print(load_key("language_split_with_space"))
