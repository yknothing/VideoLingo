# ------------
# Configuration Manager - 配置管理封装层
# 统一配置访问，提供类型安全、默认值与版本迁移、运行期快照
# ------------

from typing import Any, Dict, List, TypeVar, Type
import os
import json
import stat

from core.utils.config_utils import load_key, update_key, load_all_config
from core.constants import (
    AudioConstants, RetryConstants, SubtitleConstants,
    NetworkConstants, ProcessingConstants, FileConstants,
    ConfigContract
)

T = TypeVar('T')


class ConfigurationError(Exception):
    pass


class ConfigManager:
    """配置管理器 - 提供类型安全的配置访问"""

    # 统一从契约常量取默认值与类型
    _CONFIG_DEFAULTS = ConfigContract.DEFAULTS
    _TYPE_MAPPING = ConfigContract.TYPES

    _initialized = False
    _effective_config: Dict[str, Any] = {}

    # ------------
    # lifecycle
    # ------------
    @classmethod
    def initialize(cls) -> None:
        """
        启动时执行：
        - 校验 config.yaml 必须字段/类型
        - 应用默认值并执行简易版本迁移（当前仅填充默认值）
        - 校验本地密钥文件权限（若存在）
        """
        if cls._initialized:
            return
        cfg = load_all_config()
        cls._effective_config = cls._apply_defaults_and_types(cfg)
        errors = cls._validate_required(cls._effective_config)
        if errors:
            raise ConfigurationError("; ".join(errors))
        cls._check_keys_ini_security()
        cls._initialized = True

    @classmethod
    def _apply_defaults_and_types(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, def_val in cls._CONFIG_DEFAULTS.items():
            try:
                val = load_key(key)
            except Exception:
                val = None
            if val is None or val == "":
                val = def_val
            if key in cls._TYPE_MAPPING:
                val = cls._convert_type(val, cls._TYPE_MAPPING[key], key)
            result[key] = val
        return result

    @classmethod
    def _validate_required(cls, cfg: Dict[str, Any]) -> List[str]:
        errors = []
        api_key = cfg.get('api.key', '')
        if isinstance(api_key, str) and api_key.lower().startswith('your'):
            errors.append("api.key is a placeholder; use environment variable or remove placeholder")
        max_workers = cfg.get('max_workers', 2)
        try:
            mw = int(max_workers)
            if not (1 <= mw <= 10):
                errors.append(f"max_workers should be between 1 and 10, got {max_workers}")
        except Exception:
            errors.append(f"max_workers must be int, got {type(max_workers).__name__}")
        try:
            min_v = float(cfg.get('speed_factor.min'))
            acc_v = float(cfg.get('speed_factor.accept'))
            max_v = float(cfg.get('speed_factor.max'))
            if not (min_v <= acc_v <= max_v):
                errors.append("invalid speed_factor: min <= accept <= max")
        except Exception:
            errors.append("invalid speed_factor configuration")
        return errors

    @classmethod
    def _check_keys_ini_security(cls) -> None:
        config_dir = os.getenv('VIDEOLINGO_CONFIG_DIR', '').strip() or '.'
        keys_path = os.path.join(config_dir, 'keys.ini')
        if not os.path.exists(keys_path):
            return
        if os.name == 'posix':
            mode = stat.S_IMODE(os.stat(keys_path).st_mode)
            if mode & 0o077:
                raise ConfigurationError(f"keys.ini permission must be 600, current: {oct(mode)}")

    # ------------
    # CRUD
    # ------------
    @classmethod
    def get(cls, key: str, default: Any = None, expected_type: Type[T] = None) -> T:
        try:
            if not cls._initialized:
                cls.initialize()
            value = cls._effective_config.get(key)
            if value is None or value == '':
                if default is not None:
                    value = default
                elif key in cls._CONFIG_DEFAULTS:
                    value = cls._CONFIG_DEFAULTS[key]
                else:
                    raise ConfigurationError(f"No default value found for config key: {key}")
            if expected_type:
                value = cls._convert_type(value, expected_type, key)
            elif key in cls._TYPE_MAPPING:
                value = cls._convert_type(value, cls._TYPE_MAPPING[key], key)
            return value
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load config key '{key}': {str(e)}") from e

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        try:
            if key in cls._TYPE_MAPPING:
                value = cls._convert_type(value, cls._TYPE_MAPPING[key], key)
            update_key(key, value)
            # 更新内存快照
            cls._effective_config[key] = value
        except Exception as e:
            raise ConfigurationError(f"Failed to set config key '{key}': {str(e)}") from e

    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        return {
            'key': cls.get('api.key', ''),
            'base_url': cls.get('api.base_url', 'https://openrouter.ai/api/v1'),
            'model': cls.get('api.model', 'google/gemini-2.5-flash-preview-05-20'),
            'llm_support_json': cls.get('api.llm_support_json', True, bool),
            'max_workers': cls.get('max_workers', 2, int),
        }

    @classmethod
    def get_whisper_config(cls) -> Dict[str, Any]:
        return {
            'model': cls.get('whisper.model', 'large-v3'),
            'language': cls.get('whisper.language', 'en'),
            'runtime': cls.get('whisper.runtime', 'local'),
        }

    @classmethod
    def get_subtitle_config(cls) -> Dict[str, Any]:
        return {
            'max_length': cls.get('subtitle.max_length', SubtitleConstants.DEFAULT_MAX_LENGTH, int),
            'target_multiplier': cls.get('subtitle.target_multiplier', SubtitleConstants.TARGET_MULTIPLIER, float),
        }

    @classmethod
    def get_speed_factor_config(cls) -> Dict[str, float]:
        return {
            'min': cls.get('speed_factor.min', 1.0, float),
            'accept': cls.get('speed_factor.accept', AudioConstants.MAX_ACCEPTABLE_SPEED, float),
            'max': cls.get('speed_factor.max', 1.4, float),
        }

    @classmethod
    def get_audio_config(cls) -> Dict[str, float]:
        return {
            'min_subtitle_duration': cls.get('min_subtitle_duration', AudioConstants.MIN_SUBTITLE_DURATION, float),
            'min_trim_duration': cls.get('min_trim_duration', SubtitleConstants.MIN_TRIM_DURATION, float),
            'tolerance': cls.get('tolerance', AudioConstants.TIME_EXTENSION_TOLERANCE, float),
        }

    @classmethod
    def get_tts_method(cls) -> str:
        return cls.get('tts_method', 'openai_tts', str)

    @classmethod
    def get_youtube_resolution(cls) -> str:
        return cls.get('youtube_resolution', '1080', str)

    @classmethod
    def get_target_language(cls) -> str:
        return cls.get('target_language', '简体中文', str)

    @classmethod
    def validate_config(cls) -> List[str]:
        if not cls._initialized:
            cls.initialize()
        errors = cls._validate_required(cls._effective_config)
        tts_method = cls.get_tts_method()
        try:
            from core.tts_backend.tts_engine_factory import TTSEngineFactory
            if not TTSEngineFactory.is_engine_available(tts_method):
                available = TTSEngineFactory.get_available_engines()
                errors.append(f"Invalid TTS method '{tts_method}'. Available: {available}")
        except ImportError:
            pass
        return errors

    @classmethod
    def export_runtime_snapshot(cls, video_id: str) -> None:
        try:
            if not cls._initialized:
                cls.initialize()
            snapshot = {}
            for key in cls._CONFIG_DEFAULTS.keys():
                val = cls._effective_config.get(key)
                if key in ConfigContract.SECRET_KEYS:
                    snapshot[key] = "[REDACTED]"
                else:
                    snapshot[key] = val
            from core.utils.video_manager import get_video_manager
            manager = get_video_manager()
            paths = manager.get_video_paths(video_id)
            metadata_file = paths.get('metadata_file')
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f) or {}
                except Exception:
                    metadata = {}
            metadata[FileConstants.RUNTIME_SNAPSHOT_FIELD] = {
                'version': ConfigContract.SCHEMA_VERSION,
                'values': snapshot
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------
    # helpers
    # ------------
    @staticmethod
    def _convert_type(value: Any, target_type: Type[T], key: str) -> T:
        if isinstance(value, target_type):
            return value
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            if target_type == int:
                return int(float(value))
            if target_type == float:
                return float(value)
            if target_type == str:
                return str(value)
            return target_type(value)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Cannot convert config value '{value}' to {target_type.__name__} for key '{key}'"
            ) from e


# ------------
# 向后兼容便捷函数
# ------------
def get_config(key: str, default: Any = None, expected_type: Type[T] = None) -> T:
    return ConfigManager.get(key, default, expected_type)


def set_config(key: str, value: Any) -> None:
    return ConfigManager.set(key, value)


def validate_configuration() -> List[str]:
    return ConfigManager.validate_config()


