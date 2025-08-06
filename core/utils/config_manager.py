# ------------
# Configuration Manager - 配置管理封装层
# 统一配置访问，提供类型安全和默认值处理
# ------------

from typing import Any, Dict, List, Optional, Union, TypeVar, Type
from core.utils.config_utils import load_key, update_key
from core.constants import (
    AudioConstants, RetryConstants, SubtitleConstants, 
    NetworkConstants, ProcessingConstants, FileConstants
)

T = TypeVar('T')


class ConfigurationError(Exception):
    """配置相关异常"""
    pass


class ConfigManager:
    """配置管理器 - 提供类型安全的配置访问"""
    
    # 配置键映射和默认值
    _CONFIG_DEFAULTS = {
        # API配置
        'api.key': '',
        'api.base_url': 'https://openrouter.ai/api/v1',
        'api.model': 'google/gemini-2.5-flash-preview-05-20',
        'api.llm_support_json': True,
        'max_workers': 2,
        
        # 语言配置
        'target_language': '简体中文',
        'display_language': 'zh-CN',
        
        # 视频配置
        'youtube_resolution': '1080',
        'demucs': True,
        'burn_subtitles': True,
        'ffmpeg_gpu': False,
        
        # Whisper配置
        'whisper.model': 'large-v3',
        'whisper.language': 'en',
        'whisper.runtime': 'local',
        
        # TTS配置
        'tts_method': 'openai_tts',
        
        # 字幕配置
        'subtitle.max_length': SubtitleConstants.DEFAULT_MAX_LENGTH,
        'subtitle.target_multiplier': SubtitleConstants.TARGET_MULTIPLIER,
        
        # 处理配置
        'summary_length': ProcessingConstants.DEFAULT_SUMMARY_LENGTH,
        'max_split_length': ProcessingConstants.DEFAULT_MAX_SPLIT_LENGTH,
        'reflect_translate': True,
        'pause_before_translate': False,
        'batch_translate_size': ProcessingConstants.DEFAULT_BATCH_SIZE,
        
        # 音频配置
        'speed_factor.min': 1.0,
        'speed_factor.accept': AudioConstants.MAX_ACCEPTABLE_SPEED,
        'speed_factor.max': 1.4,
        'min_subtitle_duration': AudioConstants.MIN_SUBTITLE_DURATION,
        'min_trim_duration': SubtitleConstants.MIN_TRIM_DURATION,
        'tolerance': AudioConstants.TIME_EXTENSION_TOLERANCE,
    }
    
    # 类型映射
    _TYPE_MAPPING = {
        'api.llm_support_json': bool,
        'max_workers': int,
        'demucs': bool,
        'burn_subtitles': bool,
        'ffmpeg_gpu': bool,
        'reflect_translate': bool,
        'pause_before_translate': bool,
        'subtitle.max_length': int,
        'subtitle.target_multiplier': float,
        'summary_length': int,
        'max_split_length': int,
        'batch_translate_size': int,
        'speed_factor.min': float,
        'speed_factor.accept': float,
        'speed_factor.max': float,
        'min_subtitle_duration': float,
        'min_trim_duration': float,
        'tolerance': float,
    }
    
    @classmethod
    def get(cls, key: str, default: Any = None, expected_type: Type[T] = None) -> T:
        """
        获取配置值，提供类型安全和默认值
        
        Args:
            key: 配置键
            default: 默认值（优先级高于内置默认值）
            expected_type: 期望的返回类型
            
        Returns:
            配置值
            
        Raises:
            ConfigurationError: 配置错误
        """
        try:
            # 尝试从配置文件加载
            value = load_key(key)
            
            # 如果值为空或None，使用默认值
            if value is None or value == '':
                if default is not None:
                    value = default
                elif key in cls._CONFIG_DEFAULTS:
                    value = cls._CONFIG_DEFAULTS[key]
                else:
                    raise ConfigurationError(f"No default value found for config key: {key}")
            
            # 类型转换
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
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            
        Raises:
            ConfigurationError: 配置错误
        """
        try:
            # 类型验证
            if key in cls._TYPE_MAPPING:
                value = cls._convert_type(value, cls._TYPE_MAPPING[key], key)
            
            update_key(key, value)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to set config key '{key}': {str(e)}") from e
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """获取API相关配置"""
        return {
            'key': cls.get('api.key', ''),
            'base_url': cls.get('api.base_url', 'https://openrouter.ai/api/v1'),
            'model': cls.get('api.model', 'google/gemini-2.5-flash-preview-05-20'),
            'llm_support_json': cls.get('api.llm_support_json', True, bool),
            'max_workers': cls.get('max_workers', 2, int),
        }
    
    @classmethod
    def get_whisper_config(cls) -> Dict[str, Any]:
        """获取Whisper相关配置"""
        return {
            'model': cls.get('whisper.model', 'large-v3'),
            'language': cls.get('whisper.language', 'en'),
            'runtime': cls.get('whisper.runtime', 'local'),
        }
    
    @classmethod
    def get_subtitle_config(cls) -> Dict[str, Any]:
        """获取字幕相关配置"""
        return {
            'max_length': cls.get('subtitle.max_length', SubtitleConstants.DEFAULT_MAX_LENGTH, int),
            'target_multiplier': cls.get('subtitle.target_multiplier', SubtitleConstants.TARGET_MULTIPLIER, float),
        }
    
    @classmethod
    def get_speed_factor_config(cls) -> Dict[str, float]:
        """获取速度因子配置"""
        return {
            'min': cls.get('speed_factor.min', 1.0, float),
            'accept': cls.get('speed_factor.accept', AudioConstants.MAX_ACCEPTABLE_SPEED, float),
            'max': cls.get('speed_factor.max', 1.4, float),
        }
    
    @classmethod
    def get_audio_config(cls) -> Dict[str, float]:
        """获取音频相关配置"""
        return {
            'min_subtitle_duration': cls.get('min_subtitle_duration', AudioConstants.MIN_SUBTITLE_DURATION, float),
            'min_trim_duration': cls.get('min_trim_duration', SubtitleConstants.MIN_TRIM_DURATION, float),
            'tolerance': cls.get('tolerance', AudioConstants.TIME_EXTENSION_TOLERANCE, float),
        }
    
    @classmethod
    def get_tts_method(cls) -> str:
        """获取TTS方法"""
        return cls.get('tts_method', 'openai_tts', str)
    
    @classmethod
    def get_youtube_resolution(cls) -> str:
        """获取YouTube分辨率"""
        return cls.get('youtube_resolution', '1080', str)
    
    @classmethod
    def get_target_language(cls) -> str:
        """获取目标语言"""
        return cls.get('target_language', '简体中文', str)
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """
        验证配置完整性
        
        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []
        
        # 检查必需的API配置
        api_key = cls.get('api.key', '')
        if not api_key or api_key.lower().startswith('your'):
            errors.append("API key is not configured properly")
        
        # 检查TTS方法有效性
        tts_method = cls.get_tts_method()
        try:
            from core.tts_backend.tts_engine_factory import TTSEngineFactory
            if not TTSEngineFactory.is_engine_available(tts_method):
                available = TTSEngineFactory.get_available_engines()
                errors.append(f"Invalid TTS method '{tts_method}'. Available: {available}")
        except ImportError:
            pass  # 在测试环境中可能无法导入
        
        # 检查数值范围
        max_workers = cls.get('max_workers', 2, int)
        if max_workers < 1 or max_workers > 10:
            errors.append(f"max_workers should be between 1 and 10, got {max_workers}")
        
        speed_config = cls.get_speed_factor_config()
        if speed_config['min'] > speed_config['accept'] or speed_config['accept'] > speed_config['max']:
            errors.append("Speed factor configuration is invalid: min <= accept <= max")
        
        return errors
    
    @staticmethod
    def _convert_type(value: Any, target_type: Type[T], key: str) -> T:
        """类型转换辅助函数"""
        if isinstance(value, target_type):
            return value
        
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif target_type == int:
                return int(float(value))  # 先转float再转int，处理"1.0"这种情况
            elif target_type == float:
                return float(value)
            elif target_type == str:
                return str(value)
            else:
                return target_type(value)
                
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Cannot convert config value '{value}' to {target_type.__name__} for key '{key}'"
            ) from e


# ------------
# 向后兼容的便捷函数
# ------------

def get_config(key: str, default: Any = None, expected_type: Type[T] = None) -> T:
    """获取配置值 - 向后兼容函数"""
    return ConfigManager.get(key, default, expected_type)


def set_config(key: str, value: Any) -> None:
    """设置配置值 - 向后兼容函数"""
    return ConfigManager.set(key, value)


def validate_configuration() -> List[str]:
    """验证配置 - 向后兼容函数"""
    return ConfigManager.validate_config()