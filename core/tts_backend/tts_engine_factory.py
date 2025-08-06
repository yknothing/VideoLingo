# ------------
# TTS Engine Factory - 策略模式实现
# 解决开闭原则违反问题，支持动态扩展TTS引擎
# ------------

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
# 移除未使用的导入


class TTSEngine(ABC):
    """TTS引擎抽象基类"""
    
    @abstractmethod
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        """生成TTS音频"""
        pass
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        """引擎名称"""
        pass


class OpenAITTSEngine(TTSEngine):
    """OpenAI TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .openai_tts import openai_tts
        openai_tts(text, save_as)
    
    @property
    def engine_name(self) -> str:
        return "openai_tts"


class GPTSoVITSEngine(TTSEngine):
    """GPT-SoVITS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .gpt_sovits_tts import gpt_sovits_tts_for_videolingo
        gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
    
    @property
    def engine_name(self) -> str:
        return "gpt_sovits"


class FishTTSEngine(TTSEngine):
    """Fish TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .fish_tts import fish_tts
        fish_tts(text, save_as)
    
    @property
    def engine_name(self) -> str:
        return "fish_tts"


class AzureTTSEngine(TTSEngine):
    """Azure TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .azure_tts import azure_tts
        azure_tts(text, save_as)
    
    @property
    def engine_name(self) -> str:
        return "azure_tts"


class SiliconFlowFishTTSEngine(TTSEngine):
    """SiliconFlow Fish TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .sf_fishtts import siliconflow_fish_tts_for_videolingo
        siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df)
    
    @property
    def engine_name(self) -> str:
        return "sf_fish_tts"


class EdgeTTSEngine(TTSEngine):
    """Edge TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .edge_tts import edge_tts
        edge_tts(text, save_as)
    
    @property
    def engine_name(self) -> str:
        return "edge_tts"


class CustomTTSEngine(TTSEngine):
    """自定义TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .custom_tts import custom_tts
        custom_tts(text, save_as)
    
    @property
    def engine_name(self) -> str:
        return "custom_tts"


class CosyVoice2Engine(TTSEngine):
    """CosyVoice2引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from .sf_cosyvoice2 import cosyvoice_tts_for_videolingo
        cosyvoice_tts_for_videolingo(text, save_as, number, task_df)
    
    @property
    def engine_name(self) -> str:
        return "sf_cosyvoice2"


class F5TTSEngine(TTSEngine):
    """F5 TTS引擎"""
    
    def generate(self, text: str, save_as: str, number: int = None, task_df=None) -> None:
        from ._302_f5tts import f5_tts_for_videolingo
        f5_tts_for_videolingo(text, save_as, number, task_df)
    
    @property
    def engine_name(self) -> str:
        return "f5tts"


class TTSEngineFactory:
    """TTS引擎工厂类 - 实现策略模式"""
    
    _engines: Dict[str, Type[TTSEngine]] = {}
    _instances: Dict[str, TTSEngine] = {}
    
    @classmethod
    def register_engine(cls, engine_class: Type[TTSEngine]) -> None:
        """注册TTS引擎"""
        engine_instance = engine_class()
        cls._engines[engine_instance.engine_name] = engine_class
        cls._instances[engine_instance.engine_name] = engine_instance
    
    @classmethod
    def create_engine(cls, engine_name: str) -> TTSEngine:
        """创建TTS引擎实例"""
        if engine_name not in cls._instances:
            raise ValueError(f"Unknown TTS engine: {engine_name}. Available engines: {list(cls._engines.keys())}")
        return cls._instances[engine_name]
    
    @classmethod
    def get_available_engines(cls) -> list:
        """获取可用的TTS引擎列表"""
        return list(cls._engines.keys())
    
    @classmethod
    def is_engine_available(cls, engine_name: str) -> bool:
        """检查引擎是否可用"""
        return engine_name in cls._engines


# ------------
# 自动注册所有内置引擎
# ------------

def _register_builtin_engines():
    """注册所有内置TTS引擎"""
    builtin_engines = [
        OpenAITTSEngine,
        GPTSoVITSEngine,
        FishTTSEngine,
        AzureTTSEngine,
        SiliconFlowFishTTSEngine,
        EdgeTTSEngine,
        CustomTTSEngine,
        CosyVoice2Engine,
        F5TTSEngine,
    ]
    
    for engine_class in builtin_engines:
        TTSEngineFactory.register_engine(engine_class)


# 初始化时自动注册
_register_builtin_engines()


# ------------
# 向后兼容的工厂函数
# ------------

def create_tts_engine(engine_name: str) -> TTSEngine:
    """创建TTS引擎 - 向后兼容函数"""
    return TTSEngineFactory.create_engine(engine_name)


def get_available_tts_engines() -> list:
    """获取可用TTS引擎列表 - 向后兼容函数"""
    return TTSEngineFactory.get_available_engines()