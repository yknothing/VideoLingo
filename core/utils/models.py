# ------------------------------------------
# 获取配置的路径
# ------------------------------------------
import os
from .config_utils import get_storage_paths


def _get_paths():
    """Get configured storage paths"""
    return get_storage_paths()


def _get_temp_path():
    """Get temp directory path"""
    return _get_paths()["temp"]


def _get_output_path():
    """Get output directory path"""
    return _get_paths()["output"]


# ------------------------------------------
# 定义中间产出文件 (使用temp目录)
# ------------------------------------------

_2_CLEANED_CHUNKS = os.path.join(_get_temp_path(), "log", "cleaned_chunks.xlsx")
_3_1_SPLIT_BY_NLP = os.path.join(_get_temp_path(), "log", "split_by_nlp.txt")
_3_2_SPLIT_BY_MEANING = os.path.join(_get_temp_path(), "log", "split_by_meaning.txt")
_4_1_TERMINOLOGY = os.path.join(_get_temp_path(), "log", "terminology.json")
_4_2_TRANSLATION = os.path.join(_get_temp_path(), "log", "translation_results.xlsx")
_5_SPLIT_SUB = os.path.join(_get_temp_path(), "log", "translation_results_for_subtitles.xlsx")
_5_REMERGED = os.path.join(_get_temp_path(), "log", "translation_results_remerged.xlsx")

_8_1_AUDIO_TASK = os.path.join(_get_temp_path(), "audio", "tts_tasks.xlsx")


# ------------------------------------------
# 定义音频文件 (使用temp目录)
# ------------------------------------------
_TEMP_DIR = _get_temp_path()
_AUDIO_DIR = os.path.join(_get_temp_path(), "audio")
_RAW_AUDIO_FILE = os.path.join(_get_temp_path(), "audio", "raw.mp3")
_VOCAL_AUDIO_FILE = os.path.join(_get_temp_path(), "audio", "vocal.mp3")
_BACKGROUND_AUDIO_FILE = os.path.join(_get_temp_path(), "audio", "background.mp3")
_AUDIO_REFERS_DIR = os.path.join(_get_temp_path(), "audio", "refers")
_AUDIO_SEGS_DIR = os.path.join(_get_temp_path(), "audio", "segs")
_AUDIO_TMP_DIR = os.path.join(_get_temp_path(), "audio", "tmp")

# ------------------------------------------
# 定义输出文件 (使用output目录)
# ------------------------------------------
_OUTPUT_DIR = _get_output_path()

# ------------------------------------------
# 数据模型类 (为测试兼容性添加)
# ------------------------------------------


class TranscriptionData:
    """Simple data holder for transcription results"""

    def __init__(self, text="", timestamps=None, language=""):
        self.text = text
        self.timestamps = timestamps or []
        self.language = language


class TranslationData:
    """Simple data holder for translation results"""

    def __init__(self, source="", target="", language=""):
        self.source = source
        self.target = target
        self.language = language


class AudioData:
    """Simple data holder for audio information"""

    def __init__(self, path="", duration=0.0, sample_rate=22050):
        self.path = path
        self.duration = duration
        self.sample_rate = sample_rate


# ------------------------------------------
# 导出
# ------------------------------------------

__all__ = [
    "_2_CLEANED_CHUNKS",
    "_3_1_SPLIT_BY_NLP",
    "_3_2_SPLIT_BY_MEANING",
    "_4_1_TERMINOLOGY",
    "_4_2_TRANSLATION",
    "_5_SPLIT_SUB",
    "_5_REMERGED",
    "_8_1_AUDIO_TASK",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_BACKGROUND_AUDIO_FILE",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "TranscriptionData",
    "TranslationData",
    "AudioData",
]
