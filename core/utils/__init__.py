# Core utilities module
"""
Centralized utilities for VideoLingo core modules.
This module provides commonly used utilities across the entire application.
"""

# Path and file management
from .path_adapter import PathAdapter, get_path_adapter, get_current_video_paths, with_video_context

# Model constants and paths
from .models import (
    _2_CLEANED_CHUNKS,
    _3_1_SPLIT_BY_NLP,
    _3_2_SPLIT_BY_MEANING,
    _4_1_TERMINOLOGY,
    _4_2_TRANSLATION,
    _5_SPLIT_SUB,
    _5_REMERGED,
    _8_1_AUDIO_TASK,
    _OUTPUT_DIR,
    _AUDIO_DIR,
    _RAW_AUDIO_FILE,
    _VOCAL_AUDIO_FILE,
    _BACKGROUND_AUDIO_FILE,
    _AUDIO_REFERS_DIR,
    _AUDIO_SEGS_DIR,
    _AUDIO_TMP_DIR,
    _get_temp_path,
    _get_output_path,
    TranscriptionData,
    TranslationData,
    AudioData,
)

# Config management - only import what actually exists
from .config_utils import (
    load_key,
    update_key,
    save_key,
    load_all_config,
    update_config,
    get_storage_paths,
    ensure_storage_dirs,
    validate_config_structure,
)

# LLM interaction (lightweight import)
from .ask_gpt import (
    ask_gpt,
)

# Decorators
from .decorator import check_file_exists, except_handler

# Rich console printing
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, track

# Create default console instance
console = Console()


# Path convenience functions
def get_temp_path():
    """Get temp directory path"""
    return _get_temp_path()


def get_output_path():
    """Get output directory path"""
    return _get_output_path()


# Export all
__all__ = [
    # Path variables from models
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
    # Path functions
    "get_temp_path",
    "get_output_path",
    # Data models
    "TranscriptionData",
    "TranslationData",
    "AudioData",
    # Path adapter functions
    "PathAdapter",
    "get_path_adapter",
    "get_current_video_paths",
    "with_video_context",
    # Config management
    "load_key",
    "update_key",
    "save_key",
    "load_all_config",
    "update_config",
    "get_storage_paths",
    "ensure_storage_dirs",
    "validate_config_structure",
    # LLM interaction
    "ask_gpt",
    # Decorators
    "check_file_exists",
    "except_handler",
    # Rich printing
    "rprint",
    "console",
    "Console",
    "Progress",
    "track",
]
