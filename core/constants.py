# ------------
# VideoLingo Constants Management
# Unified constants to avoid scattered magic values
# ------------


class AudioConstants:
    """Audio processing related constants"""

    MIN_AUDIO_DURATION = 3.0
    MIN_BASE_DURATION = 1.0
    TIME_TOLERANCE = 0.1
    DURATION_MULTIPLIER = 1.02
    TIME_DIFF_TOLERANCE = 0.6
    MIN_SUBTITLE_DURATION = 2.5
    TIME_EXTENSION_TOLERANCE = 1.5
    SPEED_FACTOR_PRECISION = 0.001
    BASE_SPEED_FACTOR = 1.0
    MAX_ACCEPTABLE_SPEED = 1.2
    MIN_SPEED_THRESHOLD = 0.7
    SILENT_AUDIO_DURATION = 100
    SAMPLE_RATE_DEFAULT = 16000


class RetryConstants:
    """Retry mechanism related constants"""

    DEFAULT_MAX_RETRIES = 3
    GPT_MAX_RETRIES = 5
    AUDIO_SPEED_MAX_RETRIES = 2
    DOWNLOAD_MAX_RETRIES = 3
    DEFAULT_DELAY = 1.0
    SHORT_DELAY = 0.1
    NETWORK_RETRY_DELAY = 0.5


class SubtitleConstants:
    """Subtitle processing related constants"""

    DEFAULT_MAX_LENGTH = 75
    TARGET_MULTIPLIER = 1.2
    MIN_TRIM_DURATION = 3.5
    MAX_MERGE_COUNT = 5
    MIN_MERGE_THRESHOLD = 2


class NetworkConstants:
    """Network related constants"""

    MIN_FREE_SPACE_GB = 1.0
    MIN_DOWNLOAD_SPEED_MBPS = 0.1
    SERVER_STARTUP_TIMEOUT = 50
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 9880


class SimilarityConstants:
    """Similarity and quality related constants"""

    MIN_SIMILARITY_THRESHOLD = 0.9
    HIGH_QUALITY_THRESHOLD = 0.9
    GOOD_QUALITY_THRESHOLD = 0.8
    ACCEPTABLE_QUALITY_THRESHOLD = 0.75


class ProcessingConstants:
    """Processing flow related constants"""

    DEFAULT_BATCH_SIZE = 15
    DEFAULT_SUMMARY_LENGTH = 8000
    DEFAULT_MAX_SPLIT_LENGTH = 20
    WARMUP_SIZE = 5


class FileConstants:
    """File handling related constants"""

    MAX_FILENAME_LENGTH = 200
    MAX_USER_INPUT_LENGTH = 10000
    RUNTIME_SNAPSHOT_FIELD = "runtime_config"
    METADATA_FILENAME = ".metadata.json"
    LOGS_DIRNAME = "logs"


class SecurityConstants:
    """File upload security related constants"""

    # File size limits (in bytes)
    DEFAULT_MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    MIN_FILE_SIZE = 1024  # 1KB
    
    # Filename security
    MAX_SECURE_FILENAME_LENGTH = 100
    ALLOWED_FILENAME_CHARS = r"[\w\-_\.]"
    
    # Content validation
    MAGIC_NUMBER_CHECK_SIZE = 1024  # First 1KB for magic number verification
    
    # Upload timeout and retry
    UPLOAD_TIMEOUT_SECONDS = 300  # 5 minutes
    MAX_UPLOAD_RETRIES = 2
    
    # Virus scanning
    VIRUS_SCAN_TIMEOUT = 60  # 1 minute
    
    # Temporary file cleanup
    TEMP_FILE_CLEANUP_HOURS = 24  # Clean up temp files after 24 hours


class LanguageConstants:
    """Language processing related constants"""

    DURATION_PER_SYLLABLE = {"en": 0.225, "zh": 0.21, "default": 0.22}
    CJK_CHAR_WEIGHT = 1.75
    KOREAN_CHAR_WEIGHT = 1.5
    DEFAULT_CHAR_WEIGHT = 1.0


# ------------
# Configuration contract and defaults
# ------------


class ConfigContract:
    """Configuration contract: centralize keys, defaults and types"""

    SCHEMA_VERSION = "3.0.0"

    K_API_KEY = "api.key"
    K_API_BASE_URL = "api.base_url"
    K_API_MODEL = "api.model"
    K_API_LLM_JSON = "api.llm_support_json"
    K_MAX_WORKERS = "max_workers"
    K_TARGET_LANGUAGE = "target_language"
    K_DISPLAY_LANGUAGE = "display_language"
    K_YT_RESOLUTION = "youtube_resolution"
    K_DEMUCS = "demucs"
    K_BURN_SUBS = "burn_subtitles"
    K_FFMPEG_GPU = "ffmpeg_gpu"
    K_WHISPER_MODEL = "whisper.model"
    K_WHISPER_LANGUAGE = "whisper.language"
    K_WHISPER_RUNTIME = "whisper.runtime"
    K_TTS_METHOD = "tts_method"
    K_SUBTITLE_MAX_LEN = "subtitle.max_length"
    K_SUBTITLE_TARGET_MULTI = "subtitle.target_multiplier"
    K_SUMMARY_LENGTH = "summary_length"
    K_MAX_SPLIT_LENGTH = "max_split_length"
    K_REFLECT_TRANSLATE = "reflect_translate"
    K_PAUSE_BEFORE_TRANSLATE = "pause_before_translate"
    K_BATCH_TRANSLATE_SIZE = "batch_translate_size"
    K_SPEED_MIN = "speed_factor.min"
    K_SPEED_ACCEPT = "speed_factor.accept"
    K_SPEED_MAX = "speed_factor.max"
    K_MIN_SUBTITLE_DURATION = "min_subtitle_duration"
    K_MIN_TRIM_DURATION = "min_trim_duration"
    K_TOLERANCE = "tolerance"
    
    # Security configuration keys
    K_SECURITY_ENABLE_UPLOAD_VALIDATION = "security.enable_upload_validation"
    K_SECURITY_MAX_UPLOAD_SIZE = "security.max_upload_size"
    K_SECURITY_ENABLE_VIRUS_SCAN = "security.enable_virus_scan"
    K_SECURITY_ENABLE_CONTENT_VALIDATION = "security.enable_content_validation"
    K_SECURITY_ENABLE_LOGGING = "security.enable_logging"
    K_SECURITY_QUARANTINE_SUSPICIOUS_FILES = "security.quarantine_suspicious_files"

    SECRET_KEYS = {
        "api.key",
        "whisper.whisperX_302_api_key",
        "whisper.elevenlabs_api_key",
        "sf_fish_tts.api_key",
        "openai_tts.api_key",
        "azure_tts.api_key",
        "sf_cosyvoice2.api_key",
        "f5tts.302_api",
    }

    DEFAULTS = {
        K_API_KEY: "",
        K_API_BASE_URL: "https://openrouter.ai/api/v1",
        K_API_MODEL: "google/gemini-2.5-flash-preview-05-20",
        K_API_LLM_JSON: True,
        K_MAX_WORKERS: 2,
        K_TARGET_LANGUAGE: "简体中文",
        K_DISPLAY_LANGUAGE: "zh-CN",
        K_YT_RESOLUTION: "1080",
        K_DEMUCS: True,
        K_BURN_SUBS: True,
        K_FFMPEG_GPU: False,
        K_WHISPER_MODEL: "large-v3",
        K_WHISPER_LANGUAGE: "en",
        K_WHISPER_RUNTIME: "local",
        K_TTS_METHOD: "openai_tts",
        K_SUBTITLE_MAX_LEN: SubtitleConstants.DEFAULT_MAX_LENGTH,
        K_SUBTITLE_TARGET_MULTI: SubtitleConstants.TARGET_MULTIPLIER,
        K_SUMMARY_LENGTH: ProcessingConstants.DEFAULT_SUMMARY_LENGTH,
        K_MAX_SPLIT_LENGTH: ProcessingConstants.DEFAULT_MAX_SPLIT_LENGTH,
        K_REFLECT_TRANSLATE: True,
        K_PAUSE_BEFORE_TRANSLATE: False,
        K_BATCH_TRANSLATE_SIZE: ProcessingConstants.DEFAULT_BATCH_SIZE,
        K_SPEED_MIN: 1.0,
        K_SPEED_ACCEPT: AudioConstants.MAX_ACCEPTABLE_SPEED,
        K_SPEED_MAX: 1.4,
        K_MIN_SUBTITLE_DURATION: AudioConstants.MIN_SUBTITLE_DURATION,
        K_MIN_TRIM_DURATION: SubtitleConstants.MIN_TRIM_DURATION,
        K_TOLERANCE: AudioConstants.TIME_EXTENSION_TOLERANCE,
        # -------------
        # File upload defaults (extensions) - 无点小写格式，与代码逻辑一致
        # -------------
        "allowed_video_formats": ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
        "allowed_audio_formats": ["mp3", "wav", "m4a", "flac", "aac", "ogg"],
        
        # -------------
        # Security defaults
        # -------------
        K_SECURITY_ENABLE_UPLOAD_VALIDATION: True,
        K_SECURITY_MAX_UPLOAD_SIZE: SecurityConstants.DEFAULT_MAX_UPLOAD_SIZE,
        K_SECURITY_ENABLE_VIRUS_SCAN: False,  # Disabled by default, requires external scanner
        K_SECURITY_ENABLE_CONTENT_VALIDATION: True,
        K_SECURITY_ENABLE_LOGGING: True,
        K_SECURITY_QUARANTINE_SUSPICIOUS_FILES: False,
    }

    TYPES = {
        K_API_LLM_JSON: bool,
        K_MAX_WORKERS: int,
        K_DEMUCS: bool,
        K_BURN_SUBS: bool,
        K_FFMPEG_GPU: bool,
        K_REFLECT_TRANSLATE: bool,
        K_PAUSE_BEFORE_TRANSLATE: bool,
        K_SUBTITLE_MAX_LEN: int,
        K_SUBTITLE_TARGET_MULTI: float,
        K_SUMMARY_LENGTH: int,
        K_MAX_SPLIT_LENGTH: int,
        K_BATCH_TRANSLATE_SIZE: int,
        K_SPEED_MIN: float,
        K_SPEED_ACCEPT: float,
        K_SPEED_MAX: float,
        K_MIN_SUBTITLE_DURATION: float,
        K_MIN_TRIM_DURATION: float,
        K_TOLERANCE: float,
        # Security types
        K_SECURITY_ENABLE_UPLOAD_VALIDATION: bool,
        K_SECURITY_MAX_UPLOAD_SIZE: int,
        K_SECURITY_ENABLE_VIRUS_SCAN: bool,
        K_SECURITY_ENABLE_CONTENT_VALIDATION: bool,
        K_SECURITY_ENABLE_LOGGING: bool,
        K_SECURITY_QUARANTINE_SUSPICIOUS_FILES: bool,
    }


# ------------
# Backward compatible constant exports
# ------------

MIN_AUDIO_DURATION = AudioConstants.MIN_AUDIO_DURATION
TIME_TOLERANCE = AudioConstants.TIME_TOLERANCE
DURATION_MULTIPLIER = AudioConstants.DURATION_MULTIPLIER
TIME_DIFF_TOLERANCE = AudioConstants.TIME_DIFF_TOLERANCE
SPEED_FACTOR_PRECISION = AudioConstants.SPEED_FACTOR_PRECISION
DEFAULT_MAX_RETRIES = RetryConstants.DEFAULT_MAX_RETRIES
GPT_MAX_RETRIES = RetryConstants.GPT_MAX_RETRIES
DEFAULT_DELAY = RetryConstants.DEFAULT_DELAY
MAX_MERGE_COUNT = SubtitleConstants.MAX_MERGE_COUNT
MIN_MERGE_THRESHOLD = SubtitleConstants.MIN_MERGE_THRESHOLD
MIN_FREE_SPACE_GB = NetworkConstants.MIN_FREE_SPACE_GB
MIN_DOWNLOAD_SPEED_MBPS = NetworkConstants.MIN_DOWNLOAD_SPEED_MBPS
MIN_SIMILARITY_THRESHOLD = SimilarityConstants.MIN_SIMILARITY_THRESHOLD
